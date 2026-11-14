from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

DATOS = Path("data/processed/preparacion/datos_sin_imputar.csv")
INFORME = Path("docs/reports/experimentacion/comparacion_modelos.json")
SEMILLA = 42
PLIEGUES = 5
TERMINOS_TFIDF = 300
CLASES = ("Negativo", "Neutral", "Positivo")
TEXTO = "texto_resena"
NUMERICAS = [
    "edad", "antiguedad_cliente_meses", "importe_compra",
    "tiempo_respuesta_horas", "longitud_resena", "cantidad_palabras_resena",
]
CATEGORICAS = [
    "genero", "ciudad", "categoria_producto", "canal_compra", "plataforma",
    "problema_resuelto", "reclamo_formal",
]
ENTRADAS = ["id_cliente", TEXTO, *NUMERICAS, *CATEGORICAS]


class ImputacionAP1(TransformerMixin, BaseEstimator):
    """Recalcula las medianas para cada subconjunto de ajuste."""

    def fit(self, X: pd.DataFrame, y=None):
        clientes = X.drop_duplicates(subset="id_cliente")
        self.edades_por_ciudad_ = clientes.groupby("ciudad")["edad"].median()
        self.edad_general_ = clientes["edad"].median()
        self.horas_por_categoria_ = X.groupby("categoria_producto")[
            "tiempo_respuesta_horas"
        ].median()
        self.horas_generales_ = X["tiempo_respuesta_horas"].median()
        if not np.isfinite([self.edad_general_, self.horas_generales_]).all():
            raise ValueError("El pliegue no tiene edades o tiempos válidos para imputar.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        datos = X.copy()
        datos["edad"] = (
            datos["edad"].fillna(datos["ciudad"].map(self.edades_por_ciudad_))
            .fillna(self.edad_general_)
        )
        datos["tiempo_respuesta_horas"] = (
            datos["tiempo_respuesta_horas"]
            .fillna(datos["categoria_producto"].map(self.horas_por_categoria_))
            .fillna(self.horas_generales_)
        )
        return datos


def crear_pipeline(modelo) -> Pipeline:
    preprocesador = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("texto", TfidfVectorizer(max_features=TERMINOS_TFIDF), TEXTO),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("imputacion", ImputacionAP1()), ("preprocesado", preprocesador), ("modelo", modelo)]
    )


def leer_entrenamiento(ruta: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    datos = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    requeridas = {*ENTRADAS, "particion", "sentimiento"}
    faltan = sorted(requeridas - set(datos.columns))
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {', '.join(faltan)}.")
    particion = datos["particion"].astype("string").str.strip().str.casefold()
    nombres = ("entrenamiento", "validacion", "prueba")
    if not particion.isin(nombres).all():
        raise ValueError("La columna particion contiene valores distintos de entrenamiento, validacion o prueba.")
    counts = {p: int(particion.eq(p).sum()) for p in nombres}
    if any(n == 0 for n in counts.values()):
        raise ValueError(f"Falta una de las particiones de AP1: {counts}.")
    train = datos.loc[particion.eq("entrenamiento")]
    y = train["sentimiento"].astype("string").str.strip()
    if not y.isin(CLASES).all() or any(y.eq(c).sum() < PLIEGUES for c in CLASES):
        raise ValueError(f"Entrenamiento debe tener al menos {PLIEGUES} casos por clase: {CLASES}.")
    X = train[ENTRADAS].copy()
    for c in NUMERICAS:
        X[c] = pd.to_numeric(X[c], errors="raise").astype(float)
    sin_ausentes = X.drop(columns=["edad", "tiempo_respuesta_horas"])
    if sin_ausentes.isna().any().any():
        columnas = sin_ausentes.columns[sin_ausentes.isna().any()].tolist()
        raise ValueError(f"Hay ausentes en columnas de entrenamiento no previstas: {columnas}.")
    return X, y, counts


def metricas_cv() -> dict:
    medidas = {"accuracy": "accuracy"}
    for nombre, funcion in (
        ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)
    ):
        medidas[f"{nombre}_macro"] = make_scorer(
            funcion, labels=CLASES, average="macro", zero_division=0
        )
        for clase in CLASES:
            medidas[f"{nombre}_{clase.lower()}"] = make_scorer(
                funcion, labels=[clase], average="macro", zero_division=0
            )
    return medidas


def candidatos() -> list:
    return [
        DummyClassifier(strategy="most_frequent"),
        LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=SEMILLA),
        LinearSVC(C=1.0, loss="squared_hinge", max_iter=5000, random_state=SEMILLA),
        KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2),
        DecisionTreeClassifier(criterion="gini", random_state=SEMILLA),
    ]


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def comparar(args: argparse.Namespace) -> None:
    X, y, counts = leer_entrenamiento(args.datos)
    cv = list(StratifiedKFold(n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA).split(X, y))
    medidas = metricas_cv()
    mlflow = None
    if not args.sin_mlflow:
        import mlflow

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            Path("tmp").mkdir(exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///tmp/mlflow_ap2.db")
        mlflow.set_experiment(args.experimento)

    informe = {
        "datos": args.datos.as_posix(),
        "particiones_ap1": counts,
        "clases_en_entrenamiento": {c: int(y.eq(c).sum()) for c in CLASES},
        "pliegues": PLIEGUES,
        "semilla": SEMILLA,
        "terminos_tfidf_max": TERMINOS_TFIDF,
        "versiones": {"pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "modelos": [],
    }
    print(f"Particiones AP1: {counts}", flush=True)
    print(f"Comparación: {PLIEGUES} pliegues solo sobre {len(X):,} reseñas de entrenamiento.")
    print(f"{'Modelo':<25} {'Accuracy':>9} {'F1 macro':>9} {'Recall neg.':>12}", flush=True)

    for modelo in candidatos():
        nombre = type(modelo).__name__
        cv_scores = cross_validate(
            crear_pipeline(modelo), X, y, cv=cv, scoring=medidas, error_score="raise"
        )
        resultados = {
            clave: {
                "media": float(np.mean(cv_scores[f"test_{clave}"])),
                "desviacion": float(np.std(cv_scores[f"test_{clave}"])),
            }
            for clave in medidas
        }
        fila = {"modelo": nombre, "parametros": modelo.get_params(), "cv": resultados}
        if mlflow is not None:
            with mlflow.start_run(run_name=nombre) as run:
                mlflow.log_params(
                    {
                        "modelo": nombre,
                        "pliegues": PLIEGUES,
                        "semilla_cv": SEMILLA,
                        "terminos_tfidf_max": TERMINOS_TFIDF,
                        "n_entrenamiento": len(X),
                        **{f"modelo_{k}": str(v) for k, v in modelo.get_params().items()},
                    }
                )
                mlflow.log_metrics({
                    f"cv_{k}_{estadistico}": v
                    for k, valores in resultados.items()
                    for estadistico, v in valores.items()
                })
                fila["mlflow_run_id"] = run.info.run_id
                with TemporaryDirectory() as tmp:
                    ruta_run = Path(tmp) / "resultado_cv.json"
                    escribir_json(ruta_run, fila)
                    mlflow.log_artifact(str(ruta_run), artifact_path="evaluacion_cv")
        informe["modelos"].append(fila)
        print(
            f"{nombre:<25} {resultados['accuracy']['media']:>9.4f} "
            f"{resultados['f1_macro']['media']:>9.4f} "
            f"{resultados['recall_negativo']['media']:>12.4f}",
            flush=True,
        )

    escribir_json(args.salida, informe)
    print(f"Informe: {args.salida}")
    if mlflow is not None:
        print(f"MLflow: 5 ejecuciones en el experimento «{args.experimento}».")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=DATOS)
    parser.add_argument("--salida", type=Path, default=INFORME)
    parser.add_argument("--experimento", default="AP2 - Comparacion de modelos")
    parser.add_argument("--sin-mlflow", action="store_true", help="Ejecuta la comparación sin registrar en MLflow.")
    args = parser.parse_args()
    try:
        comparar(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
