from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.experimentacion import (
    CLASES,
    DATOS,
    ENTRADAS,
    NUMERICAS,
    TERMINOS_TFIDF,
    candidatos,
    crear_pipeline,
    escribir_json,
)

PARTICIONES = ("entrenamiento", "validacion", "prueba")
MODELO = Path("models/modelo.joblib")
INFORMES = Path("docs/reports/entrenamiento")
OBJETIVO_ACCURACY = 0.80


def leer_particiones(ruta: Path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    datos = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    faltan = sorted({*ENTRADAS, "particion", "sentimiento"} - set(datos))
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {', '.join(faltan)}.")
    particion = datos["particion"].astype("string").str.strip().str.casefold()
    if not particion.isin(PARTICIONES).all():
        raise ValueError("La columna particion contiene un nombre de partición no válido.")

    conjuntos = {}
    for nombre in PARTICIONES:
        parte = datos.loc[particion.eq(nombre)]
        X = parte[ENTRADAS].copy()
        y = parte["sentimiento"].astype("string").str.strip()
        if not y.isin(CLASES).all() or set(y) != set(CLASES):
            raise ValueError(f"La partición {nombre} debe contener las tres clases y ninguna etiqueta inválida.")
        for columna in NUMERICAS:
            X[columna] = pd.to_numeric(X[columna], errors="raise").astype(float)
        sin_imputacion = X.drop(columns=["edad", "tiempo_respuesta_horas"])
        if sin_imputacion.isna().any().any():
            faltantes = sin_imputacion.columns[sin_imputacion.isna().any()].tolist()
            raise ValueError(f"Valores ausentes no previstos en {nombre}: {', '.join(faltantes)}.")
        conjuntos[nombre] = X, y
    return conjuntos


def evaluar(y_real: pd.Series, y_predicho) -> dict:
    informe = classification_report(
        y_real, y_predicho, labels=list(CLASES), output_dict=True, zero_division=0
    )
    return {
        "accuracy": informe["accuracy"],
        "precision_macro": informe["macro avg"]["precision"],
        "recall_macro": informe["macro avg"]["recall"],
        "f1_macro": informe["macro avg"]["f1-score"],
        "por_clase": {
            clase: {
                "precision": informe[clase]["precision"],
                "recall": informe[clase]["recall"],
                "f1": informe[clase]["f1-score"],
                "soporte": int(informe[clase]["support"]),
            }
            for clase in CLASES
        },
        "matriz_confusion": confusion_matrix(
            y_real, y_predicho, labels=list(CLASES)
        ).tolist(),
    }


def guardar_grafico(y_real: pd.Series, y_predicho, ruta: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matriz = confusion_matrix(y_real, y_predicho, labels=list(CLASES), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 4.6), layout="constrained")
    ConfusionMatrixDisplay(matriz, display_labels=CLASES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".1%"
    )
    ax.set(xlabel="Sentimiento predicho", ylabel="Sentimiento real",
           title="Matriz de confusión del conjunto de prueba")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def registrar_mlflow(
    args: argparse.Namespace, informe: dict, ruta_metricas: Path, grafico: Path
) -> str:
    import mlflow

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        Path("tmp").mkdir(exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///tmp/mlflow_ap2.db")
    mlflow.set_experiment(args.experimento)
    with mlflow.start_run(run_name="Modelo final - LogisticRegression") as run:
        mlflow.log_params({
            "modelo": informe["modelo"],
            "terminos_tfidf_max": TERMINOS_TFIDF,
            "scikit_learn": sklearn.__version__,
            **{f"n_{nombre}": n for nombre, n in informe["particiones_ap1"].items()},
            **{f"modelo_{k}": str(v) for k, v in informe["parametros"].items()},
        })
        metricas = {}
        for conjunto, medidas in informe["metricas"].items():
            for clave in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                metricas[f"{conjunto}_{clave}"] = medidas[clave]
            for clase, valores in medidas["por_clase"].items():
                for clave, valor in valores.items():
                    metricas[f"{conjunto}_{clase.lower()}_{clave}"] = valor
        mlflow.log_metrics(metricas)
        run_id = run.info.run_id
        informe["mlflow_run_id"] = run_id
        escribir_json(ruta_metricas, informe)
        mlflow.log_artifact(str(args.modelo), artifact_path="modelo")
        for archivo in (ruta_metricas, grafico):
            mlflow.log_artifact(str(archivo), artifact_path="evaluacion")
    return run_id


def entrenar(args: argparse.Namespace) -> None:
    conjuntos = leer_particiones(args.datos)
    conteos = {nombre: len(y) for nombre, (_, y) in conjuntos.items()}
    print(f"Particiones AP1: {conteos}", flush=True)
    estimador = next(m for m in candidatos() if isinstance(m, LogisticRegression))
    pipeline = crear_pipeline(estimador)
    X_entrenamiento, y_entrenamiento = conjuntos["entrenamiento"]
    pipeline.fit(X_entrenamiento, y_entrenamiento)

    metricas = {}
    for nombre, titulo in (("validacion", "Validación"), ("prueba", "Prueba")):
        X, y = conjuntos[nombre]
        predicho = pipeline.predict(X)
        m = evaluar(y, predicho)
        metricas[nombre] = m
        negativas = m["por_clase"]["Negativo"]
        print(
            f"{titulo} ({len(y)} reseñas): accuracy={m['accuracy']:.2%} | "
            f"F1 macro={m['f1_macro']:.2%} | precisión neg.={negativas['precision']:.2%} | "
            f"recall neg.={negativas['recall']:.2%}",
            flush=True,
        )
        if nombre == "prueba":
            args.informes.mkdir(parents=True, exist_ok=True)
            grafico = args.informes / "matriz_confusion_prueba.png"
            guardar_grafico(y, predicho, grafico)

    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.modelo, compress=3)
    ruta_metricas = args.informes / "metricas_modelo.json"
    supera_objetivo = metricas["prueba"]["accuracy"] >= OBJETIVO_ACCURACY
    informe = {
        "modelo": type(estimador).__name__,
        "parametros": estimador.get_params(),
        "datos": args.datos.as_posix(),
        "artefacto_modelo": args.modelo.as_posix(),
        "versiones": {"pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "particiones_ap1": conteos,
        "clases": list(CLASES),
        "metricas": metricas,
        "objetivo_accuracy_prueba": OBJETIVO_ACCURACY,
        "cumple_objetivo_accuracy_prueba": supera_objetivo,
    }
    escribir_json(ruta_metricas, informe)
    print(f"Objetivo de accuracy del 80 % en prueba: {'cumplido' if supera_objetivo else 'no cumplido'}")
    print(f"Modelo guardado: {args.modelo}")
    print(f"Informe y matriz: {args.informes}", flush=True)
    if not args.sin_mlflow:
        run_id = registrar_mlflow(args, informe, ruta_metricas, grafico)
        print(f"MLflow: ejecución final {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=DATOS)
    parser.add_argument("--modelo", type=Path, default=MODELO)
    parser.add_argument("--informes", type=Path, default=INFORMES)
    parser.add_argument("--experimento", default="AP2 - Comparacion de modelos")
    parser.add_argument("--sin-mlflow", action="store_true")
    args = parser.parse_args()
    try:
        entrenar(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
