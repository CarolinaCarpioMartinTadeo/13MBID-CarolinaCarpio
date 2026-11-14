from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from src import monitoreo_drift as deriva
from src.entrenamiento import INFORMES, MODELO
from src.experimentacion import CLASES, DATOS, ENTRADAS, NUMERICAS, TEXTO, escribir_json

LOTES = (Path("data/raw/resenas_nuevas.csv"), Path("data/raw/resenas_nuevas_drift.csv"))
SALIDA = Path("docs/reports/monitoreo")
METRICAS = INFORMES / "metricas_modelo.json"


def leer_referencia(ruta: Path) -> pd.DataFrame:
    datos = deriva.leer_csv(ruta)
    if "particion" not in datos:
        raise ValueError(f"Falta la columna particion en {ruta}.")
    particion = datos["particion"].astype("string").str.strip().str.casefold()
    referencia = datos.loc[particion.eq("entrenamiento")].copy()
    if referencia.empty:
        raise ValueError(f"No hay filas de entrenamiento en {ruta}.")
    return referencia


def predecir(modelo, datos: pd.DataFrame) -> pd.Series:
    if TEXTO not in datos or datos[TEXTO].isna().any():
        raise ValueError("Las reseñas que se van a predecir deben incluir texto.")
    X = datos.copy()
    texto = X[TEXTO].astype(str)
    X["longitud_resena"] = texto.str.len()
    X["cantidad_palabras_resena"] = texto.str.split().str.len()
    # id_cliente se usa al ajustar la imputación, no al predecir.
    columnas = [c for c in ENTRADAS if c != "id_cliente"]
    faltan = sorted(set(columnas) - set(X))
    if faltan:
        raise ValueError(f"Faltan columnas para predecir: {', '.join(faltan)}.")
    X = X[columnas]
    for columna in NUMERICAS:
        X[columna] = pd.to_numeric(X[columna], errors="raise").astype(float)
    return pd.Series(modelo.predict(X), index=datos.index, name="prediccion")


def evaluar_acierto(lote: pd.DataFrame, predicciones: pd.Series, minimo: float) -> dict:
    if deriva.OBJETIVO not in lote:
        return {"n_etiquetadas": 0, "accuracy": None, "umbral": minimo, "cumple_umbral": None}
    etiquetas = lote[deriva.OBJETIVO].astype("string").str.strip()
    presentes = etiquetas.notna()
    if not etiquetas[presentes].isin(CLASES).all():
        raise ValueError("El lote contiene etiquetas de sentimiento no reconocidas.")
    n = int(presentes.sum())
    accuracy = float(accuracy_score(etiquetas[presentes], predicciones[presentes])) if n else None
    return {
        "n_etiquetadas": n,
        "accuracy": accuracy,
        "umbral": minimo,
        "cumple_umbral": accuracy >= minimo if accuracy is not None else None,
    }


def monitorear_lotes(args: argparse.Namespace) -> None:
    referencia = leer_referencia(args.referencia)
    modelo = joblib.load(args.modelo)
    metricas = json.loads(args.metricas.read_text(encoding="utf-8"))
    minimo = float(metricas["objetivo_accuracy_prueba"])
    pred_ref = predecir(modelo, referencia)
    print(f"Referencia: {len(referencia)} reseñas de entrenamiento.")

    for ruta in args.lotes or LOTES:
        lote = deriva.leer_csv(ruta)
        if lote.empty:
            raise ValueError(f"El lote {ruta} está vacío.")
        informe = deriva.analizar(referencia, lote)
        pred_nuevo = predecir(modelo, lote)
        distribucion = deriva.analizar_categorica(pred_ref, pred_nuevo)
        rendimiento = evaluar_acierto(lote, pred_nuevo, minimo)
        psis = informe["variables"]
        resumen = informe["resumen"]
        alertas = [f"Deriva en {col}" for col in resumen["variables_criticas_con_deriva"]]
        if resumen["deriva_en_objetivo"]:
            alertas.append("Cambió la distribución de las etiquetas reales.")
        if distribucion["deriva"]:
            alertas.append("Cambió la distribución de las predicciones.")
        if rendimiento["cumple_umbral"] is False:
            alertas.append("El acierto del lote queda por debajo del umbral.")

        informe.update(
            entradas={
                "referencia": args.referencia.as_posix(),
                "particion_referencia": "entrenamiento",
                "lote": ruta.as_posix(),
                "modelo": args.modelo.as_posix(),
                "metricas_modelo": args.metricas.as_posix(),
                "mlflow_run_id": metricas.get("mlflow_run_id"),
            },
            distribucion_predicciones=distribucion,
            rendimiento_lote={
                **rendimiento,
                "accuracy_prueba_modelo": metricas["metricas"]["prueba"]["accuracy"],
            },
        )
        resumen["alertas"] = alertas
        resumen["recomendacion"] = (
            "Revisar el lote antes de decidir si reentrenar."
            if alertas else "No se detectaron alertas con los criterios definidos."
        )
        salida = args.salida / f"{ruta.stem}.json"
        escribir_json(salida, informe)
        accuracy = rendimiento["accuracy"]
        acierto = f"{accuracy:.2%}" if accuracy is not None else "sin etiquetas"
        print(f"\n{ruta.name}: {len(lote)} reseñas")
        print(f"  PSI tiempo de respuesta: {psis['tiempo_respuesta_horas']['psi']:.4f}")
        print(f"  PSI importe de compra: {psis['importe_compra']['psi']:.4f}")
        print(f"  PSI clases predichas: {distribucion['psi']:.4f}")
        print(f"  Acierto del modelo: {acierto}")
        print(f"  Alertas: {', '.join(alertas) if alertas else 'ninguna'}")
        print(f"  Informe: {salida}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--referencia", type=Path, default=DATOS)
    parser.add_argument("--modelo", type=Path, default=MODELO)
    parser.add_argument("--metricas", type=Path, default=METRICAS)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("lotes", nargs="*", type=Path, help="CSV a analizar; por defecto, los dos de SP7.")
    try:
        monitorear_lotes(parser.parse_args())
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
