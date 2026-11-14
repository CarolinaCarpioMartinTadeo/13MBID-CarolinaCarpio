from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- Configuración (ajustar si cambian las variables del proyecto) -----------
NUMERICAS = ["importe_compra", "tiempo_respuesta_horas", "edad", "antiguedad_cliente_meses"]
CATEGORICAS = [
    "categoria_producto", "canal_compra", "plataforma",
    "genero", "problema_resuelto", "reclamo_formal",
]
OBJETIVO = "sentimiento"
CRITICAS = ["tiempo_respuesta_horas", "importe_compra"]  # variables que bloquean el reentrenamiento
UMBRAL_PSI_MODERADO = 0.10
UMBRAL_PSI_ALTO = 0.25
ALFA = 0.01  # nivel de significación de los contrastes
# ------------------------------------------------------------------------------


def leer_csv(ruta: str | Path) -> pd.DataFrame:
    """Lee un CSV detectando el separador (';' o ',')."""
    return pd.read_csv(ruta, sep=None, engine="python")


def psi_numerica(ref: pd.Series, nuevo: pd.Series, bins: int = 10) -> float:
    """Population Stability Index con deciles de la referencia."""
    ref, nuevo = ref.dropna(), nuevo.dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref)
    q = np.histogram(nuevo, cortes)[0] / len(nuevo)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorica(ref: pd.Series, nuevo: pd.Series) -> float:
    cats = sorted(set(ref.dropna().unique()) | set(nuevo.dropna().unique()))
    p = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    q = nuevo.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def nivel_psi(psi: float) -> str:
    if psi >= UMBRAL_PSI_ALTO:
        return "alto"
    if psi >= UMBRAL_PSI_MODERADO:
        return "moderado"
    return "estable"


def analizar_numerica(ref: pd.Series, nuevo: pd.Series) -> dict:
    ks = stats.ks_2samp(ref.dropna(), nuevo.dropna())
    psi = psi_numerica(ref, nuevo)
    return {
        "tipo": "numerica",
        "media_ref": round(float(ref.mean()), 3),
        "media_nuevo": round(float(nuevo.mean()), 3),
        "mediana_ref": round(float(ref.median()), 3),
        "mediana_nuevo": round(float(nuevo.median()), 3),
        "ks_estadistico": round(float(ks.statistic), 4),
        "ks_pvalor": float(ks.pvalue),
        "psi": round(psi, 4),
        "nivel_psi": nivel_psi(psi),
        "deriva": bool(ks.pvalue < ALFA and psi >= UMBRAL_PSI_MODERADO),
    }


def analizar_categorica(ref: pd.Series, nuevo: pd.Series) -> dict:
    cats = sorted(set(ref.dropna().unique()) | set(nuevo.dropna().unique()))
    obs = nuevo.value_counts().reindex(cats).fillna(0).to_numpy()
    prop_ref = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    esperado = np.clip(prop_ref, 1e-9, None)
    esperado = esperado / esperado.sum() * obs.sum()
    chi = stats.chisquare(obs, esperado)
    psi = psi_categorica(ref, nuevo)
    return {
        "tipo": "categorica",
        "proporciones_ref": {c: round(float(v), 4) for c, v in zip(cats, prop_ref)},
        "proporciones_nuevo": {c: round(float(v / obs.sum()), 4) for c, v in zip(cats, obs)},
        "categorias_nuevas": sorted(set(nuevo.dropna().unique()) - set(ref.dropna().unique())),
        "chi2_pvalor": float(chi.pvalue),
        "psi": round(psi, 4),
        "nivel_psi": nivel_psi(psi),
        "deriva": bool(chi.pvalue < ALFA and psi >= UMBRAL_PSI_MODERADO),
    }


def calidad_basica(df: pd.DataFrame) -> dict:
    return {
        "filas": int(len(df)),
        "nulos_por_columna": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "filas_duplicadas": int(df.duplicated().sum()),
    }


def analizar(ref: pd.DataFrame, nuevo: pd.DataFrame) -> dict:
    esperadas = NUMERICAS + CATEGORICAS
    faltan = [c for c in esperadas if c not in ref.columns or c not in nuevo.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en referencia o lote nuevo: {faltan}")

    variables = {c: analizar_numerica(ref[c], nuevo[c]) for c in NUMERICAS}
    variables.update({c: analizar_categorica(ref[c], nuevo[c]) for c in CATEGORICAS})

    objetivo = None
    if OBJETIVO in ref.columns and OBJETIVO in nuevo.columns:
        objetivo = analizar_categorica(ref[OBJETIVO], nuevo[OBJETIVO])

    con_deriva = sorted(c for c, r in variables.items() if r["deriva"])
    criticas = sorted(c for c in con_deriva if c in CRITICAS)
    return {
        "calidad_referencia": calidad_basica(ref),
        "calidad_nuevo": calidad_basica(nuevo),
        "variables": variables,
        "distribucion_objetivo": objetivo,
        "resumen": {
            "variables_con_deriva": con_deriva,
            "variables_criticas_con_deriva": criticas,
            "deriva_en_objetivo": bool(objetivo and objetivo["deriva"]),
            "recomendacion": (
                "Revisar los datos y evaluar el reentrenamiento antes de promover un nuevo modelo."
                if criticas or (objetivo and objetivo["deriva"])
                else "Sin deriva crítica: el lote puede utilizarse en el ciclo de reentrenamiento."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--referencia", required=True)
    ap.add_argument("--nuevos", required=True)
    ap.add_argument("--salida", default="docs/reports/monitoreo/informe_drift.json")
    ap.add_argument("--estricto", action="store_true", help="fallar ante deriva en cualquier variable")
    args = ap.parse_args(argv)

    try:
        informe = analizar(leer_csv(args.referencia), leer_csv(args.nuevos))
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    informe["entradas"] = {"referencia": str(args.referencia), "nuevos": str(args.nuevos)}
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    r = informe["resumen"]
    print(f"Informe guardado en {salida}")
    print(f"Variables con deriva: {r['variables_con_deriva'] or 'ninguna'}")
    print(f"Críticas con deriva:  {r['variables_criticas_con_deriva'] or 'ninguna'}")
    print(f"Deriva en objetivo:   {r['deriva_en_objetivo']}")
    print(r["recomendacion"])

    hay_deriva = bool(r["variables_criticas_con_deriva"]) or r["deriva_en_objetivo"]
    if args.estricto:
        hay_deriva = hay_deriva or bool(r["variables_con_deriva"])
    return 1 if hay_deriva else 0


if __name__ == "__main__":
    sys.exit(main())
