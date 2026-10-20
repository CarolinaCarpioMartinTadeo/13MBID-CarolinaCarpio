"""Selección inicial de los datos para preparar el conjunto de análisis."""

from pathlib import Path
import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent
OBJETIVO = "sentimiento"

VARIABLES_CLIENTES = [
    "edad",
    "genero",
    "ciudad",
    "antiguedad_cliente_meses",
    "canal_registro",
]

VARIABLES_RESENAS = [
    "texto_resena",
    "categoria_producto",
    "canal_compra",
    "plataforma",
    "importe_compra",
    "tiempo_respuesta_horas",
    "problema_resuelto",
    "reclamo_formal",
]

METADATOS = ["id_cliente", "id_resena", "fecha_resena"]


def seleccionar_datos(clientes, resenas):
    columnas_clientes = ["id_cliente"] + VARIABLES_CLIENTES
    columnas_resenas = (
        ["id_resena", "id_cliente", "fecha_resena"]
        + VARIABLES_RESENAS
        + [OBJETIVO]
    )
    return (
        clientes.loc[:, columnas_clientes].copy(),
        resenas.loc[:, columnas_resenas].copy(),
    )


def limpiar_datos(clientes, resenas):
    clientes = clientes.copy()
    resenas = resenas.copy()

    duplicados = int(resenas.duplicated().sum())
    resenas = resenas.drop_duplicates().reset_index(drop=True)

    edades_fuera = (
        clientes["edad"].notna()
        & ~clientes["edad"].between(18, 79)
    )
    clientes.loc[edades_fuera, "edad"] = float("nan")

    print("\n--- Limpieza inicial (C2) ---")
    print(f"Reseñas duplicadas eliminadas: {duplicados}")
    print(
        "Edades fuera de dominio convertidas en ausentes:",
        int(edades_fuera.sum()),
    )
    print(f"Clientes conservados: {len(clientes)}")
    print(f"Reseñas conservadas: {len(resenas)}")
    print("Edades pendientes de imputar:", int(clientes["edad"].isna().sum()))
    print(
        "Tiempos de respuesta pendientes de imputar:",
        int(resenas["tiempo_respuesta_horas"].isna().sum()),
    )
    return clientes, resenas


def main():
    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv", sep=";",
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv", sep=";",
    )

    correspondencia = {
        1: "Negativo", 2: "Negativo", 3: "Neutral",
        4: "Positivo", 5: "Positivo",
    }
    etiqueta_por_calificacion = resenas["calificacion"].map(correspondencia)
    coincidencia = etiqueta_por_calificacion.eq(resenas[OBJETIVO]).mean() * 100
    print(f"Coincidencia entre calificación y sentimiento: {coincidencia:.2f} %")

    clientes, resenas = limpiar_datos(clientes, resenas)
    clientes_seleccionados, resenas_seleccionadas = seleccionar_datos(
        clientes, resenas,
    )

    print("\n--- Selección de datos ---")
    print(f"Clientes: {clientes_seleccionados.shape}")
    print(f"Reseñas: {resenas_seleccionadas.shape}")
    print(f"Variable objetivo: {OBJETIVO}")
    print(f"Metadatos: {METADATOS}")

    predictoras = VARIABLES_CLIENTES + VARIABLES_RESENAS
    print(f"Variables predictoras candidatas ({len(predictoras)}): {predictoras}")
    print(
        "Columnas descartadas de reseñas:",
        sorted(set(resenas.columns) - set(resenas_seleccionadas.columns)),
    )


if __name__ == "__main__":
    main()