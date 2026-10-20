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