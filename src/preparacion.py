"""Ejecuta y exporta la preparación reproducible de los datos (C1-C4)."""

import json
from pathlib import Path

import pandas as pd

from esquemas import esquema_clientes, esquema_resenas
from imputacion import crear_particiones, ajustar_imputacion, aplicar_imputacion
from integracion import integrar_datos, construir_variables, VARIABLES_DERIVADAS


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

METADATOS = ["id_cliente", "id_resena", "fecha_resena", "particion"]
AUXILIARES = ["tiempo_respuesta_horas"]

PREDICTORAS = (
    VARIABLES_CLIENTES
    + [c for c in VARIABLES_RESENAS if c not in AUXILIARES]
    + VARIABLES_DERIVADAS
)


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
    print(
        "Edades pendientes de imputar:",
        int(clientes["edad"].isna().sum()),
    )
    print(
        "Tiempos de respuesta pendientes de imputar:",
        int(resenas["tiempo_respuesta_horas"].isna().sum()),
    )

    return clientes, resenas


def preparar_conjunto(clientes, resenas, particiones):
    """Selecciona, integra y deriva, conservando las particiones recibidas."""
    clientes, resenas = seleccionar_datos(clientes, resenas)
    resenas["particion"] = particiones

    return construir_variables(integrar_datos(clientes, resenas))


def validar_conjunto(datos):
    """Comprueba la estructura y la completitud del conjunto final imputado."""
    esperadas = set(
        METADATOS + AUXILIARES + PREDICTORAS + [OBJETIVO]
    )

    if set(datos.columns) != esperadas:
        raise ValueError(
            "Las columnas finales no coinciden con las previstas."
        )

    if datos.empty or datos.isna().any().any():
        raise ValueError(
            "El conjunto preparado está vacío o contiene ausencias."
        )

    if not datos["id_resena"].is_unique:
        raise ValueError(
            "Los identificadores de reseña deben ser únicos."
        )


def main():
    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv",
        sep=";",
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv",
        sep=";",
    )

    correspondencia = {
        1: "Negativo",
        2: "Negativo",
        3: "Neutral",
        4: "Positivo",
        5: "Positivo",
    }

    etiqueta_por_calificacion = resenas["calificacion"].map(
        correspondencia
    )
    coincidencia = (
        etiqueta_por_calificacion.eq(resenas[OBJETIVO]).mean() * 100
    )
    print(
        "Coincidencia entre calificación y sentimiento: "
        f"{coincidencia:.2f} %"
    )

    clientes, resenas = limpiar_datos(clientes, resenas)
    particiones = crear_particiones(resenas)

    # Permite reajustar la imputación dentro de cada fold de AP2.
    datos_sin_imputar = preparar_conjunto(
        clientes, resenas, particiones,
    )

    resenas_entrenamiento = resenas.loc[
        particiones.eq("entrenamiento")
    ]
    clientes_entrenamiento = clientes.loc[
        clientes["id_cliente"].isin(
            resenas_entrenamiento["id_cliente"]
        )
    ]

    parametros = ajustar_imputacion(
        clientes_entrenamiento, resenas_entrenamiento,
    )
    clientes_imputados, resenas_imputadas = aplicar_imputacion(
        clientes, resenas, parametros,
    )

    clientes_imputados = esquema_clientes().validate(
        clientes_imputados, lazy=True,
    )
    resenas_imputadas = esquema_resenas(
        clientes_imputados["id_cliente"],
    ).validate(resenas_imputadas, lazy=True)

    datos_preparados = preparar_conjunto(
        clientes_imputados, resenas_imputadas, particiones,
    )
    validar_conjunto(datos_preparados)

    if not datos_sin_imputar[["id_resena", "particion"]].equals(
        datos_preparados[["id_resena", "particion"]]
    ):
        raise ValueError(
            "Las dos versiones no conservan las mismas particiones."
        )

    salida = RAIZ / "data" / "processed" / "preparacion"
    informes = RAIZ / "docs" / "reports" / "preparacion"

    salida.mkdir(parents=True, exist_ok=True)
    informes.mkdir(parents=True, exist_ok=True)

    archivos = []

    for nombre, datos in [
        ("datos_preparados.csv", datos_preparados),
        ("datos_sin_imputar.csv", datos_sin_imputar),
    ]:
        ruta = salida / nombre
        datos.to_csv(
            ruta,
            sep=";",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        archivos.append(ruta)

    resumen = {
        "filas": len(datos_preparados),
        "columnas": len(datos_preparados.columns),
        "objetivo": OBJETIVO,
        "metadatos": METADATOS,
        "predictoras_candidatas": PREDICTORAS,
        "auxiliares": AUXILIARES,
        "formato_csv": {
            "separador": ";",
            "codificacion": "utf-8",
            "indice": False,
            "fecha_resena": "AAAA-MM-DD",
        },
        "particiones": particiones.value_counts().to_dict(),
        "semilla": 42,
        "particion_por": "resena",
        "coincidencia_calificacion_sentimiento_pct": round(
            coincidencia, 2,
        ),
        "parametros_imputacion_entrenamiento": parametros,
        "validacion_tablas_imputadas": "CUMPLE",
        "nulos_preparados": int(
            datos_preparados.isna().sum().sum()
        ),
        "nulos_sin_imputar_por_columna": (
            datos_sin_imputar.isna().sum().to_dict()
        ),
        "uso_ap2": (
            "En validacion cruzada usar datos_sin_imputar.csv, "
            "reajustar la imputacion dentro de cada fold de "
            "entrenamiento y recalcular tiempo_respuesta_dias "
            "despues de imputar las horas. "
            "Excluir metadatos y auxiliares de los predictores. "
            "Los indicadores de atencion requieren estar disponibles "
            "en el momento de la prediccion."
        ),
    }

    ruta_resumen = informes / "resumen_preparacion.json"
    ruta_resumen.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    archivos.append(ruta_resumen)

    print("\n--- Exportación de datos (C4) ---")
    print(f"Conjunto preparado: {datos_preparados.shape}")
    print(
        "Valores ausentes tras imputar:",
        resumen["nulos_preparados"],
    )
    print(f"Variable objetivo: {OBJETIVO}")
    print(f"Metadatos: {METADATOS}")
    print(f"Predictoras candidatas: {len(PREDICTORAS)}")

    print("\nParticiones:")
    print(particiones.value_counts().to_string())

    for ruta in archivos:
        print(f"Archivo guardado en: {ruta}")


if __name__ == "__main__":
    main()