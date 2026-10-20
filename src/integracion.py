"""Integra clientes y reseñas y construye las variables derivadas de C3."""

import pandas as pd


VARIABLES_DERIVADAS = [
    "longitud_resena",
    "cantidad_palabras_resena",
    "tiempo_respuesta_dias",
]


def integrar_datos(clientes, resenas):
    """Conserva una fila por reseña y comprueba su relación con clientes."""
    datos = resenas.merge(
        clientes,
        on="id_cliente",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    sin_cliente = int(datos["_merge"].eq("left_only").sum())
    if sin_cliente:
        raise ValueError(f"Hay {sin_cliente} reseñas sin cliente asociado.")

    if len(datos) != len(resenas):
        raise ValueError("La integración ha cambiado el número de reseñas.")

    if not datos["id_resena"].is_unique:
        raise ValueError("El conjunto integrado tiene identificadores repetidos.")

    return datos.drop(columns="_merge")


def construir_variables(datos):
    """Añade medidas del texto y convierte el tiempo de respuesta a días."""
    datos = datos.copy()

    datos["longitud_resena"] = datos["texto_resena"].str.len()
    datos["cantidad_palabras_resena"] = (
        datos["texto_resena"].str.split().str.len()
    )
    datos["tiempo_respuesta_dias"] = datos["tiempo_respuesta_horas"] / 24

    return datos


def main():
    from preparacion import RAIZ, limpiar_datos, seleccionar_datos
    from imputacion import (
        crear_particiones,
        ajustar_imputacion,
        aplicar_imputacion,
    )
    from esquemas import esquema_clientes, esquema_resenas

    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv", sep=";",
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv", sep=";",
    )
    clientes, resenas = limpiar_datos(clientes, resenas)

    particiones = crear_particiones(resenas)
    resenas_entrenamiento = resenas.loc[
        particiones.eq("entrenamiento")
    ]
    clientes_entrenamiento = clientes.loc[
        clientes["id_cliente"].isin(resenas_entrenamiento["id_cliente"])
    ]

    parametros = ajustar_imputacion(
        clientes_entrenamiento, resenas_entrenamiento,
    )
    clientes, resenas = aplicar_imputacion(
        clientes, resenas, parametros,
    )

    esquema_clientes().validate(clientes, lazy=True)
    esquema_resenas(clientes["id_cliente"]).validate(
        resenas, lazy=True,
    )

    clientes, resenas = seleccionar_datos(clientes, resenas)
    resenas["particion"] = particiones

    datos = integrar_datos(clientes, resenas)
    datos = construir_variables(datos)

    print("\n--- Integración y construcción de variables (C3) ---")
    print(
        f"Conjunto integrado: {datos.shape[0]} filas "
        f"y {datos.shape[1]} columnas."
    )
    print(
        "Identificadores de reseña únicos:",
        datos["id_resena"].is_unique,
    )
    print("Valores ausentes:", int(datos.isna().sum().sum()))
    print("Variables derivadas:", VARIABLES_DERIVADAS)

    print("\nEjemplo de las variables derivadas:")
    print(
        datos[["id_resena"] + VARIABLES_DERIVADAS]
        .head()
        .to_string(index=False)
    )

    print("\nParticiones conservadas:")
    print(datos["particion"].value_counts().to_string())


if __name__ == "__main__":
    main()