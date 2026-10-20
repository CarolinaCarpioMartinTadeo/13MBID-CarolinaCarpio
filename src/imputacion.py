"""Particiones del MVP e imputación ajustada con entrenamiento."""

import pandas as pd
from sklearn.model_selection import train_test_split


def crear_particiones(resenas):
    """Reproduce la división estratificada del MVP con semilla 42."""
    indices_temporales, indices_prueba = train_test_split(
        resenas.index,
        test_size=0.10,
        random_state=42,
        stratify=resenas["sentimiento"],
    )

    indices_entrenamiento, indices_validacion = train_test_split(
        indices_temporales,
        test_size=0.22,
        random_state=42,
        stratify=resenas.loc[indices_temporales, "sentimiento"],
    )

    particiones = pd.Series(
        index=resenas.index,
        dtype="string",
        name="particion",
    )
    particiones.loc[indices_entrenamiento] = "entrenamiento"
    particiones.loc[indices_validacion] = "validacion"
    particiones.loc[indices_prueba] = "prueba"

    return particiones


def ajustar_imputacion(clientes_entrenamiento, resenas_entrenamiento):
    """Aprende las medianas usando las muestras de entrenamiento."""
    edad_global = clientes_entrenamiento["edad"].median()
    tiempo_global = resenas_entrenamiento["tiempo_respuesta_horas"].median()

    if pd.isna(edad_global):
        raise ValueError(
            "Faltan edades válidas en los clientes de entrenamiento."
        )

    if pd.isna(tiempo_global):
        raise ValueError(
            "Faltan tiempos válidos en las reseñas de entrenamiento."
        )

    medianas_edad = (
        clientes_entrenamiento.groupby("ciudad")["edad"]
        .median()
        .dropna()
    )
    medianas_tiempo = (
        resenas_entrenamiento.groupby("categoria_producto")
        ["tiempo_respuesta_horas"]
        .median()
        .dropna()
    )

    return {
        "edad_por_ciudad": medianas_edad.to_dict(),
        "edad_global": float(edad_global),
        "tiempo_por_categoria": medianas_tiempo.to_dict(),
        "tiempo_global": float(tiempo_global),
    }


def aplicar_imputacion(clientes, resenas, parametros):
    """Completa las ausencias sin recalcular las medianas recibidas."""
    clientes = clientes.copy()
    resenas = resenas.copy()

    edad_referencia = clientes["ciudad"].map(
        parametros["edad_por_ciudad"]
    )
    edad_referencia = edad_referencia.fillna(parametros["edad_global"])
    clientes["edad"] = clientes["edad"].fillna(edad_referencia)

    tiempo_referencia = resenas["categoria_producto"].map(
        parametros["tiempo_por_categoria"]
    )
    tiempo_referencia = tiempo_referencia.fillna(
        parametros["tiempo_global"]
    )
    resenas["tiempo_respuesta_horas"] = (
        resenas["tiempo_respuesta_horas"].fillna(tiempo_referencia)
    )

    return clientes, resenas


def main():
    from preparacion import RAIZ, limpiar_datos
    from esquemas import esquema_clientes, esquema_resenas

    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv",
        sep=";",
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv",
        sep=";",
    )
    clientes, resenas = limpiar_datos(clientes, resenas)

    particiones = crear_particiones(resenas)

    resenas_entrenamiento = resenas.loc[
        particiones.eq("entrenamiento")
    ]
    clientes_entrenamiento = clientes.loc[
        clientes["id_cliente"].isin(
            resenas_entrenamiento["id_cliente"]
        )
    ]

    parametros = ajustar_imputacion(
        clientes_entrenamiento,
        resenas_entrenamiento,
    )
    clientes_imputados, resenas_imputadas = aplicar_imputacion(
        clientes,
        resenas,
        parametros,
    )

    esquema_clientes().validate(clientes_imputados, lazy=True)
    esquema_resenas(clientes_imputados["id_cliente"]).validate(
        resenas_imputadas,
        lazy=True,
    )

    print("\n--- Imputación y validación (C2) ---")
    print("Particiones de reseñas:")
    print(particiones.value_counts().to_string())
    print(
        "Edades ausentes tras imputar:",
        int(clientes_imputados["edad"].isna().sum()),
    )
    print(
        "Tiempos ausentes tras imputar:",
        int(resenas_imputadas["tiempo_respuesta_horas"].isna().sum()),
    )
    print("Ambas tablas cumplen los esquemas de calidad.")


if __name__ == "__main__":
    main()