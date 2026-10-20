"""Reglas de calidad para las tablas de clientes y reseñas del MVP."""

import pandas as pd
import pandera.pandas as pa


DOMINIOS_CLIENTES = {
    "genero": ["Femenino", "Masculino", "Otro"],
    "ciudad": [
        "Buenos Aires", "Córdoba", "La Plata", "Mar del Plata", "Mendoza",
        "Posadas", "Rosario", "Salta", "San Miguel de Tucumán", "Santa Fe",
    ],
    "canal_registro": ["Online", "Tienda física"],
}

DOMINIOS_RESENAS = {
    "categoria_producto": [
        "Alimentos", "Belleza", "Calzado", "Deportes", "Electrónica",
        "Hogar", "Indumentaria", "Libros",
    ],
    "canal_compra": ["Online", "Offline"],
    "plataforma": [
        "Amazon", "Falabella", "Flipkart", "JioMart", "Meesho",
        "MercadoLibre", "Nykaa", "Zepto",
    ],
    "problema_resuelto": ["Sí", "No"],
    "reclamo_formal": ["Sí", "No"],
    "sentimiento": ["Positivo", "Neutral", "Negativo"],
}


def esquema_clientes():
    """Define los requisitos que deberá cumplir la tabla de clientes limpia."""
    columnas = {
        "id_cliente": pa.Column(
            int, pa.Check.gt(0), unique=True,
            report_duplicates="exclude_first",
        ),
        "edad": pa.Column(float, pa.Check.in_range(18, 79), coerce=True),
        "antiguedad_cliente_meses": pa.Column(int, pa.Check.in_range(1, 95)),
    }
    columnas.update({
        nombre: pa.Column(str, pa.Check.isin(valores))
        for nombre, valores in DOMINIOS_CLIENTES.items()
    })
    return pa.DataFrameSchema(
        columnas,
        checks=pa.Check(lambda df: not df.empty, name="tabla_no_vacia"),
        strict=True,
        name="clientes",
    )


def esquema_resenas(ids_clientes):
    """Define las reglas de reseñas y su relación con los clientes existentes."""
    columnas = {
        "id_resena": pa.Column(
            int, pa.Check.gt(0), unique=True,
            report_duplicates="exclude_first",
        ),
        # Un mismo cliente puede tener varias reseñas.
        "id_cliente": pa.Column(
            int, [pa.Check.gt(0), pa.Check.isin(list(ids_clientes))],
        ),
        "calificacion": pa.Column(int, pa.Check.in_range(1, 5)),
        "texto_resena": pa.Column(
            str,
            pa.Check(
                lambda s: s.str.strip().str.len().gt(0),
                name="texto_no_vacio",
            ),
        ),
        "tiempo_respuesta_horas": pa.Column(
            float, pa.Check.ge(0), coerce=True,
        ),
        "importe_compra": pa.Column(float, pa.Check.ge(0), coerce=True),
        "fecha_resena": pa.Column(
            str,
            [
                pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
                pa.Check(
                    lambda s: pd.to_datetime(
                        s, format="%Y-%m-%d", errors="coerce",
                    ).notna(),
                    name="fecha_valida",
                ),
            ],
        ),
    }
    columnas.update({
        nombre: pa.Column(str, pa.Check.isin(valores))
        for nombre, valores in DOMINIOS_RESENAS.items()
    })
    return pa.DataFrameSchema(
        columnas,
        checks=pa.Check(lambda df: not df.empty, name="tabla_no_vacia"),
        strict=True,
        name="resenas",
    )