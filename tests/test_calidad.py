import pandas as pd
import pandera.pandas as pa
import pytest

from src.esquemas import esquema_clientes, esquema_resenas


@pytest.fixture
def clientes():
    return pd.DataFrame({
        "id_cliente": [1, 2],
        "edad": [34.0, 46.0],
        "genero": ["Femenino", "Masculino"],
        "ciudad": ["Córdoba", "Rosario"],
        "antiguedad_cliente_meses": [5, 12],
        "canal_registro": ["Online", "Tienda física"],
    })


@pytest.fixture
def resenas():
    return pd.DataFrame({
        "id_resena": [100, 101],
        "id_cliente": [1, 1],
        "categoria_producto": ["Libros", "Hogar"],
        "canal_compra": ["Online", "Offline"],
        "plataforma": ["Amazon", "MercadoLibre"],
        "calificacion": [5, 2],
        "texto_resena": ["Muy buena compra", "El pedido llegó tarde"],
        "tiempo_respuesta_horas": [2.0, 5.0],
        "problema_resuelto": ["Sí", "No"],
        "reclamo_formal": ["No", "Sí"],
        "importe_compra": [500.0, 750.0],
        "fecha_resena": ["2026-10-18", "2026-10-19"],
        "sentimiento": ["Positivo", "Negativo"],
    })


def test_acepta_tablas_validas_y_varias_resenas_por_cliente(clientes, resenas):
    esquema_clientes().validate(clientes, lazy=True)
    esquema_resenas(clientes["id_cliente"]).validate(resenas, lazy=True)


@pytest.mark.parametrize("edad", [18.0, 79.0])
def test_acepta_limites_del_dominio_de_edad(clientes, edad):
    clientes.loc[0, "edad"] = edad
    esquema_clientes().validate(clientes, lazy=True)


@pytest.mark.parametrize("columna, valor", [
    ("edad", 17.0),
    ("edad", 80.0),
    ("edad", float("nan")),
    ("id_cliente", 2),
    ("antiguedad_cliente_meses", 96),
    ("genero", "Desconocido"),
])
def test_detecta_anomalias_de_clientes(clientes, columna, valor):
    clientes.loc[0, columna] = valor
    with pytest.raises(pa.errors.SchemaErrors) as error:
        esquema_clientes().validate(clientes, lazy=True)
    assert columna in error.value.failure_cases["column"].values


@pytest.mark.parametrize("columna, valor", [
    ("id_resena", 101),
    ("id_cliente", 9999),
    ("calificacion", 0),
    ("calificacion", 6),
    ("tiempo_respuesta_horas", -1.0),
    ("tiempo_respuesta_horas", float("nan")),
    ("importe_compra", -1.0),
    ("fecha_resena", "2026-02-30"),
    ("fecha_resena", "20/10/2026"),
    ("texto_resena", "  \t"),
    ("sentimiento", "Mixto"),
])
def test_detecta_anomalias_de_resenas(clientes, resenas, columna, valor):
    resenas.loc[0, columna] = valor
    with pytest.raises(pa.errors.SchemaErrors) as error:
        esquema_resenas(clientes["id_cliente"]).validate(resenas, lazy=True)
    assert columna in error.value.failure_cases["column"].values


def test_detecta_duplicados_exactos(clientes, resenas):
    repetidas = pd.concat([resenas, resenas.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaErrors) as error:
        esquema_resenas(clientes["id_cliente"]).validate(repetidas, lazy=True)
    assert "id_resena" in error.value.failure_cases["column"].values


def test_rechaza_ausencia_de_variable_objetivo(clientes, resenas):
    sin_objetivo = resenas.drop(columns="sentimiento")
    with pytest.raises(pa.errors.SchemaErrors):
        esquema_resenas(clientes["id_cliente"]).validate(sin_objetivo, lazy=True)


def test_rechaza_tabla_vacia(clientes):
    with pytest.raises(pa.errors.SchemaErrors):
        esquema_clientes().validate(clientes.iloc[:0], lazy=True)