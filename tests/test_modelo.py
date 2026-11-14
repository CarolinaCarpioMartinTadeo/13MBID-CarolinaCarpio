import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.entrenamiento import DATOS, INFORMES, MODELO, leer_particiones

RAIZ = Path(__file__).resolve().parents[1]
CLASES = ("Negativo", "Neutral", "Positivo")
ACCURACY_MINIMA = 0.80


@lru_cache(maxsize=1)
def _informe():
    ruta = RAIZ / INFORMES / "metricas_modelo.json"
    assert ruta.is_file(), f"Falta {ruta}. Ejecuta primero python -m src.entrenamiento."
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _modelo():
    ruta = RAIZ / MODELO
    assert ruta.is_file(), f"Falta {ruta}. Ejecuta primero python -m src.entrenamiento."
    return joblib.load(ruta)


@lru_cache(maxsize=1)
def _particiones():
    return leer_particiones(RAIZ / DATOS)


@lru_cache(maxsize=1)
def _resultado_prueba():
    X, y = _particiones()["prueba"]
    return y, _modelo().predict(X)


def test_informe_completo_en_validacion_y_prueba():
    informe = _informe()
    assert tuple(informe["clases"]) == CLASES
    for nombre, (_, y) in _particiones().items():
        assert informe["particiones_ap1"][nombre] == len(y)

    for nombre in ("validacion", "prueba"):
        medidas = informe["metricas"][nombre]
        for clave in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0 <= medidas[clave] <= 1, f"{nombre}: {clave} fuera de [0, 1]"

        matriz = np.asarray(medidas["matriz_confusion"])
        assert matriz.shape == (3, 3)
        assert np.issubdtype(matriz.dtype, np.integer) and (matriz >= 0).all()
        y = _particiones()[nombre][1]
        for indice, clase in enumerate(CLASES):
            m = medidas["por_clase"][clase]
            assert m["soporte"] == int(y.eq(clase).sum()) == matriz[indice].sum()
            for clave in ("precision", "recall", "f1"):
                assert 0 <= m[clave] <= 1, f"{nombre}: {clave} de {clase} fuera de [0, 1]"


def test_modelo_alcanza_accuracy_minima_en_prueba():
    y, predicciones = _resultado_prueba()
    accuracy = accuracy_score(y, predicciones)
    assert accuracy >= ACCURACY_MINIMA, (
        f"Accuracy en prueba: {accuracy:.2%}; mínimo: {ACCURACY_MINIMA:.0%}."
    )


def test_resultados_de_prueba_corresponden_al_modelo_guardado():
    y, predicciones = _resultado_prueba()
    informe = _informe()
    prueba = informe["metricas"]["prueba"]
    matriz = confusion_matrix(y, predicciones, labels=list(CLASES))
    accuracy = accuracy_score(y, predicciones)
    assert np.isclose(prueba["accuracy"], accuracy, rtol=0, atol=1e-10)
    assert np.array_equal(prueba["matriz_confusion"], matriz)
    assert informe["objetivo_accuracy_prueba"] == ACCURACY_MINIMA
    assert informe["cumple_objetivo_accuracy_prueba"] == bool(accuracy >= ACCURACY_MINIMA)


def test_modelo_devuelve_probabilidades_para_las_tres_clases():
    X, _ = _particiones()["prueba"]
    ejemplo = X.head(12)
    modelo = _modelo()
    probabilidades = modelo.predict_proba(ejemplo)
    assert set(modelo.classes_) == set(CLASES)
    assert probabilidades.shape == (len(ejemplo), len(CLASES))
    assert np.isfinite(probabilidades).all()
    assert ((0 <= probabilidades) & (probabilidades <= 1)).all()
    assert np.allclose(probabilidades.sum(axis=1), 1.0, rtol=0, atol=1e-8)
