import pytest
from fastapi.testclient import TestClient

from app.api import MODEL_PATH, app

EJEMPLO = {
    "texto_resena": "Muy conforme con la compra, la calidad es muy buena.",
    "categoria_producto": "Belleza", "canal_compra": "Online", "plataforma": "Flipkart",
    "importe_compra": 38901.9, "tiempo_respuesta_horas": 2.5, "problema_resuelto": "Sí",
    "reclamo_formal": "No", "edad": 44, "genero": "Femenino", "ciudad": "Salta",
    "antiguedad_cliente_meses": 70,
}


def test_health_responde_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_entrada_invalida_devuelve_422():
    with TestClient(app) as c:
        r = c.post("/predict", json={**EJEMPLO, "edad": 5})
        assert r.status_code == 422


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="modelo no disponible")
def test_prediccion_valida():
    with TestClient(app) as c:
        r = c.post("/predict", json=EJEMPLO)
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["sentimiento"] in {"Positivo", "Neutral", "Negativo"}
        assert abs(sum(cuerpo["probabilidades"].values()) - 1) < 1e-3
