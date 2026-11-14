from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/modelo.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "mvp-v2")

estado: dict = {"modelo": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        estado["modelo"] = joblib.load(MODEL_PATH)
    yield
    estado["modelo"] = None


app = FastAPI(
    title="API de predicción de sentimiento de reseñas",
    description="Servicio de apoyo al equipo de atención al cliente (13MBID).",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class ResenaEntrada(BaseModel):
    texto_resena: str = Field(..., min_length=3, max_length=2000)
    categoria_producto: str
    canal_compra: str
    plataforma: str
    importe_compra: float = Field(..., ge=0)
    tiempo_respuesta_horas: float = Field(..., ge=0)
    problema_resuelto: str
    reclamo_formal: str
    edad: float = Field(..., ge=16, le=110)
    genero: str
    ciudad: str
    antiguedad_cliente_meses: int = Field(..., ge=0)


class Prediccion(BaseModel):
    sentimiento: str
    probabilidades: dict[str, float]
    prioridad_atencion: bool
    version_modelo: str


def construir_features(r: ResenaEntrada) -> pd.DataFrame:
    """Reproduce las variables derivadas de src/preparación.py.

    VERIFICAR que coinciden con el preprocesado del entrenamiento.
    """
    fila = r.model_dump()
    fila["longitud_resena"] = len(r.texto_resena)
    fila["cantidad_palabras_resena"] = len(r.texto_resena.split())
    fila["tiempo_respuesta_dias"] = r.tiempo_respuesta_horas / 24
    return pd.DataFrame([fila])


@app.get("/")
def raiz() -> dict:
    return {"servicio": app.title, "version": MODEL_VERSION, "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"estado": "ok", "modelo_cargado": estado["modelo"] is not None, "version": MODEL_VERSION}


@app.post("/predict", response_model=Prediccion)
def predict(entrada: ResenaEntrada) -> Prediccion:
    modelo = estado["modelo"]
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en el servidor.")
    X = construir_features(entrada)
    try:
        etiqueta = str(modelo.predict(X)[0])
        proba = modelo.predict_proba(X)[0]
    except Exception as exc:  # columnas/valores incompatibles con el pipeline entrenado
        raise HTTPException(status_code=422, detail=f"No se pudo predecir: {exc}") from exc
    probs = {str(c): round(float(p), 4) for c, p in zip(modelo.classes_, proba)}
    return Prediccion(
        sentimiento=etiqueta,
        probabilidades=probs,
        prioridad_atencion=etiqueta.lower() == "negativo",
        version_modelo=MODEL_VERSION,
    )
