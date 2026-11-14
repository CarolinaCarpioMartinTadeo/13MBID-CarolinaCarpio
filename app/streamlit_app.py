import os

import requests
import streamlit as st

st.set_page_config(page_title="Sentimiento de reseñas", page_icon="💬")


def url_api() -> str:
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


CATEGORIAS = ["Alimentos", "Belleza", "Calzado", "Deportes", "Electrónica", "Hogar", "Indumentaria", "Libros"]
PLATAFORMAS = ["Amazon", "Falabella", "Flipkart", "JioMart", "MercadoLibre", "Meesho", "Nykaa", "Zepto"]
CIUDADES = ["Buenos Aires", "Córdoba", "La Plata", "Mendoza", "Rosario", "Salta", "Santa Fe", "Otra"]

st.title("💬 Análisis de sentimiento de reseñas")
st.caption("Apoyo a la priorización de atención al cliente · 13MBID")

with st.form("formulario"):
    texto = st.text_area("Texto de la reseña", height=120, placeholder="Escribe o pega la reseña del cliente…")
    c1, c2 = st.columns(2)
    categoria = c1.selectbox("Categoría del producto", CATEGORIAS)
    plataforma = c2.selectbox("Plataforma", PLATAFORMAS)
    canal = c1.selectbox("Canal de compra", ["Online", "Offline"])
    ciudad = c2.selectbox("Ciudad", CIUDADES)
    importe = c1.number_input("Importe de la compra", min_value=0.0, value=10000.0, step=100.0)
    horas = c2.number_input("Tiempo de respuesta (horas)", min_value=0.0, value=12.0, step=0.5)
    resuelto = c1.selectbox("¿Problema resuelto?", ["Sí", "No"])
    reclamo = c2.selectbox("¿Reclamo formal?", ["No", "Sí"])
    edad = c1.number_input("Edad del cliente", min_value=16, max_value=110, value=35)
    genero = c2.selectbox("Género", ["Femenino", "Masculino", "Otro"])
    antig = st.number_input("Antigüedad del cliente (meses)", min_value=0, value=24)
    enviar = st.form_submit_button("Predecir sentimiento")

if enviar:
    if len(texto.strip()) < 3:
        st.warning("Introduce el texto de la reseña.")
    else:
        carga = {
            "texto_resena": texto.strip(), "categoria_producto": categoria, "canal_compra": canal,
            "plataforma": plataforma, "importe_compra": importe, "tiempo_respuesta_horas": horas,
            "problema_resuelto": resuelto, "reclamo_formal": reclamo, "edad": float(edad),
            "genero": genero, "ciudad": ciudad, "antiguedad_cliente_meses": int(antig),
        }
        try:
            with st.spinner("Consultando el modelo… (la primera llamada puede tardar si el servicio estaba dormido)"):
                resp = requests.post(f"{url_api()}/predict", json=carga, timeout=90)
            resp.raise_for_status()
            res = resp.json()
        except requests.RequestException as e:
            st.error(f"No se pudo contactar con la API: {e}")
        else:
            if res["prioridad_atencion"]:
                st.error(f"Sentimiento previsto: **{res['sentimiento']}** → priorizar atención")
            else:
                st.success(f"Sentimiento previsto: **{res['sentimiento']}**")
            st.bar_chart(res["probabilidades"])
            st.caption(f"Versión del modelo: {res['version_modelo']}")
