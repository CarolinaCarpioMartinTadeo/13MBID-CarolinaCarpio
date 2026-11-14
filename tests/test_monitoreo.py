import numpy as np
import pandas as pd

from src import monitoreo_drift as m


def _df(media_horas: float, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "importe_compra": rng.gamma(2.0, 12000, n),
        "tiempo_respuesta_horas": rng.gamma(2.0, media_horas / 2, n),
        "edad": rng.integers(18, 75, n),
        "antiguedad_cliente_meses": rng.integers(1, 96, n),
        "categoria_producto": rng.choice(["A", "B", "C"], n),
        "canal_compra": rng.choice(["Online", "Offline"], n, p=[0.8, 0.2]),
        "plataforma": rng.choice(["X", "Y"], n),
        "genero": rng.choice(["F", "M"], n),
        "problema_resuelto": rng.choice(["Sí", "No"], n, p=[0.85, 0.15]),
        "reclamo_formal": rng.choice(["Sí", "No"], n, p=[0.2, 0.8]),
        "sentimiento": rng.choice(["Positivo", "Neutral", "Negativo"], n, p=[0.66, 0.17, 0.17]),
    })


def test_sin_deriva_con_misma_distribucion():
    informe = m.analizar(_df(13, seed=1), _df(13, seed=2))
    assert informe["resumen"]["variables_criticas_con_deriva"] == []


def test_detecta_deriva_en_tiempo_de_respuesta():
    informe = m.analizar(_df(13, seed=1), _df(24, seed=2))
    assert "tiempo_respuesta_horas" in informe["resumen"]["variables_criticas_con_deriva"]


def test_columnas_faltantes_lanzan_error():
    import pytest
    with pytest.raises(ValueError):
        m.analizar(_df(13), _df(13).drop(columns=["edad"]))


def test_codigo_de_salida(tmp_path):
    ref, nue = tmp_path / "ref.csv", tmp_path / "nue.csv"
    _df(13, seed=1).to_csv(ref, index=False)
    _df(24, seed=2).to_csv(nue, index=False)
    salida = tmp_path / "informe.json"
    assert m.main(["--referencia", str(ref), "--nuevos", str(nue), "--salida", str(salida)]) == 1
    assert salida.exists()
