"""Diagnóstico de los datos originales e informe automatizado de calidad."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pandera.pandas as pa

from esquemas import esquema_clientes, esquema_resenas


RAIZ = Path(__file__).resolve().parent.parent


def diagnostico_inicial(clientes, resenas):
    for archivo, datos, columna_id in [
        ("datos_clientes.csv", clientes, "id_cliente"),
        ("datos_resenas.csv", resenas, "id_resena"),
    ]:
        print(f"\n--- {archivo} ---")
        print(f"Total de filas: {len(datos)}")
        nulos = datos.isna().sum()
        informe = pd.DataFrame({
            "Nulos": nulos,
            "Nulos (%)": (nulos / len(datos) * 100).round(2),
        })
        print(informe.to_string())
        print("Filas duplicadas exactas:", datos.duplicated().sum())
        print(
            f"Repeticiones de {columna_id} tras la primera aparición:",
            datos[columna_id].duplicated().sum(),
        )

    controles = {
        "Edades presentes fuera de 18-79":
            clientes["edad"].notna() & ~clientes["edad"].between(18, 79),
        "Antigüedades presentes fuera de 1-95 meses":
            clientes["antiguedad_cliente_meses"].notna()
            & ~clientes["antiguedad_cliente_meses"].between(1, 95),
        "Calificaciones presentes fuera de 1-5":
            resenas["calificacion"].notna()
            & ~resenas["calificacion"].between(1, 5),
        "Tiempos de respuesta negativos":
            resenas["tiempo_respuesta_horas"].lt(0),
        "Importes de compra negativos": resenas["importe_compra"].lt(0),
        "Fechas ausentes o no interpretables": pd.to_datetime(
            resenas["fecha_resena"], format="%Y-%m-%d", errors="coerce",
        ).isna(),
        "Textos ausentes o vacíos":
            resenas["texto_resena"].fillna("").str.strip().eq(""),
        "Reseñas sin cliente asociado": ~resenas["id_cliente"].isin(
            clientes["id_cliente"].dropna(),
        ),
    }
    print("\n--- Comprobaciones adicionales ---")
    for nombre, incidencias in controles.items():
        print(f"{nombre}: {int(incidencias.sum())}")


def evaluar_con_pandera(clientes, resenas, carpeta_informes):
    errores = []
    resumen_tablas = []
    columnas_fallos = [
        "tabla", "schema_context", "column", "check", "check_number",
        "failure_case", "index",
    ]

    for nombre, datos, esquema in [
        ("clientes", clientes, esquema_clientes()),
        (
            "resenas", resenas,
            esquema_resenas(clientes["id_cliente"].dropna()),
        ),
    ]:
        print(f"\n--- Validación Pandera: {nombre} ---")
        try:
            esquema.validate(datos, lazy=True)
        except pa.errors.SchemaErrors as error:
            fallos = error.failure_cases.copy()
            fallos.insert(0, "tabla", nombre)
            errores.append(fallos)
            estado = "REQUIERE_TRATAMIENTO"
            casos_fallo = len(fallos)
            print(fallos.groupby(["column", "check"], dropna=False).size())
        else:
            estado = "CUMPLE"
            casos_fallo = 0
            print("Cumple todas las reglas del esquema.")

        resumen_tablas.append({
            "tabla": nombre,
            "filas": len(datos),
            "columnas": len(datos.columns),
            "estado": estado,
            "casos_de_fallo": casos_fallo,
        })

    detalle = (
        pd.concat(errores, ignore_index=True)
        if errores else pd.DataFrame(columns=columnas_fallos)
    )
    resumen = pd.DataFrame(resumen_tablas)
    detalle.to_csv(
        carpeta_informes / "incidencias_pandera.csv",
        index=False, encoding="utf-8-sig",
    )
    resumen.to_csv(
        carpeta_informes / "resumen_tablas.csv",
        index=False, encoding="utf-8-sig",
    )
    print("\n--- Estado de calidad de los datos originales ---")
    print(resumen.to_string(index=False))
    print("Los casos de fallo cuentan incumplimientos de reglas por valor;")
    print("una misma fila puede incumplir más de una regla.")


def main():
    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv", sep=";",
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv", sep=";",
    )
    carpeta_informes = RAIZ / "docs" / "reports" / "calidad"
    carpeta_informes.mkdir(parents=True, exist_ok=True)

    # Se evalúan ambos esquemas antes de calcular estadísticas específicas.
    evaluar_con_pandera(clientes, resenas, carpeta_informes)
    diagnostico_inicial(clientes, resenas)

    print("\n--- Pruebas del comportamiento de los controles ---", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(RAIZ / "tests" / "test_calidad.py"),
            "-q", "--junitxml", str(carpeta_informes / "pytest.xml"),
        ],
        cwd=RAIZ,
        check=True,
    )
    print(f"\nInformes guardados en: {carpeta_informes}")
    print("El informe Pandera muestra la calidad de los datos originales.")
    print("Pytest comprueba la detección de anomalías con casos controlados.")


if __name__ == "__main__":
    main()