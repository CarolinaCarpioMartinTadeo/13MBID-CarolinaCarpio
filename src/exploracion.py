from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAIZ = Path(__file__).resolve().parent.parent


def main():
    clientes = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_clientes.csv", sep=";"
    )
    resenas = pd.read_csv(
        RAIZ / "data" / "raw" / "datos_resenas.csv", sep=";"
    )

    print(f"Clientes: {clientes.shape[0]} filas y {clientes.shape[1]} columnas.")
    print(f"Reseñas: {resenas.shape[0]} filas y {resenas.shape[1]} columnas.")

    carpeta_figuras = RAIZ / "docs" / "figures"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    orden = ["Negativo", "Neutral", "Positivo"]
    conteos = resenas["sentimiento"].value_counts().reindex(orden, fill_value=0)
    porcentajes = (conteos / len(resenas) * 100).round(2)

    print("\nDistribución del sentimiento:")
    print(pd.DataFrame({"Registros": conteos, "Porcentaje": porcentajes}))

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(
        conteos.index, conteos.values,
        color=["#C97973", "#B3B3B3", "#70A68D"],
    )
    ax.bar_label(barras, padding=4)
    ax.set_title("Distribución del sentimiento de las reseñas")
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Número de registros")
    ax.set_ylim(0, conteos.max() * 1.15)
    fig.tight_layout()

    destino = carpeta_figuras / "01_distribucion_sentimiento.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)

    print(f"\nGráfico guardado en: {destino}")

    # Longitud de las reseñas según su sentimiento.
    longitudes = resenas["texto_resena"].str.len()
    grupos = [
        longitudes[resenas["sentimiento"] == categoria].dropna()
        for categoria in orden
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    cajas = ax.boxplot(
        grupos,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for caja, color in zip(
        cajas["boxes"], ["#C97973", "#B3B3B3", "#70A68D"]
    ):
        caja.set_facecolor(color)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(orden)
    ax.set_title("Longitud de las reseñas según el sentimiento")
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Longitud del texto (caracteres)")
    fig.tight_layout()

    destino = carpeta_figuras / "02_longitud_por_sentimiento.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)
    print(f"\nGráfico guardado en: {destino}")

    resumen = (
        resenas.assign(longitud_texto=longitudes)
        .groupby("sentimiento")["longitud_texto"]
        .agg(["count", "median", "mean", "min", "max"])
        .reindex(orden)
        .round(2)
    )
    print("\nLongitud del texto por sentimiento:")
    print(resumen)
    
        # Distribuciones de las variables categóricas.
    conjuntos = [
        ("clientes", clientes, ["genero", "ciudad", "canal_registro"]),
        (
            "resenas",
            resenas,
            [
                "categoria_producto",
                "canal_compra",
                "plataforma",
                "problema_resuelto",
                "reclamo_formal",
            ],
        ),
    ]

    numero = 3
    for nombre, datos, columnas in conjuntos:
        for columna in columnas:
            frecuencias = datos[columna].value_counts(dropna=False)
            porcentajes = (frecuencias / len(datos) * 100).round(2)

            print(f"\n{nombre} - {columna}:")
            print(pd.DataFrame({
                "Registros": frecuencias,
                "Porcentaje": porcentajes,
            }))

            fig, ax = plt.subplots(
                figsize=(9, max(4, len(frecuencias) * 0.45))
            )
            barras = ax.barh(
                frecuencias.index.astype(str),
                frecuencias.values,
                color="#7699A6",
            )
            ax.invert_yaxis()
            ax.bar_label(barras, padding=4)
            ax.set_xlim(0, frecuencias.max() * 1.18)
            ax.set_title(f"Distribución de {columna} ({nombre})")
            ax.set_xlabel("Número de registros")
            ax.set_ylabel(columna)
            fig.tight_layout()

            destino = carpeta_figuras / f"{numero:02d}_{nombre}_{columna}.png"
            fig.savefig(destino, dpi=200)
            plt.close(fig)
            print(f"Gráfico guardado en: {destino}")
            numero += 1
        # Correlaciones entre variables numéricas.
    matrices = [
        ("clientes", clientes, ["edad", "antiguedad_cliente_meses"]),
        (
            "resenas",
            resenas,
            ["calificacion", "tiempo_respuesta_horas", "importe_compra"],
        ),
    ]

    for numero, (nombre, datos, columnas) in enumerate(matrices, start=11):
        correlacion = datos[columnas].corr(method="pearson")

        print(f"\nMatriz de correlaciones - {nombre}:")
        print(correlacion.round(4).to_string())

        fig, ax = plt.subplots(figsize=(8, 6))
        imagen = ax.imshow(
            correlacion,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        posiciones = range(len(columnas))
        ax.set_xticks(posiciones)
        ax.set_yticks(posiciones)
        ax.set_xticklabels(columnas, rotation=30, ha="right")
        ax.set_yticklabels(columnas)

        for i in posiciones:
            for j in posiciones:
                valor = correlacion.iloc[i, j]
                ax.text(
                    j, i, f"{valor:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(valor) > 0.5 else "black",
                )

        ax.set_title(f"Correlaciones de Pearson ({nombre})")
        fig.colorbar(imagen, ax=ax, label="Correlación de Pearson")
        fig.tight_layout()

        destino = carpeta_figuras / f"{numero:02d}_correlaciones_{nombre}.png"
        fig.savefig(destino, dpi=200)
        plt.close(fig)
        print(f"Gráfico guardado en: {destino}")
    
        # Análisis propios: sentimiento por canal y categoría de producto.
    analisis_propios = [
        ("canal_compra", "Canal de compra"),
        ("categoria_producto", "Categoría de producto"),
    ]

    for numero, (columna, etiqueta) in enumerate(analisis_propios, start=13):
        conteos_grupo = pd.crosstab(
            resenas[columna], resenas["sentimiento"]
        ).reindex(columns=orden, fill_value=0)

        totales = conteos_grupo.sum(axis=1)
        porcentajes_grupo = conteos_grupo.div(totales, axis=0) * 100

        tabla = pd.DataFrame({
            "Total": totales,
            "Negativas": conteos_grupo["Negativo"],
            "Negativas (%)": porcentajes_grupo["Negativo"],
        }).sort_values("Negativas (%)", ascending=False)

        porcentajes_grupo = porcentajes_grupo.reindex(tabla.index)
        print(f"\nAnálisis propio - {columna}:")
        print(tabla.round(2).to_string())

        fig, ax = plt.subplots(figsize=(10, max(4.5, len(tabla) * 0.6)))
        porcentajes_grupo.plot.barh(
            stacked=True,
            color=["#C97973", "#B3B3B3", "#70A68D"],
            ax=ax,
        )
        ax.invert_yaxis()
        for contenedor in ax.containers:
            etiquetas = [
                f"{barra.get_width():.1f} %"
                if barra.get_width() >= 5 else ""
                for barra in contenedor
            ]
            ax.bar_label(
                contenedor, labels=etiquetas,
                label_type="center", fontsize=9,
            )

        ax.set_xlim(0, 100)
        ax.set_title(f"Sentimiento según {etiqueta.lower()}")
        ax.set_xlabel("Porcentaje de reseñas dentro de cada grupo")
        ax.set_ylabel(etiqueta)
        ax.legend(
            title="Sentimiento", ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, -0.20),
        )
        fig.tight_layout()

        destino = carpeta_figuras / f"{numero:02d}_sentimiento_por_{columna}.png"
        fig.savefig(destino, dpi=200)
        plt.close(fig)
        print(f"Gráfico guardado en: {destino}")
        
if __name__ == "__main__":
    main()