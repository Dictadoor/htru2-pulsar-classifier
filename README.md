# HTRU2: clasificador de candidatos a púlsar

Proyecto académico de clasificación binaria supervisada con Random Forest. El modelo usa las ocho características estadísticas de HTRU2 y la etiqueta `class` (0 = no púlsar, 1 = candidato a púlsar). Una predicción del modelo es una clasificación automática; no confirma por sí sola un púlsar.

## Dataset

Descarga HTRU2 desde [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/372/htru2). El dataset no está incluido en este repositorio. Extrae `HTRU_2.csv` en `datos_originales/` y crea el CSV con encabezados que consume el entrenamiento:

```python
from pathlib import Path
import pandas as pd

columnas = [
    "Profile_mean", "Profile_stdev", "Profile_skewness", "Profile_kurtosis",
    "DM_mean", "DM_stdev", "DM_skewness", "DM_kurtosis", "class",
]
Path("datos_preparados").mkdir(exist_ok=True)
datos = pd.read_csv("datos_originales/HTRU_2.csv", header=None, names=columnas)
datos.to_csv("datos_preparados/HTRU_2_con_encabezados.csv", index=False)
```

## Instalación y ejecución en Windows

Se requiere Python compatible con las dependencias de `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
py modelo_random_forest.py
py graficar_roc.py
py -m streamlit run app.py
```

`modelo_random_forest.py` valida los datos, crea particiones estratificadas de entrenamiento, validación y prueba, entrena el clasificador y guarda resultados y el modelo en `output/modelo_random_forest/`. `graficar_roc.py` genera la curva ROC a partir de las probabilidades del conjunto de prueba. El dashboard necesita que el modelo ya se haya generado.

Para usar el dashboard, carga un CSV que contenga las ocho columnas predictoras con los nombres anteriores. La columna `class` es opcional y no se usa para predecir. La aplicación devuelve la clase estimada y la probabilidad por registro.

## Archivos principales

- `modelo_random_forest.py`: validación, partición, entrenamiento y evaluación.
- `graficar_roc.py`: generación de la curva ROC.
- `app.py`: interfaz Streamlit para clasificar archivos CSV.
- `requirements.txt`: dependencias de Python.

Los datos descargados, modelos generados, resultados y entornos locales están excluidos del repositorio.
