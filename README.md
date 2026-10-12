# Clasificación de candidatos a púlsar con HTRU2

Proyecto académico de clasificación binaria supervisada con Random Forest. El modelo recibe ocho características estadísticas del conjunto HTRU2 y estima si un registro está etiquetado como no púlsar (`class = 0`) o candidato a púlsar (`class = 1`). La predicción es una ayuda de clasificación; no confirma un descubrimiento astronómico.

## Dataset

El dataset no está incluido en este repositorio. Descárgalo desde [UCI Machine Learning Repository: HTRU2](https://archive.ics.uci.edu/dataset/372/htru2), extrae `HTRU_2.csv` en `datos_originales/` y genera el archivo con encabezados que consume el entrenamiento:

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

## Instalación y ejecución

Con Python compatible con las dependencias declaradas en `requirements.txt`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
py modelo_random_forest.py
py graficar_roc.py
py -m streamlit run app.py
```

El script de entrenamiento valida las columnas, los valores vacíos, los tipos numéricos, los valores finitos y las etiquetas. Después genera particiones estratificadas de entrenamiento, validación y prueba, entrena el modelo y guarda métricas, predicciones y el archivo del clasificador en `output/modelo_random_forest/`. El dashboard necesita que el modelo se haya generado antes.

Para clasificar desde el dashboard, carga un CSV con las ocho columnas predictoras y sus nombres exactos. `class` puede omitirse; si está presente, no se usa como entrada del modelo. La salida incluye la clase estimada y su probabilidad por registro.

## Relación con CRISP-ML(Q)

El informe adopta CRISP-ML(Q). El código no pretende ejecutar por sí solo todas sus fases; las tareas se distribuyen así:

1. **Comprensión del negocio y de los datos:** el informe establece el objetivo y el alcance. El script carga el HTRU2 seleccionado y revisa su estructura y distribución de clases.
2. **Ingeniería de datos:** el script valida los valores, separa las ocho variables predictoras de `class` y prepara las particiones estratificadas. No crea ni escala variables.
3. **Ingeniería del modelo:** `modelo_random_forest.py` configura y entrena `RandomForestClassifier`.
4. **Aseguramiento de calidad:** el script calcula las métricas y la matriz de confusión para validación y prueba. `graficar_roc.py` genera la curva ROC con las probabilidades de prueba; la prueba se reserva para la evaluación final.
5. **Despliegue:** `app.py` implementa la interfaz Streamlit desplegada, descrita en el informe. La configuración de infraestructura de la VPS no se incluye aquí.
6. **Monitoreo y mantenimiento:** no hay monitoreo automatizado ni reentrenamiento continuo; el informe los presenta como trabajo propuesto.

## Archivos principales

- `modelo_random_forest.py`: validación, partición, entrenamiento y evaluación.
- `graficar_roc.py`: generación de la curva ROC.
- `app.py`: interfaz Streamlit para clasificar CSV.
- `requirements.txt`: dependencias de Python.

Los datos descargados, el entorno virtual y los resultados generados se excluyen del repositorio.
