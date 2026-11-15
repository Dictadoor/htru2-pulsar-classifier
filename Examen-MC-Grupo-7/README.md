# Examen de Medio Ciclo — Grupo 7

Comparación de cuatro modelos para clasificar candidatos de HTRU2. Esta carpeta incluye el código, los datos de entrada, la referencia histórica y los resultados de la ejecución del examen.

## Instalación

La ejecución registrada utiliza Python 3.14.0 y scikit-learn 1.9.1. Abrir la terminal en esta carpeta e instalar las dependencias.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux, crear el entorno con `python3.14 -m venv .venv` y usar `.venv/bin/python` en los comandos siguientes.

## Ejecutar el experimento

```powershell
.\.venv\Scripts\python.exe codigo\entrenar_examen.py
.\.venv\Scripts\python.exe codigo\verificar_examen.py
```

El entrenamiento genera las métricas de las tres particiones, las curvas ROC individuales y comparadas, las matrices de confusión y las curvas de aprendizaje de la red. El verificador recalcula las doce evaluaciones y comprueba las predicciones de los modelos guardados. Los resultados de la ejecución utilizada en el informe acompañan esta carpeta.

## Ejecutar el dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abrir http://localhost:8501 y cargar `datos_demo_sin_etiqueta.csv` para probar la interfaz. El selector ofrece los cuatro modelos. La MLP aparece como opción inicial por el criterio de validación. El dashboard aplica los modelos guardados al CSV de entrada y permite descargar las predicciones.

El CSV de entrada utiliza estas ocho columnas, en el esquema que conserva el código.

```text
Profile_mean,Profile_stdev,Profile_skewness,Profile_kurtosis,DM_mean,DM_stdev,DM_skewness,DM_kurtosis
```

La interfaz excluye `class` de las entradas cuando el archivo la contiene. La clase 1 representa una predicción de candidato a púlsar y requiere interpretación astronómica especializada.

## Modelos y selección

| Modelo | Configuración |
| --- | --- |
| Random Forest original | Referencia histórica de 300 árboles con pesos de clases balanceados. |
| Random Forest modificado | 150 árboles. Compara profundidades 5, 10 y 15 mediante validación. |
| Regresión logística | StandardScaler y regresión logística con pesos balanceados. Compara C de 0,01, 0,1, 1 y 10. |
| MLP profunda | Ocho entradas, capas ocultas de 64, 32 y 16 neuronas y una salida logística. Adam, ReLU, lote 128 y tasa inicial 0,001. |

La regresión logística y la red ajustan StandardScaler exclusivamente con entrenamiento. La red utiliza pesos por muestra para compensar el desbalance. Las particiones estratificadas conservan la proporción 70/15/15 y la semilla 42.

La selección prioriza exactitud balanceada de validación y utiliza F1 como desempate. La red completó 55 épocas y recuperó el modelo de la época 25. La diferencia de validación frente al bosque modificado es pequeña. El conjunto de prueba procede de las prácticas anteriores y el grupo ya había consultado sus resultados.

Las curvas de la red muestran exactitud, exactitud balanceada y entropía cruzada binaria sin ponderar en entrenamiento y validación. El objetivo interno de entrenamiento incorpora los pesos por muestra y la regularización L2.

## Estructura

- `codigo/entrenar_examen.py` construye y compara los cuatro modelos.
- `codigo/comparacion_ml.py` contiene la preparación de particiones, métricas y selección de configuraciones ML.
- `codigo/modelo_random_forest.py` contiene las variables y la validación del dataset.
- `codigo/verificar_examen.py` comprueba las salidas y los modelos guardados.
- `datos_originales/` conserva los archivos de origen y su documentación.
- `datos_preparados/` contiene el CSV con encabezados que consume el programa.
- `modelo_historico/` conserva el bosque original y sus resultados de referencia.
- `resultados/` contiene métricas, predicciones, modelos serializados y figuras.
- `app.py` implementa la interfaz Streamlit.

## Fuentes

- [Dataset HTRU2](https://archive.ics.uci.edu/dataset/372/htru2).
- [Caso de estudio de Lyon et al.](https://doi.org/10.1093/mnras/stw656).
- [Dashboard público](https://htru2.bananacorp.dev).
