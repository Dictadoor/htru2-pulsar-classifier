"""Entrena y evalúa Random Forest para HTRU2 siguiendo las tareas de CRISP-ML(Q)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "datos_preparados" / "HTRU_2_con_encabezados.csv"
OUTPUT_DIR = ROOT / "output" / "modelo_random_forest"
RANDOM_STATE = 42

FEATURES = [
    "Profile_mean",
    "Profile_stdev",
    "Profile_skewness",
    "Profile_kurtosis",
    "DM_mean",
    "DM_stdev",
    "DM_skewness",
    "DM_kurtosis",
]
TARGET = "class"
CLASS_NAMES = {0: "No pulsar", 1: "Candidato a pulsar"}


def validate_data(data: pd.DataFrame) -> dict[str, int]:
    """Comprueba estructura y calidad sin eliminar filas silenciosamente."""
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing_columns}")

    data = data[required_columns]
    missing_cells = int(data.isna().sum().sum())
    if missing_cells:
        raise ValueError(
            f"Se encontraron {missing_cells} valores vacíos. Revise el CSV antes de entrenar; "
            "el programa no los elimina ni los imputa automáticamente."
        )

    try:
        numeric_data = data.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Las ocho características y la etiqueta deben ser numéricas.") from error

    if not np.isfinite(numeric_data.to_numpy()).all():
        raise ValueError("El CSV contiene valores infinitos o no finitos.")

    labels = set(numeric_data[TARGET].unique())
    if not labels.issubset({0.0, 1.0}) or labels != {0.0, 1.0}:
        raise ValueError(f"La columna class debe contener las clases 0 y 1; se encontró {labels}.")

    return {
        "filas": int(len(data)),
        "columnas_requeridas": len(required_columns),
        "celdas_vacias": missing_cells,
        "filas_duplicadas_exactas": int(data.duplicated().sum()),
    }


def get_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula medidas globales y medidas para la clase minoritaria (1)."""
    return {
        "exactitud": float(accuracy_score(y_true, y_pred)),
        "exactitud_balanceada": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_clase_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_clase_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_clase_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de datos: {DATA_FILE}\n"
            "Coloque HTRU_2_con_encabezados.csv dentro de datos_preparados."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CRISP-ML(Q), comprensión de datos: carga el HTRU2 seleccionado y revisa su calidad.
    data = pd.read_csv(DATA_FILE)
    quality = validate_data(data)
    data = data[FEATURES + [TARGET]].astype(float)
    data[TARGET] = data[TARGET].astype(int)

    print("=== CRISP-ML(Q): comprensión y revisión de datos ===")
    print(f"Archivo: {DATA_FILE}")
    print(f"Registros: {quality['filas']:,}")
    print(f"Filas duplicadas exactas (solo auditadas): {quality['filas_duplicadas_exactas']}")
    print("Distribución de clases:")
    print(data[TARGET].value_counts().sort_index().rename(index=CLASS_NAMES).to_string())

    # CRISP-ML(Q), ingeniería de datos: se definen X e y y se preparan las particiones.
    # Se conserva la escala original; Random Forest no requiere estandarización.
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    # CRISP-ML(Q), ingeniería del modelo: entrenamiento del clasificador supervisado.
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # CRISP-ML(Q), aseguramiento de calidad: evalúa validación y prueba.
    # La prueba se reserva para informar el resultado final.
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)
    y_test_probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    validation_metrics = get_metrics(y_validation, y_validation_pred)
    test_metrics = get_metrics(y_test, y_test_pred)
    test_confusion = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )

    metrics_output = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "archivo_datos": str(DATA_FILE.relative_to(ROOT)),
        "version_scikit_learn": sklearn.__version__,
        "semilla": RANDOM_STATE,
        "distribucion": {
            "entrenamiento": int(len(y_train)),
            "validacion": int(len(y_validation)),
            "prueba": int(len(y_test)),
        },
        "auditoria": quality,
        "parametros_modelo": model.get_params(),
        "metricas_validacion": validation_metrics,
        "metricas_prueba": test_metrics,
        "matriz_confusion_prueba": test_confusion.tolist(),
        "reporte_clasificacion_prueba": report,
    }

    (OUTPUT_DIR / "metricas.json").write_text(
        json.dumps(metrics_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        test_confusion,
        index=["Real 0 (no pulsar)", "Real 1 (candidato a pulsar)"],
        columns=["Predicho 0", "Predicho 1"],
    ).to_csv(OUTPUT_DIR / "matriz_confusion.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "clase_real": y_test.to_numpy(),
            "clase_predicha": y_test_pred,
            "probabilidad_candidato_pulsar": y_test_probability,
        },
        index=y_test.index,
    ).sort_index().to_csv(OUTPUT_DIR / "predicciones_prueba.csv", index_label="fila_csv")
    pd.DataFrame(
        {
            "variable": FEATURES,
            "importancia_random_forest": model.feature_importances_,
        }
    ).sort_values("importancia_random_forest", ascending=False).to_csv(
        OUTPUT_DIR / "importancia_variables.csv", index=False
    )
    joblib.dump(model, OUTPUT_DIR / "modelo_random_forest.joblib")

    print("\n=== Resultados en el conjunto de prueba ===")
    print(f"Exactitud:            {test_metrics['exactitud']:.4f}")
    print(f"Exactitud balanceada: {test_metrics['exactitud_balanceada']:.4f}")
    print(f"Precisión clase 1:    {test_metrics['precision_clase_1']:.4f}")
    print(f"Recall clase 1:       {test_metrics['recall_clase_1']:.4f}")
    print(f"F1 clase 1:           {test_metrics['f1_clase_1']:.4f}")
    print("\nMatriz de confusión (filas = clase real; columnas = clase predicha):")
    print(pd.DataFrame(test_confusion, index=["Real 0", "Real 1"], columns=["Predicho 0", "Predicho 1"]))
    print(f"\nArchivos generados en: {OUTPUT_DIR}")
    print("La importancia de variables es descriptiva; no demuestra causalidad ni selecciona variables automáticamente.")


if __name__ == "__main__":
    main()
