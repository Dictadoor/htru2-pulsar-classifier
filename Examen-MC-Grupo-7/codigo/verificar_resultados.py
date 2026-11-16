"""Audita las salidas de la ejecución y la inferencia del modelo exportado."""
import json

import joblib
import numpy as np
import pandas as pd

from comparacion_ml import metrics
from modelo_random_forest import ROOT, FEATURES, DATA_FILE


def main():
    out = ROOT / "resultados"
    summary = pd.read_csv(out / "metricas_completas.csv")
    data = pd.read_csv(DATA_FILE)
    if len(summary) != 12:
        raise AssertionError("Se esperaban cuatro modelos por tres particiones.")
    parts = {}
    for part in summary.particion.unique():
        indices = pd.read_csv(out / "comparacion_ml" / f"indices_{part}.csv").fila_csv
        parts[part] = set(indices)
    train, val, test = parts["entrenamiento"], parts["validación"], parts["prueba"]
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(data.index)
    for number, name in enumerate(summary.modelo.unique(), 1):
        folder = out / f"modelo_{number}"
        model = joblib.load(folder / "modelo.joblib")
        for part in parts:
            pred = pd.read_csv(folder / f"predicciones_{part}.csv")
            assert set(pred.fila_csv) == parts[part]
            assert np.array_equal(pred.clase_real, data.loc[pred.fila_csv, "class"])
            x = data.loc[pred.fila_csv, FEATURES]
            np.testing.assert_array_equal(model.predict(x), pred.clase_predicha)
            np.testing.assert_allclose(model.predict_proba(x)[:, 1], pred.probabilidad_clase_1, atol=1e-12)
            actual = metrics(pred.clase_real, pred.clase_predicha, pred.probabilidad_clase_1)
            expected = summary[(summary.modelo == name) & (summary.particion == part)].iloc[0]
            for key, value in actual.items():
                if key != "matriz_confusion":
                    assert abs(value - expected[key]) < 1e-12, (name, part, key)
        if number in (3, 4):
            scaler = model.named_steps["estandarizador"]
            np.testing.assert_allclose(scaler.mean_, data.loc[sorted(train), FEATURES].mean().to_numpy())
    selection = json.loads((out / "seleccion.json").read_text(encoding="utf-8"))
    expected_winner = max(selection["validacion"], key=lambda name: (
        selection["validacion"][name]["exactitud_balanceada"], selection["validacion"][name]["f1"]))
    assert expected_winner == selection["modelo_elegido"]
    model_number = list(summary.modelo.unique()).index(expected_winner) + 1
    active = joblib.load(out / "modelo_funcional.joblib")
    selected = joblib.load(out / f"modelo_{model_number}" / "modelo.joblib")
    sample = data.loc[sorted(test)[:30], FEATURES]
    np.testing.assert_allclose(active.predict_proba(sample), selected.predict_proba(sample))
    network = joblib.load(out / "modelo_4" / "modelo.joblib").named_steps["red"]
    assert network.hidden_layer_sizes == (64, 32, 16)
    history = pd.read_csv(out / "historial_red.csv")
    assert np.isfinite(history.to_numpy()).all()
    for name in ("rendimiento_y_perdida.png", "roc_comparacion.png", "matrices_confusion.png"):
        assert (out / name).stat().st_size > 1000
    # Archivo de demostración sin etiqueta tomado de la partición de prueba.
    data.loc[sorted(test)[:50], FEATURES].to_csv(ROOT / "datos_demo_sin_etiqueta.csv", index=False)
    print("OK: 12 evaluaciones, particiones, escaladores, cuatro modelos serializados y exportación funcional.")


if __name__ == "__main__":
    main()
