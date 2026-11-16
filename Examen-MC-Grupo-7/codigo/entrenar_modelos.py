"""Compara tres modelos ML y una red profunda sobre HTRU2.

Ejecutar desde la carpeta de entrega: python codigo/entrenar_modelos.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, log_loss, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from modelo_random_forest import ROOT, FEATURES, DATA_FILE, RANDOM_STATE
from comparacion_ml import OUT as ML_OUT, REFERENCE, prepare_data, metrics, predict, select_on_validation

OUT = ROOT / "resultados"
NAMES = ["RF original", "RF modificado", "Regresión logística", "MLP profunda"]


def guardar_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def entrenar_red(splits):
    """Ingeniería del modelo: tres capas ocultas y control por validación externa."""
    x_train, y_train = splits["entrenamiento"]
    x_val, y_val = splits["validación"]
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    val_scaled = scaler.transform(x_val)
    red = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
        learning_rate_init=0.001, batch_size=128, alpha=0.001,
        random_state=RANDOM_STATE, shuffle=True, early_stopping=False,
    )
    pesos = compute_sample_weight("balanced", y_train)
    history = []
    best_key = (-1.0, -1.0)
    best_model = None
    best_epoch = 0
    patience = 30
    maximum_epochs = 200
    started = time.perf_counter()
    # partial_fit conserva los pesos y el estado de Adam entre épocas.
    for epoch in range(1, maximum_epochs + 1):
        red.partial_fit(train_scaled, y_train, classes=np.array([0, 1]), sample_weight=pesos)
        train_prob = red.predict_proba(train_scaled)[:, 1]
        val_prob = red.predict_proba(val_scaled)[:, 1]
        train_result = metrics(y_train, red.predict(train_scaled), train_prob)
        val_result = metrics(y_val, red.predict(val_scaled), val_prob)
        history.append({
            "epoca": epoch,
            "perdida_entrenamiento": float(log_loss(y_train, train_prob, labels=[0, 1])),
            "perdida_validacion": float(log_loss(y_val, val_prob, labels=[0, 1])),
            "exactitud_entrenamiento": train_result["exactitud"],
            "exactitud_validacion": val_result["exactitud"],
            "balanceada_entrenamiento": train_result["exactitud_balanceada"],
            "balanceada_validacion": val_result["exactitud_balanceada"],
            "f1_validacion": val_result["f1"],
            "objetivo_interno_ponderado": float(red.loss_),
        })
        key = (val_result["exactitud_balanceada"], val_result["f1"])
        if key > best_key:
            best_key = key
            best_model = copy.deepcopy(red)
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(f"Época {epoch}: BA validación={key[0]:.4f}, F1={key[1]:.4f}", flush=True)
        if epoch - best_epoch >= patience:
            break
    pipeline = Pipeline([("estandarizador", scaler), ("red", best_model)])
    config = {
        "arquitectura": [8, 64, 32, 16, 1],
        "activacion_ocultas": "relu", "activacion_salida": "logística",
        "optimizador": "Adam", "tasa_aprendizaje": 0.001,
        "batch_size": 128, "alpha_L2": 0.001, "semilla": RANDOM_STATE,
        "epocas_maximas": maximum_epochs, "paciencia": patience,
        "epoca_elegida": best_epoch, "epocas_ejecutadas": len(history),
        "criterio_checkpoint": "exactitud balanceada de validación; desempate F1",
        "pesos_muestras": "inversamente proporcionales a frecuencia de clase en entrenamiento",
        "perdida_curvas": "entropía cruzada binaria sin ponderar, calculada al terminar cada época",
        "objetivo_optimizador": "entropía cruzada ponderada más regularización L2",
        "parametros_entrenables": int(sum(w.size for w in best_model.coefs_) + sum(b.size for b in best_model.intercepts_)),
    }
    return pipeline, config, pd.DataFrame(history), time.perf_counter() - started


def graficar_aprendizaje(history, epoch):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, prefix, title in zip(axes, ["exactitud", "balanceada", "perdida"],
                                 ["Exactitud", "Exactitud balanceada", "Pérdida BCE sin ponderar"]):
        ax.plot(history.epoca, history[f"{prefix}_entrenamiento"], label="Entrenamiento")
        ax.plot(history.epoca, history[f"{prefix}_validacion"], label="Validación")
        ax.axvline(epoch, color="gray", linestyle="--", label="Época elegida")
        ax.set(title=title, xlabel="Época")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "rendimiento_y_perdida.png", dpi=220)
    plt.close(fig)


def main():
    """CRISP-ML(Q): validar, preparar, entrenar, evaluar y exportar el modelo."""
    OUT.mkdir(exist_ok=True)
    ML_OUT.mkdir(parents=True, exist_ok=True)
    quality, splits = prepare_data()
    baseline = joblib.load(REFERENCE / "modelo_random_forest.joblib")
    parameters = json.loads((REFERENCE / "metricas.json").read_text(encoding="utf-8"))["parametros_modelo"]
    if not isinstance(baseline, RandomForestClassifier) or baseline.get_params() != parameters:
        raise ValueError("El modelo histórico no coincide con su configuración registrada.")
    archived = pd.read_csv(REFERENCE / "predicciones_prueba.csv").set_index("fila_csv")
    x_test, y_test = splits["prueba"]
    old_pred, old_prob = predict(baseline, x_test)
    aligned = archived.loc[x_test.index]
    if not (np.array_equal(y_test.to_numpy(), aligned.clase_real)
            and np.array_equal(old_pred, aligned.clase_predicha)
            and np.allclose(old_prob, aligned.probabilidad_candidato_pulsar, atol=1e-14, rtol=0)):
        raise ValueError("Las particiones no reproducen la prueba histórica.")
    rf_candidates = []
    for depth in (5, 10, 15):
        config = {**parameters, "n_estimators": 150, "max_depth": depth}
        rf_candidates.append((config, RandomForestClassifier(**config)))
    rf, rf_config, rf_time, rf_trials = select_on_validation(rf_candidates, splits, NAMES[1])
    lr_candidates = []
    for c in (0.01, 0.1, 1.0, 10.0):
        lr = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-4,
                                class_weight="balanced", random_state=RANDOM_STATE)
        lr_candidates.append((lr.get_params(), Pipeline([("estandarizador", StandardScaler()), ("logistica", lr)])))
    lr, lr_config, lr_time, lr_trials = select_on_validation(lr_candidates, splits, NAMES[2])
    dnn, dnn_config, history, dnn_time = entrenar_red(splits)
    models = dict(zip(NAMES, [baseline, rf, lr, dnn]))
    configs = dict(zip(NAMES, [parameters, rf_config, lr_config, dnn_config]))
    xv, yv = splits["validación"]
    validation_results = {}
    for name, model in models.items():
        pred, prob = predict(model, xv)
        validation_results[name] = metrics(yv, pred, prob)
    winner = max(NAMES, key=lambda name: (validation_results[name]["exactitud_balanceada"], validation_results[name]["f1"]))
    # Esta elección se guarda antes de evaluar la prueba de los modelos nuevos.
    guardar_json(OUT / "seleccion.json", {
        "criterio": "exactitud balanceada de validación; desempate F1; orden declarado",
        "modelo_elegido": winner, "configuraciones": configs,
        "validacion": validation_results, "busqueda_ml": rf_trials + lr_trials,
    })
    rows = []
    roc_fig, roc_axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cm_fig, cm_axes = plt.subplots(2, 2, figsize=(9, 8))
    for i, (name, model) in enumerate(models.items()):
        folder = OUT / f"modelo_{i + 1}"
        folder.mkdir(exist_ok=True)
        joblib.dump(model, folder / "modelo.joblib")
        individual, individual_axes = plt.subplots(1, 2, figsize=(10, 4))
        for part, (x, y) in splits.items():
            pred, prob = predict(model, x)
            result = metrics(y, pred, prob)
            rows.append({"tipo": "Deep learning" if i == 3 else "Machine learning",
                         "modelo": name, "particion": part, "n": len(y),
                         **{k: v for k, v in result.items() if k != "matriz_confusion"}})
            pd.DataFrame({"fila_csv": x.index, "clase_real": y.to_numpy(),
                          "clase_predicha": pred, "probabilidad_clase_1": prob}).to_csv(folder / f"predicciones_{part}.csv", index=False)
            if part in ("entrenamiento", "prueba"):
                j = 0 if part == "entrenamiento" else 1
                fpr, tpr, _ = roc_curve(y, prob)
                roc_axes[j].plot(fpr, tpr, label=f"{name}: {result['roc_auc']:.4f}")
                individual_axes[j].plot(fpr, tpr, label=f"AUC={result['roc_auc']:.4f}")
            if part == "prueba":
                pd.DataFrame(result["matriz_confusion"], index=["Real 0", "Real 1"], columns=["Predicho 0", "Predicho 1"]).to_csv(folder / "matriz_confusion_prueba.csv")
                ConfusionMatrixDisplay(np.array(result["matriz_confusion"]), display_labels=[0, 1]).plot(ax=cm_axes.flat[i], colorbar=False, cmap="Blues")
                cm_axes.flat[i].set_title(name)
        for ax, title in zip(individual_axes, ["Entrenamiento", "Prueba"]):
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set(title=title, xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
            ax.legend()
        individual.tight_layout()
        individual.savefig(folder / "roc.png", dpi=220)
        plt.close(individual)
    for ax, title in zip(roc_axes, ["Entrenamiento", "Prueba"]):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(title=title, xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
        ax.legend(fontsize=8)
    roc_fig.tight_layout()
    roc_fig.savefig(OUT / "roc_comparacion.png", dpi=220)
    cm_fig.tight_layout()
    cm_fig.savefig(OUT / "matrices_confusion.png", dpi=220)
    plt.close(roc_fig)
    plt.close(cm_fig)
    history.to_csv(OUT / "historial_red.csv", index=False)
    graficar_aprendizaje(history, dnn_config["epoca_elegida"])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "metricas_completas.csv", index=False, encoding="utf-8-sig")
    for part in ("entrenamiento", "prueba"):
        comparison[comparison.particion == part].to_csv(OUT / f"comparativa_{part}.csv", index=False, encoding="utf-8-sig")
    joblib.dump(models[winner], OUT / "modelo_funcional.joblib")
    guardar_json(OUT / "modelo_funcional.json", {
        "modelo": winner, "elegido_por": "validación", "features": FEATURES,
        "umbral": "predict(): mayor probabilidad; empate binario favorece clase 0",
        "validacion": validation_results[winner], "version_sklearn": sklearn.__version__,
    })
    guardar_json(OUT / "ejecucion.json", {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "auditoria": quality,
        "dataset_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "segundos_ajuste": dict(zip(NAMES, [None, rf_time, lr_time, dnn_time])),
        "modelo_elegido": winner,
        "limitacion": "Prueba histórica consultada en prácticas anteriores. Se conserva para comparar. La selección usa validación.",
    })
    print(comparison[comparison.particion == "prueba"].to_string(index=False))
    print(f"Modelo funcional elegido con validación: {winner}")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
