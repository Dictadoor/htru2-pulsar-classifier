from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

base = Path(__file__).resolve().parent
salida = base / "output" / "modelo_random_forest"
datos = pd.read_csv(salida / "predicciones_prueba.csv")

y_real = datos["clase_real"]
probabilidad_pulsar = datos["probabilidad_candidato_pulsar"]

fpr, tpr, _ = roc_curve(y_real, probabilidad_pulsar)
auc = roc_auc_score(y_real, probabilidad_pulsar)

plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC - conjunto de prueba")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(salida / "curva_roc.png", dpi=300, bbox_inches="tight")
plt.show()