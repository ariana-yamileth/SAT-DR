"""Entrena el MLP, ajusta el umbral para Recall >= 85% y guarda modelo y métricas."""
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

FEATURES = ["asistencia", "calificaciones", "entregas"]
df = pd.read_csv("data/estudiantes.csv")
X_tmp, X_te, y_tmp, y_te = train_test_split(df[FEATURES], df["riesgo"], test_size=0.2, stratify=df["riesgo"], random_state=1)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=1)

modelo = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                       max_iter=2000, alpha=1e-3, learning_rate_init=0.003, random_state=1))
modelo.fit(X_tr, y_tr)

# Umbral más alto que alcance Recall >= 0.88 en validación (margen sobre el 85% pedido)
p_val = modelo.predict_proba(X_val)[:, 1]
umbral = 0.5
for t in np.arange(0.60, 0.05, -0.01):
    if recall_score(y_val, p_val >= t) >= 0.88:
        umbral = round(float(t), 2); break

pred = modelo.predict_proba(X_te)[:, 1] >= umbral
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
metricas = {"umbral": umbral, "recall": round(recall_score(y_te, pred), 3),
            "precision": round(precision_score(y_te, pred), 3), "exactitud": round(accuracy_score(y_te, pred), 3),
            "matriz": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
joblib.dump({"modelo": modelo, "umbral": umbral, "features": FEATURES}, "models/modelo.joblib")
json.dump(metricas, open("models/metricas.json", "w"), indent=2)
print(metricas)
