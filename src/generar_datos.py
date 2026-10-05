"""Genera un dataset sintético de estudiantes para el SAT-DR."""
import numpy as np, pandas as pd

def generar(n=1500, semilla=42):
    rng = np.random.default_rng(semilla)
    asistencia = np.clip(rng.beta(6, 2, n) * 100, 20, 100)
    calif = np.clip(rng.normal(7.0, 1.6, n) + 0.02 * (asistencia - 75), 1, 10)
    entregas = np.clip(0.6 * asistencia + rng.normal(25, 15, n), 0, 100)
    # Relación no lineal + interacción + ruido
    z = (-0.07 * (asistencia - 70) - 1.1 * (calif - 6.5) - 0.03 * (entregas - 65)
         - 0.012 * (asistencia - 70) * (calif - 6.5) + rng.normal(0, 0.8, n) - 0.4)
    riesgo = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"asistencia": asistencia.round(1), "calificaciones": calif.round(2),
                         "entregas": entregas.round(1), "riesgo": riesgo})

if __name__ == "__main__":
    df = generar()
    df.to_csv("data/estudiantes.csv", index=False)
    print(df.shape, "| % en riesgo:", round(df.riesgo.mean() * 100, 1))
