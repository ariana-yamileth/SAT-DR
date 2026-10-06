"""SAT-DR: Sistema de Alerta Temprana de Deserción y Reprobación (app web)."""
import json, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="SAT-DR", page_icon="🎓", layout="centered")

@st.cache_resource
def cargar():
    return joblib.load("models/modelo.joblib"), json.load(open("models/metricas.json"))

paquete, metricas = cargar()
modelo, umbral, FEATURES = paquete["modelo"], paquete["umbral"], paquete["features"]

def clasificar(p):
    return "🔴 Riesgo alto" if p >= umbral else ("🟡 Vigilancia" if p >= umbral * 0.6 else "🟢 Sin riesgo")

st.title("🎓 SAT-DR")
st.caption("Sistema de Alerta Temprana de Deserción y Reprobación · Red neuronal densa (MLP)")
t1, t2, t3 = st.tabs(["Estudiante individual", "Carga por lote (CSV)", "Rendimiento del modelo"])

with t1:
    a = st.slider("Asistencia acumulada (%)", 0, 100, 80)
    c = st.slider("Promedio de calificaciones parciales", 1.0, 10.0, 7.0, 0.1)
    e = st.slider("Tareas entregadas a tiempo (%)", 0, 100, 75)
    if st.button("Evaluar riesgo", type="primary"):
        p = float(modelo.predict_proba(pd.DataFrame([[a, c, e]], columns=FEATURES))[0, 1])
        st.metric("Probabilidad de riesgo", f"{p:.2f}")
        st.progress(min(p, 1.0))
        st.subheader(clasificar(p))
        st.info("Se recomienda canalizar al estudiante a tutoría." if p >= umbral else "Sin acción inmediata; continuar monitoreo.")

with t2:
    st.write("Sube un CSV con las columnas: `asistencia`, `calificaciones`, `entregas`.")
    st.download_button("Descargar CSV de ejemplo", open("data/estudiantes.csv").read(), "ejemplo.csv")
    f = st.file_uploader("Archivo CSV", type="csv")
    if f:
        try:
            df = pd.read_csv(f)
            df["probabilidad"] = modelo.predict_proba(df[FEATURES])[:, 1].round(3)
            df["clasificacion"] = df["probabilidad"].apply(clasificar)
            st.dataframe(df.sort_values("probabilidad", ascending=False), use_container_width=True)
            st.write(f"**{(df.probabilidad >= umbral).sum()}** de {len(df)} estudiantes en riesgo alto.")
            st.download_button("Descargar resultados", df.to_csv(index=False), "resultados.csv")
        except KeyError:
            st.error("El archivo debe tener las columnas: asistencia, calificaciones, entregas.")

with t3:
    m = metricas["matriz"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall (meta ≥ 85%)", f"{metricas['recall']*100:.1f}%")
    c2.metric("Precisión", f"{metricas['precision']*100:.1f}%")
    c3.metric("Exactitud", f"{metricas['exactitud']*100:.1f}%")
    st.write("**Matriz de confusión** (conjunto de prueba)")
    st.table(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
             index=["Real: sin riesgo", "Real: en riesgo"], columns=["Pred: sin riesgo", "Pred: en riesgo"]))
    st.caption(f"Umbral de decisión: {umbral}. Datos sintéticos generados para fines académicos.")