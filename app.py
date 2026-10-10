"""SAT-DR: Sistema de Alerta Temprana de Deserción y Reprobación (app web)."""
import io, json, joblib, altair as alt, pandas as pd, streamlit as st

st.set_page_config(page_title="SAT-DR", layout="centered")

st.markdown(
    """
    <style>
    :root {
        --primario: 59, 143, 219;      /* azul de los controles */
        --acento: 40, 170, 160;        /* verde azulado del fondo */
    }
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 15% 0%, rgba(var(--primario), 0.28), transparent 60%),
            radial-gradient(ellipse 70% 45% at 90% 10%, rgba(var(--acento), 0.18), transparent 60%),
            linear-gradient(180deg, #0b1220 0%, #0e1627 55%, #0a101c 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* Textos secundarios con tono azulado en lugar de gris neutro */
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] { color: #9FB1CC; }

    /* Líneas y bordes (tabs, tablas) con tono azul oscuro */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(159, 177, 204, 0.25); }
    [data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td {
        border-color: rgba(159, 177, 204, 0.22);
    }

    /* Reemplaza el rojo por defecto de Streamlit por el azul de la paleta */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #3B8FDB; border-color: #3B8FDB; color: #FFFFFF; font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #5BA3E6; border-color: #5BA3E6; color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #5BA3E6; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #3B8FDB; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def cargar():
    return joblib.load("models/modelo.joblib"), json.load(open("models/metricas.json"))

paquete, metricas = cargar()
modelo, umbral, FEATURES = paquete["modelo"], paquete["umbral"], paquete["features"]

def clasificar(p):
    return "🔴 Riesgo alto" if p >= umbral else ("🟡 Vigilancia" if p >= umbral * 0.6 else "🟢 Sin riesgo")

NIVELES = ["🔴 Riesgo alto", "🟡 Vigilancia", "🟢 Sin riesgo"]
COLORES = ["#E5484D", "#F5B83D", "#30A46C"]

def grafica_distribucion(df):
    """Barras: cuántos estudiantes hay en cada nivel de riesgo."""
    conteo = (df["clasificacion"].value_counts()
              .reindex(NIVELES, fill_value=0).rename_axis("nivel").reset_index(name="estudiantes"))
    base = alt.Chart(conteo).encode(
        x=alt.X("nivel:N", sort=NIVELES, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("estudiantes:Q", title="Estudiantes"),
    )
    barras = base.mark_bar().encode(
        color=alt.Color("nivel:N", scale=alt.Scale(domain=NIVELES, range=COLORES), legend=None),
        tooltip=["nivel", "estudiantes"],
    )
    etiquetas = base.mark_text(dy=-8, fontWeight="bold").encode(text="estudiantes:Q")
    return (barras + etiquetas).properties(height=280)

TEXTO_ALTO = "Se recomienda canalizar al estudiante a tutoría."
TEXTO_VIGILANCIA = "Cerca del umbral de riesgo. Dar seguimiento cercano y volver a evaluar antes del siguiente parcial."
TEXTO_SIN_RIESGO = "Sin acción inmediata; continuar monitoreo."

def recomendar(p):
    return TEXTO_ALTO if p >= umbral else (TEXTO_VIGILANCIA if p >= umbral * 0.6 else TEXTO_SIN_RIESGO)

st.title("SAT-DR")
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
        if p >= umbral:
            st.error(TEXTO_ALTO)
        elif p >= umbral * 0.6:
            st.warning(TEXTO_VIGILANCIA)
        else:
            st.success(TEXTO_SIN_RIESGO)

with t2:
    st.write("Sube un archivo CSV o Excel (.xlsx) con las columnas: `asistencia`, `calificaciones`, `entregas`.")
    st.download_button("Descargar CSV de ejemplo", open("data/estudiantes.csv").read(), "ejemplo.csv")
    f = st.file_uploader("Archivo CSV o Excel", type=["csv", "xlsx"])
    if f:
        try:
            es_excel = f.name.lower().endswith(".xlsx")
            df = pd.read_excel(f) if es_excel else pd.read_csv(f)
            df["probabilidad"] = modelo.predict_proba(df[FEATURES])[:, 1].round(3)
            df["clasificacion"] = df["probabilidad"].apply(clasificar)
            df["recomendacion"] = df["probabilidad"].apply(recomendar)
            st.write(f"**{(df.probabilidad >= umbral).sum()}** de {len(df)} estudiantes en riesgo alto.")
            st.write("**Distribución por nivel de riesgo**")
            st.altair_chart(grafica_distribucion(df), use_container_width=True)
            st.dataframe(df.sort_values("probabilidad", ascending=False), use_container_width=True)

            export = df.copy()
            export["clasificacion"] = export["clasificacion"].str.split(" ", n=1).str[1]
            if es_excel:
                buffer = io.BytesIO()
                export.to_excel(buffer, index=False, engine="openpyxl")
                st.download_button(
                    "Descargar resultados (Excel)",
                    buffer.getvalue(),
                    "resultados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.download_button(
                    "Descargar resultados (CSV)",
                    export.to_csv(index=False).encode("utf-8-sig"),
                    "resultados.csv",
                    mime="text/csv",
                )
        except KeyError:
            st.error("El archivo debe tener las columnas: asistencia, calificaciones, entregas.")
        except ImportError:
            st.error("Falta la librería `openpyxl` para leer archivos Excel. Agrégala a requirements.txt (o ejecuta `pip install openpyxl`).")
        except Exception as err:
            st.error("No se pudo leer el archivo. Verifica que sea un CSV o un Excel (.xlsx) válido.")
            st.caption(f"Detalle técnico: {type(err).__name__}: {err}")

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