from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "output" / "modelo_random_forest" / "modelo_random_forest.joblib"
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
LABELS = {0: "No pulsar", 1: "Candidato a pulsar"}


st.set_page_config(page_title="HTRU2 · Clasificador de púlsares", page_icon="🪐", layout="wide")

css = """
<style>
:root {
  color-scheme:dark;
  --page:#0c0b10; --surface:#17151d; --hero:#14121a; --ink:#f2eef7;
  --muted:#b4adbf; --line:#303947; --accent:#4da3ff; --accent-hover:#3387e8;
  --accent-ink:#ffffff; --soft:#211a29; --input:#100f15;
  --beam:rgba(77,163,255,.5); --glow:rgba(77,163,255,.42);
  --st-background-color:#0c0b10 !important;
  --st-secondary-background-color:#17151d !important;
  --st-text-color:#f2eef7 !important;
  --st-primary-color:#4da3ff !important;
  --st-border-color:#303947 !important;
}
[data-testid="stAppViewContainer"],[data-testid="stHeader"] {background-color:var(--page) !important;color:var(--ink) !important;}
[data-testid="stHeader"] {display:none !important;}
.block-container {max-width:1200px;padding-top:1.1rem;padding-bottom:3rem;}
.universe-hero {position:relative;display:flex;align-items:center;justify-content:space-between;gap:1.5rem;min-height:255px;margin:.45rem 0 1.6rem;padding:2.2rem 2.5rem;overflow:hidden;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;background-color:var(--hero);}
.hero-copy {max-width:700px;position:relative;z-index:2;}
.hero-kicker {margin:0 0 .7rem;color:var(--accent);font-size:.72rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;}
.hero-title {margin:0;color:var(--ink);font-size:clamp(2rem,4.4vw,3.3rem);line-height:1.04;font-weight:740;letter-spacing:-.045em;}
.hero-description {max-width:640px;margin:.95rem 0 1.2rem;color:var(--muted);font-size:1rem;line-height:1.6;}
.hero-tags {display:flex;flex-wrap:wrap;gap:.5rem;}
.hero-tag {padding:.38rem .65rem;border:1px solid var(--line);border-radius:5px;color:var(--muted);background:var(--surface);font-size:.76rem;}
.pulsar-scene {position:relative;flex:0 0 230px;width:230px;height:205px;isolation:isolate;}
.pulsar-star {position:absolute;width:3px;height:3px;border-radius:50%;background:#55aaff;box-shadow:0 0 7px rgba(85,170,255,.72);}
.pulsar-star.s1 {left:15%;top:22%;}.pulsar-star.s2 {right:12%;top:27%;width:2px;height:2px;}.pulsar-star.s3 {left:22%;bottom:15%;width:2px;height:2px;}.pulsar-star.s4 {right:18%;bottom:19%;}
.pulsar-orbit {position:absolute;inset:21px 7px;border:1px solid var(--line);border-radius:50%;transform:rotate(-24deg) scaleY(.48);}
.pulsar-orbit.orbit2 {inset:35px 22px;border-color:#55aaff;opacity:.55;transform:rotate(32deg) scaleY(.58);}
.pulsar-beams {position:absolute;left:50%;top:50%;width:0;height:0;animation:beam-sweep 9s linear infinite;}
.pulsar-beam {position:absolute;left:-14px;width:28px;height:94px;clip-path:polygon(50% 0,100% 100%,0 100%);background:var(--beam);}
.pulsar-beam.up {bottom:-3px;}.pulsar-beam.down {top:-3px;transform:rotate(180deg);}
.pulsar-core {position:absolute;left:50%;top:50%;z-index:2;width:28px;height:28px;transform:translate(-50%,-50%);border:2px solid #eaf5ff;border-radius:50%;background:#55aaff;box-shadow:0 0 0 5px rgba(85,170,255,.32),0 0 24px rgba(85,170,255,.72);animation:core-pulse 2.1s ease-in-out infinite;}
.pulsar-ring {position:absolute;left:50%;top:50%;width:44px;height:44px;transform:translate(-50%,-50%);border:1px solid #55aaff;border-radius:50%;animation:ring-pulse 2.1s ease-out infinite;}
@keyframes beam-sweep {to {transform:rotate(360deg);}}
@keyframes core-pulse {0%,100% {scale:.92;filter:brightness(.95);}50% {scale:1.13;filter:brightness(1.3);}}
@keyframes ring-pulse {0% {scale:.72;opacity:.78;}100% {scale:2.55;opacity:0;}}
.section-kicker {margin:1.5rem 0 .2rem;color:var(--accent);font-size:.7rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;}
.section-heading {margin:0 0 .35rem;color:var(--ink);font-size:1.55rem;font-weight:700;letter-spacing:-.03em;}
.section-copy {margin:0 0 1rem;color:var(--muted);font-size:.94rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line) !important;border-radius:10px !important;background-color:var(--surface) !important;}
[data-testid="stFileUploader"] section {border:1px dashed var(--line) !important;border-radius:7px;background-color:var(--input) !important;}
[data-testid="stFileUploader"] section {height:84px !important;min-height:84px !important;padding:10px 14px !important;box-sizing:border-box !important;display:flex !important;align-items:center !important;}
[data-testid="stFileUploader"] section:hover {border-color:var(--accent) !important;}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] small {color:var(--ink) !important;}
[data-testid="stFileUploader"] section button {border:1px solid var(--accent) !important;border-radius:6px !important;background-color:var(--accent) !important;color:var(--accent-ink) !important;font-weight:700 !important;}
[data-testid="stFileUploader"] section button span,[data-testid="stFileUploader"] section button svg {color:var(--accent-ink) !important;fill:var(--accent-ink) !important;}
[data-testid="stFileUploader"] section button:hover {background-color:var(--accent-hover) !important;}
.stButton button,.stDownloadButton button {border:1px solid var(--line) !important;border-radius:6px !important;background-color:var(--surface) !important;color:var(--ink) !important;}
.upload-field-label {height:26px;display:flex;align-items:flex-start;margin:0 0 8px;color:var(--ink);font-size:.875rem;line-height:1.25;}
.st-key-template-download button {height:48px !important;min-height:48px !important;margin-top:20px !important;}
.stButton button[kind="primary"] {border-color:var(--accent) !important;background-color:var(--accent) !important;color:var(--accent-ink) !important;font-weight:700;}
.stButton button[kind="primary"]:hover {background-color:var(--accent-hover) !important;}
[data-testid="stMetric"] {padding:.9rem 1rem;border:1px solid var(--line);border-radius:8px;background-color:var(--surface) !important;}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"] {color:var(--ink) !important;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:7px;overflow:hidden;}
.space-note {padding:.8rem 1rem;border-left:3px solid var(--accent);border-radius:0;background-color:var(--soft);color:var(--muted);font-size:.88rem;line-height:1.55;}
.space-footer {margin-top:2rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.78rem;text-align:center;}
.stMarkdown p,.stCaption,.stText,[data-testid="stWidgetLabel"] p {color:var(--ink);}
.stCaption {color:var(--muted) !important;}
h1 a,h2 a,h3 a,[data-testid="stHeaderActionElements"] {display:none !important;}
@media (max-width:760px) {.block-container {padding-top:.6rem;}.universe-hero {min-height:210px;padding:1.35rem;border-radius:9px;}.pulsar-scene {flex-basis:120px;width:120px;height:150px;transform:scale(.72);transform-origin:center right;margin-left:-1.4rem;margin-right:-1rem;}.hero-description {font-size:.92rem;}}
@media (prefers-reduced-motion:reduce) {.pulsar-beams,.pulsar-core,.pulsar-ring {animation:none;}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown(
    """
    <section class="universe-hero">
      <div class="hero-copy">
        <h1 class="hero-title">Clasificador de candidatos a púlsar</h1>
        <p class="hero-description">Clasifica registros HTRU2 con Random Forest a partir de sus ocho características estadísticas.</p>
        <div class="hero-tags"><span class="hero-tag">8 variables de entrada</span><span class="hero-tag">Archivo CSV</span><span class="hero-tag">Clases 0 y 1</span></div>
      </div>
      <div class="pulsar-scene" role="img" aria-label="Animación de un púlsar con haces de luz giratorios">
        <span class="pulsar-star s1"></span><span class="pulsar-star s2"></span><span class="pulsar-star s3"></span><span class="pulsar-star s4"></span>
        <div class="pulsar-orbit"></div><div class="pulsar-orbit orbit2"></div>
        <div class="pulsar-beams"><div class="pulsar-beam up"></div><div class="pulsar-beam down"></div></div>
        <div class="pulsar-ring"></div><div class="pulsar-core"></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("No se encuentra el modelo entrenado.")
    return joblib.load(MODEL_FILE)


st.markdown('<p class="section-kicker">Exploración</p><h2 class="section-heading">Clasifica un conjunto de datos</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Carga un CSV con las ocho características requeridas para obtener una predicción por registro.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    template = pd.DataFrame(columns=FEATURES)
    st.markdown("**Prepara tus datos**")
    st.caption("Usa la plantilla para conservar los nombres de columnas que espera el modelo.")
    left, right = st.columns([1, 2], vertical_alignment="top")
    with left:
        st.markdown('<div class="upload-field-label">Plantilla CSV</div>', unsafe_allow_html=True)
        with st.container(key="template-download"):
            st.download_button(
                "Descargar plantilla CSV",
                data=template.to_csv(index=False).encode("utf-8"),
                file_name="plantilla_candidatos_htru2.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with right:
        st.markdown('<div class="upload-field-label">Selecciona un archivo CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Selecciona un archivo CSV",
            type=["csv"],
            label_visibility="collapsed",
            help="Debe incluir las ocho columnas de características. La columna class es opcional y no se usa para predecir.",
        )

if uploaded is not None:
    try:
        incoming = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()

    missing = [column for column in FEATURES if column not in incoming.columns]
    if missing:
        st.error("Faltan estas columnas requeridas: " + ", ".join(missing))
        st.stop()
    if incoming.empty:
        st.error("El archivo no contiene registros.")
        st.stop()

    predictors = incoming[FEATURES].apply(pd.to_numeric, errors="coerce")
    if predictors.isna().any().any():
        st.error("Las ocho características deben contener números, sin celdas vacías.")
        st.stop()
    if not np.isfinite(predictors.to_numpy(dtype=float)).all():
        st.error("El archivo contiene valores no finitos.")
        st.stop()

    if TARGET in incoming.columns:
        st.info("La columna class está presente, pero se excluye de la predicción.")

    if st.button("Clasificar registros", type="primary", use_container_width=False):
        try:
            model = load_model()
            predicted = model.predict(predictors)
            class_one_index = list(model.classes_).index(1)
            probability = model.predict_proba(predictors)[:, class_one_index]
        except Exception as exc:
            st.error(f"No se pudo ejecutar el modelo: {exc}")
            st.stop()

        results = incoming.drop(columns=[TARGET], errors="ignore").copy()
        results["clase_predicha"] = predicted.astype(int)
        results["resultado"] = [LABELS[int(value)] for value in predicted]
        results["probabilidad_candidato_pulsar"] = probability

        pulsar_count = int((predicted == 1).sum())
        other_count = len(predicted) - pulsar_count
        st.markdown('<p class="section-kicker">Resultado del análisis</p><h2 class="section-heading">Clasificación del conjunto</h2>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Registros clasificados", f"{len(results):,}")
        b.metric("Candidatos · clase 1", f"{pulsar_count:,}")
        c.metric("No candidatos · clase 0", f"{other_count:,}")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "clase_predicha": st.column_config.NumberColumn("Clase (0 / 1)", format="%d"),
                "resultado": st.column_config.TextColumn("Interpretación"),
                "probabilidad_candidato_pulsar": st.column_config.NumberColumn(
                    "Probabilidad clase 1", format="percent"
                ),
            },
        )
        st.markdown(
            '<p class="space-note">La clase 1 señala un candidato según las etiquetas HTRU2. Es una predicción del modelo, no la confirmación de un descubrimiento astronómico.</p>',
            unsafe_allow_html=True,
        )

        csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar resultados CSV",
            data=csv_bytes,
            file_name="predicciones_htru2.csv",
            mime="text/csv",
        )

