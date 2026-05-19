import os
from collections import deque

import pandas as pd
import streamlit as st

from src.wrapper.client import MLOpsLifecycleClient


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=100)

client = MLOpsLifecycleClient(base_url=API_BASE_URL)

st.set_page_config(
    page_title="MLOps-Lifecycle",
    layout="wide",
)

st.title("MLOps-Lifecycle")
st.caption("Frontend para inferencia, reentrenamiento, versionado en caliente y monitorización MLOps")

left, right = st.columns([1, 1])

with left:
    st.subheader("Estado de la API")

    try:
        health = client.health()
        st.success("API disponible")
        st.json(health)
    except Exception as exc:
        st.error(f"No se pudo conectar con la API: {exc}")

    st.subheader("Cambiar versión del modelo")

    selected_version = st.selectbox(
        "Versión disponible",
        options=["v0.0.1", "v0.0.2"],
    )

    if st.button("Activar versión seleccionada"):
        try:
            response = client.change_version(version=selected_version)
            st.success("Versión cambiada correctamente")
            st.json(response)
        except Exception as exc:
            st.error(f"No se pudo cambiar la versión: {exc}")

    st.subheader("Cambiar modelo desde URL")

    model_url = st.text_input(
        "URL directa a model.safetensors",
        placeholder="https://example.com/model.safetensors",
    )

    url_version = st.text_input(
        "Versión asociada a la URL",
        value="v0.0.3",
    )

    if st.button("Descargar y activar modelo desde URL"):
        try:
            response = client.change_version(
                version=url_version,
                url=model_url,
            )
            st.success("Modelo descargado y activado")
            st.json(response)
        except Exception as exc:
            st.error(f"No se pudo descargar el modelo: {exc}")

with right:
    st.subheader("Inferencia manual")

    text = st.text_area(
        "Texto a analizar",
        value="I really love this product. It works perfectly.",
        height=140,
    )

    expected_label = st.selectbox(
        "Etiqueta esperada opcional",
        options=["", "POSITIVE", "NEGATIVE"],
    )

    if st.button("Predecir sentimiento"):
        try:
            prediction = client.predict(
                text=text,
                expected_label=expected_label or None,
            )

            st.session_state.history.append(
                {
                    "text": text,
                    "label": prediction["label"],
                    "score": prediction["score"],
                    "expected_label": prediction["expected_label"],
                    "is_correct": prediction["is_correct"],
                    "model_version": prediction["model_version"],
                }
            )

            st.success("Predicción recibida")
            st.json(prediction)

        except Exception as exc:
            st.error(f"Error durante la predicción: {exc}")

st.subheader("Reentrenamiento / nueva versión")

with st.expander("Crear nueva versión del modelo"):
    target_version = st.text_input(
        "Nueva versión",
        value="v0.0.2",
    )

    st.write("Muestras sintéticas para registrar una nueva versión del modelo")

    samples = [
        {"text": "This model is excellent and reliable.", "label": "POSITIVE"},
        {"text": "The prediction is useful and accurate.", "label": "POSITIVE"},
        {"text": "This result is bad and disappointing.", "label": "NEGATIVE"},
        {"text": "The model response is poor and unstable.", "label": "NEGATIVE"},
    ]

    st.dataframe(pd.DataFrame(samples), use_container_width=True)

    if st.button("Reentrenar y activar nueva versión"):
        try:
            response = client.retrain(
                target_version=target_version,
                samples=samples,
            )
            st.success("Nueva versión creada y activada")
            st.json(response)
        except Exception as exc:
            st.error(f"No se pudo reentrenar el modelo: {exc}")

st.subheader("Histórico de predicciones")

if st.session_state.history:
    df = pd.DataFrame(list(st.session_state.history))
    st.dataframe(df, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Predicciones por etiqueta")
        label_chart = df.groupby("label", as_index=False).size()
        st.bar_chart(label_chart.set_index("label"))

    with col_b:
        st.write("Score medio por versión")
        score_chart = df.groupby("model_version", as_index=False)["score"].mean()
        st.bar_chart(score_chart.set_index("model_version"))
else:
    st.info("Todavía no hay predicciones manuales registradas.")