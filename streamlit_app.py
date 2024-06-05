import streamlit as st
from transformers import pipeline

# Configurar el modelo de análisis de sentimientos
nlp = pipeline("sentiment-analysis")

st.title("Aplicación de Análisis de Sentimientos")
st.write("Ingrese un texto para analizar su sentimiento.")

# Entrada de texto del usuario
user_input = st.text_area("Texto:")

if user_input:
    # Analizar el sentimiento
    result = nlp(user_input)
    sentiment = result[0]

    # Mostrar resultado
    st.write(f"**Sentimiento:** {sentiment['label']}")
    st.write(f"**Confianza:** {sentiment['score']:.2f}")
