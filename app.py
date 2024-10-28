import streamlit as st
from bienvenida import app as bienvenida_app
from eda import app as eda_app
from predicciones import app as predicciones_app

# Configuración de la página principal de Streamlit
st.set_page_config(page_title="Análisis y Predicción de Ventas", layout="wide")

# Título y menú de navegación en el sidebar
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una sección", ["Bienvenida", "EDA (Análisis de Datos)", "Predicción de Ventas"])

# Enrutamiento de las páginas
if page == "Bienvenida":
    bienvenida_app()
elif page == "EDA (Análisis de Datos)":
    eda_app()
elif page == "Predicción de Ventas":
    predicciones_app()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2024 - Proyecto Final Programación II")
