import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Función para cargar animaciones Lottie desde una URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():
    lottie_animation = load_lottieurl("https://lottie.host/1b48819b-72d7-4e3f-97e2-ddd963ee94ba/9YrG27w5uW.json")  
    
    # Título de bienvenida
    st.markdown("# Bienvenido a la Aplicación de Análisis de Ventas")

    if lottie_animation:
        st_lottie(lottie_animation, height=200, key="bienvenida_animation")


    st.markdown("""
    ### 🎯 Objetivo de la Aplicación
    Esta aplicación permite analizar y predecir ventas utilizando datos históricos cargados en formato CSV. Con ella podrás visualizar tendencias, comparar modelos de predicción y descargar resultados de pronóstico.
    
    ### 🛠️ Herramientas y Modelos de Predicción
    - **Regresión Lineal**: 📈 Modelo sencillo para detectar tendencias lineales en los datos de ventas.
    - **ARIMA**: 🔄 Modelo avanzado que permite capturar tanto la tendencia como la estacionalidad de las ventas.
    - **Holt-Winters**: 🕰️ Ideal para tendencias y estacionalidad complejas, especialmente en series temporales de ventas.

    ### 📥 Instrucciones de Uso
    1. **Carga de Datos**: Sube tu archivo CSV de ventas a través del panel lateral. ("ventas.csv")
    2. **Selección de Producto**: Escoge el producto que deseas analizar.
    3. **Elección del Modelo de Predicción**: Selecciona el modelo entre Regresión Lineal, ARIMA y Holt-Winters.
    4. **Predicción Global**: Puedes realizar predicciones para todos los productos activando la opción en el panel lateral.

    ### 👤 Autor
    Desarrollado por: David Soler
    """)
