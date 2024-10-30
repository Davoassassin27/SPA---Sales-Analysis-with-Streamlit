# predicciones.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import io

def cargar_datos(uploaded_file):
    """Función para cargar y verificar el archivo CSV."""
    try:
        data = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
        columnas_necesarias = {"Fecha", "Producto", "Cantidad"}
        if not columnas_necesarias.issubset(data.columns):
            st.error("El archivo CSV debe contener las columnas: Fecha, Producto, y Cantidad.")
            return None
        return data
    except ValueError:
        st.error("Error en el formato del archivo. Verifica que la columna 'Fecha' esté en el archivo y tenga el formato correcto.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo: {str(e)}")
        return None

def optimizar_arima(datos_producto_agrupado, max_p=3, max_d=2, max_q=3):
    """Encuentra el mejor modelo ARIMA basado en el menor valor AIC."""
    mejor_aic = np.inf
    mejor_order = None
    mejor_modelo = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    modelo = ARIMA(datos_producto_agrupado['Cantidad'], order=(p, d, q)).fit()
                    if modelo.aic < mejor_aic:
                        mejor_aic = modelo.aic
                        mejor_order = (p, d, q)
                        mejor_modelo = modelo
                except:
                    continue
    return mejor_modelo, mejor_order

def realizar_prediccion(datos_producto, modelo_prediccion, num_meses):
    """Realiza predicción en base al modelo seleccionado, optimizando parámetros si es necesario."""

    datos_producto_agrupado = datos_producto.resample('ME').sum()
    datos_producto_agrupado['MesNumerico'] = (datos_producto_agrupado.index.year - datos_producto_agrupado.index.year.min()) * 12 + datos_producto_agrupado.index.month

    if modelo_prediccion == "Regresión Lineal":
        X = datos_producto_agrupado[['MesNumerico']]
        y = datos_producto_agrupado['Cantidad']
        modelo = LinearRegression().fit(X, y)
        meses_futuros = np.array([X['MesNumerico'].max() + i for i in range(1, num_meses + 1)]).reshape(-1, 1)
        predicciones = modelo.predict(meses_futuros)
        fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

    elif modelo_prediccion == "ARIMA":
        # Optimizar el modelo ARIMA
        modelo_arima, mejor_order = optimizar_arima(datos_producto_agrupado)
        st.write(f"Mejor parámetro ARIMA encontrado: {mejor_order}")
        predicciones = modelo_arima.forecast(steps=num_meses)
        fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

    elif modelo_prediccion == "Holt-Winters":
        # Modelo Holt-Winters con ajuste estacionalidad
        modelo_hw = ExponentialSmoothing(datos_producto_agrupado['Cantidad'], trend='add', seasonal='add', seasonal_periods=12).fit()
        predicciones = modelo_hw.forecast(steps=num_meses)
        fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

    return fechas_futuras, predicciones, datos_producto_agrupado

def calcular_mae(datos_reales, predicciones):
    """Calcula el Error Absoluto Medio (MAE) entre datos reales y predicciones."""
    mae = np.mean(np.abs(datos_reales - predicciones))
    return mae

def app():
    st.title("Predicción de Ventas")

    # Cargar archivo CSV
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type="csv")
    if not uploaded_file:
        st.warning("Por favor, sube un archivo CSV para comenzar.")
        return

    # Cargar datos
    data = cargar_datos(uploaded_file)
    if data is None:
        return  # Detener la ejecución si el archivo es inválido

    # Seleccionar producto y modelo
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto para analizar", data['Producto'].unique())
    modelo_prediccion = st.sidebar.selectbox("Selecciona el modelo de predicción", ["Regresión Lineal", "ARIMA", "Holt-Winters"])
    num_meses = st.sidebar.slider("Selecciona el número de meses para predecir", 1, 12, 6)

    # Filtrar datos por producto seleccionado
    datos_producto = data[data['Producto'] == producto_seleccionado].set_index('Fecha')

    # Generar predicción y mostrar resultados
    fechas_futuras, predicciones, datos_producto_agrupado = realizar_prediccion(datos_producto, modelo_prediccion, num_meses)
    st.subheader(f"Predicción de Ventas para {producto_seleccionado} usando {modelo_prediccion}")

    # Gráfico de predicción con color rojo para las predicciones
    fig_pred = px.line(title=f"Predicción para {producto_seleccionado} ({modelo_prediccion})")
    fig_pred.add_scatter(x=datos_producto_agrupado.index, y=datos_producto_agrupado['Cantidad'], mode='lines', name='Datos Históricos')
    fig_pred.add_scatter(x=fechas_futuras, y=predicciones, mode='lines', name='Predicción', line=dict(color='red'))
    st.plotly_chart(fig_pred)

    # Backtesting
    st.subheader("Evaluación de Precisión del Modelo (Backtesting)")
    test_size = st.slider("Selecciona el número de meses para evaluar el modelo", 1, min(len(datos_producto_agrupado), 12), 3)
    fechas_backtest, pred_backtest, _ = realizar_prediccion(datos_producto_agrupado.iloc[:-test_size], modelo_prediccion, test_size)
    datos_reales = datos_producto_agrupado['Cantidad'].iloc[-test_size:]

    # Cálculo de error absoluto medio (MAE)
    mae = calcular_mae(datos_reales.values, pred_backtest[:test_size])
    st.write(f"Error Absoluto Medio (MAE): {mae:.2f}")

    # Gráfico de backtesting
    fig_backtest = px.line(title="Backtesting del Modelo")
    fig_backtest.add_scatter(x=datos_producto_agrupado.index[-test_size:], y=datos_reales, mode='lines', name='Datos Reales')
    fig_backtest.add_scatter(x=fechas_backtest, y=pred_backtest, mode='lines', name='Predicción Backtest', line=dict(color='red'))
    st.plotly_chart(fig_backtest)

    # Exportar predicciones a CSV
    st.sidebar.markdown("### Descargar Predicciones")
    if fechas_futuras is not None and predicciones is not None:
        predicciones_df = pd.DataFrame({'Fecha': fechas_futuras, 'Predicción': predicciones})
        buffer = io.BytesIO()
        predicciones_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.sidebar.download_button(
            label="Descargar Predicciones en CSV",
            data=buffer,
            file_name="predicciones_ventas.csv",
            mime="text/csv"
        )
