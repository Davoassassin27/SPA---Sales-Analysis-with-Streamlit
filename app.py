import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import io

# Sidebar: Configuración global
st.sidebar.title("Configuraciones")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto para analizar", data['Producto'].unique())
    modelo_prediccion = st.sidebar.selectbox("Selecciona el modelo de predicción", ["Regresión Lineal", "ARIMA", "Holt-Winters"])
    num_meses = st.sidebar.slider("Selecciona el número de meses para predecir", 1, 12, 6)
else:
    st.sidebar.warning("Por favor, sube un archivo CSV para comenzar.")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["Bienvenida & Documentación", "Análisis Exploratorio de Datos (EDA)", "Predicción de Ventas"])

# Tab 1: Bienvenida & Documentación
with tab1:
    st.title("Bienvenido a la Aplicación de Análisis de Ventas")
    
    st.markdown("""
    Esta aplicación te permitirá cargar un archivo CSV con datos de ventas, explorar los datos y generar predicciones utilizando diferentes modelos de series temporales.
    
    ### Instrucciones de Uso
    1. **Carga de datos**: Sube un archivo CSV con una columna de fechas.
    2. **Selección de producto**: Elige un producto para analizar desde el panel lateral.
    3. **Elección del modelo**: Selecciona el modelo de predicción desde el panel lateral.
    4. **Predicciones para todos los productos**: Puedes realizar predicciones para todos los productos a la vez marcando la opción en el sidebar.

    ### Modelos de Predicción
    - **Regresión Lineal**: Modelo simple que asume una relación lineal entre el tiempo y las ventas.
    - **ARIMA**: Modelo de series temporales que captura la tendencia y la estacionalidad.
    - **Holt-Winters**: Modelo que captura tanto la tendencia como la estacionalidad.

    ### Descarga de Predicciones
    Puedes descargar las predicciones generadas en formato CSV desde el panel lateral.
    """)

    # Espacio para el logo de la universidad
    st.image("src/logo-ucasal2.png", width=150)

# Tab 2: Análisis Exploratorio de Datos (EDA)
with tab2:
    st.title("Análisis Exploratorio de Datos")
    if uploaded_file:
        datos_producto = data[data['Producto'] == producto_seleccionado]

        with st.expander(f"Estadísticas descriptivas para {producto_seleccionado}"):
            st.write(datos_producto.describe())

        with st.expander(f"Distribución de las ventas del producto: {producto_seleccionado}"):
            fig_hist = px.histogram(datos_producto, x='Cantidad', nbins=20, title=f'Distribución de ventas para {producto_seleccionado}')
            st.plotly_chart(fig_hist)

        with st.expander(f"Evolución de ventas mensuales para {producto_seleccionado}"):
            datos_producto_agrupado = datos_producto.resample('ME', on='Fecha').sum()
            fig_line = px.line(datos_producto_agrupado, x=datos_producto_agrupado.index, y='Cantidad', title=f'Ventas mensuales de {producto_seleccionado}')
            st.plotly_chart(fig_line)
    else:
        st.warning("Por favor, sube un archivo CSV para realizar el análisis.")

# Tab 3: Predicción de Ventas
with tab3:
    st.title("Predicción de Ventas")
    if uploaded_file:
        datos_producto = data[data['Producto'] == producto_seleccionado]
        datos_producto_agrupado = datos_producto.resample('ME', on='Fecha').sum()

        # Convertir fechas a formato numérico
        datos_producto_agrupado['MesNumerico'] = (datos_producto_agrupado.index.year - datos_producto_agrupado.index.year.min()) * 12 + datos_producto_agrupado.index.month
        
        # Modelos de predicción
        if modelo_prediccion == "Regresión Lineal":
            X = datos_producto_agrupado[['MesNumerico']]
            y = datos_producto_agrupado['Cantidad']
            modelo = LinearRegression().fit(X, y)
            meses_futuros = np.array([X['MesNumerico'].max() + i for i in range(1, num_meses + 1)]).reshape(-1, 1)
            predicciones = modelo.predict(meses_futuros)
            fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

        elif modelo_prediccion == "ARIMA":
            modelo_arima = ARIMA(datos_producto_agrupado['Cantidad'], order=(5, 1, 0)).fit()
            predicciones = modelo_arima.forecast(steps=num_meses)
            fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

        elif modelo_prediccion == "Holt-Winters":
            modelo_hw = ExponentialSmoothing(datos_producto_agrupado['Cantidad'], trend='add', seasonal='add', seasonal_periods=12).fit()
            predicciones = modelo_hw.forecast(steps=num_meses)
            fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')

        # Graficar datos históricos y predicciones
        fig_pred = px.line(x=fechas_futuras, y=predicciones, title=f"Predicción para {producto_seleccionado} ({modelo_prediccion})")
        fig_pred.add_scatter(x=datos_producto_agrupado.index, y=datos_producto_agrupado['Cantidad'], mode='lines', name='Datos Históricos')
        st.plotly_chart(fig_pred)

        # Multiselect para comparar predicciones entre productos
        categorias_disponibles = data['Producto'].unique().tolist()
        categorias_por_defecto = [producto_seleccionado, categorias_disponibles[0] if categorias_disponibles[0] != producto_seleccionado else categorias_disponibles[1]]

        seleccion_categorias = st.multiselect("Selecciona categorías para comparar predicciones", options=categorias_disponibles, default=categorias_por_defecto)
        if st.button("Graficar Comparación de Predicciones"):
            fig_comparacion = px.line()
            for categoria in seleccion_categorias:
                datos_categoria = data[data['Producto'] == categoria].resample('ME', on='Fecha').sum()

                # Convertir las fechas a números
                datos_categoria['MesNumerico'] = (datos_categoria.index.year - datos_categoria.index.year.min()) * 12 + datos_categoria.index.month
                X_categoria = datos_categoria[['MesNumerico']]
                y_categoria = datos_categoria['Cantidad']
                modelo_categoria = LinearRegression().fit(X_categoria, y_categoria)
                
                # Predicciones para la categoría
                meses_futuros_categoria = np.array([X_categoria['MesNumerico'].max() + i for i in range(1, num_meses + 1)]).reshape(-1, 1)
                pred_categoria = modelo_categoria.predict(meses_futuros_categoria)
                fechas_futuras_categoria = pd.date_range(datos_categoria.index.max(), periods=num_meses, freq='ME')

                # Agregar datos históricos y predicciones al gráfico
                fig_comparacion.add_scatter(x=datos_categoria.index, y=datos_categoria['Cantidad'], mode='lines', name=f"Datos Históricos {categoria}")
                fig_comparacion.add_scatter(x=fechas_futuras_categoria, y=pred_categoria, mode='lines', name=f"Predicción {categoria}")
            st.plotly_chart(fig_comparacion)

        # Exportar predicciones a CSV
        if fechas_futuras is not None and predicciones is not None:
            predicciones_df = pd.DataFrame({'Fecha': fechas_futuras, 'Predicción': predicciones})
            buffer = io.BytesIO()
            predicciones_df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.sidebar.download_button(label="Descargar Predicciones en CSV", data=buffer, file_name="predicciones_ventas.csv", mime="text/csv")

# Predicción para todos los productos
if uploaded_file and st.sidebar.checkbox("Realizar predicciones para todos los productos"):
    st.subheader("Predicción para todos los productos")
    predicciones_globales = pd.DataFrame()

    for producto in data['Producto'].unique():
        datos_producto = data[data['Producto'] == producto]
        datos_producto_agrupado = datos_producto.resample('ME', on='Fecha').sum()
        
        # Convertir las fechas a formato numérico
        datos_producto_agrupado['MesNumerico'] = (datos_producto_agrupado.index.year - datos_producto_agrupado.index.year.min()) * 12 + datos_producto_agrupado.index.month
        X = datos_producto_agrupado[['MesNumerico']]
        y = datos_producto_agrupado['Cantidad']
        modelo = LinearRegression().fit(X, y)
        
        # Predicción para los próximos meses
        meses_futuros = np.array([X['MesNumerico'].max() + i for i in range(1, num_meses + 1)]).reshape(-1, 1)
        predicciones = modelo.predict(meses_futuros)
        fechas_futuras = pd.date_range(datos_producto_agrupado.index.max(), periods=num_meses, freq='ME')
        
        # Agregar predicciones al DataFrame global
        predicciones_globales[producto] = predicciones

    predicciones_globales['Fecha'] = fechas_futuras
    st.write(predicciones_globales)

    # Descargar todas las predicciones en CSV
    buffer_global = io.BytesIO()
    predicciones_globales.to_csv(buffer_global, index=False)
    buffer_global.seek(0)
    st.sidebar.download_button(
        label="Descargar Predicciones Globales en CSV",
        data=buffer_global,
        file_name="predicciones_globales.csv",
        mime="text/csv"
    )
