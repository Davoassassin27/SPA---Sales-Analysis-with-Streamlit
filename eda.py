# eda.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


def app():
    st.title("Análisis Exploratorio de Datos (EDA)")

    # Cargar archivo CSV
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type="csv")
    if not uploaded_file:
        st.warning("Por favor, sube un archivo CSV para comenzar.")
        return

    # Cargar datos y verificar integridad
    data = cargar_datos(uploaded_file)
    if data is None:
        return

    # Mostrar las primeras filas de los datos
    st.subheader("Vista Previa de los Datos")
    st.write(data.head())

    # Seleccionar producto para análisis
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto para analizar", data['Producto'].unique())
    datos_producto = data[data['Producto'] == producto_seleccionado]

    # Filtrar por rango de fechas
    st.sidebar.markdown("### Filtro de Fecha")
    fecha_min, fecha_max = datos_producto["Fecha"].min(), datos_producto["Fecha"].max()
    rango_fechas = st.sidebar.date_input("Selecciona un rango de fechas", [fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max)
    datos_producto = datos_producto[(datos_producto["Fecha"] >= pd.to_datetime(rango_fechas[0])) & (datos_producto["Fecha"] <= pd.to_datetime(rango_fechas[1]))]

    # Agrupación y estadísticas avanzadas
    datos_producto = datos_producto.set_index("Fecha").resample('M').sum()
    datos_producto["Variación (%)"] = datos_producto["Cantidad"].pct_change() * 100
    datos_producto["Crecimiento Acumulado"] = datos_producto["Cantidad"].cumsum()

    # Histograma de distribución de ventas
    with st.expander(f"Distribución de las ventas del producto: {producto_seleccionado}", expanded=False):
        fig_hist = px.histogram(datos_producto, x='Cantidad', nbins=20, title=f'Distribución de ventas para {producto_seleccionado}')
        st.plotly_chart(fig_hist)

    # Gráfico combinado de ventas mensuales y crecimiento acumulado
    with st.expander(f"Ventas mensuales y crecimiento acumulado de {producto_seleccionado}", expanded=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos_producto.index, y=datos_producto['Cantidad'], name='Ventas Mensuales', mode='lines', yaxis='y1'))
        fig.add_trace(go.Scatter(x=datos_producto.index, y=datos_producto['Crecimiento Acumulado'], name='Crecimiento Acumulado', mode='lines', yaxis='y2', line=dict(color='red')))
        
        # Configuración de ejes
        fig.update_layout(
            title=f"Ventas Mensuales y Crecimiento Acumulado de {producto_seleccionado}",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="Cantidad"),
            yaxis2=dict(title="Crecimiento Acumulado", overlaying='y', side='right')
        )
        st.plotly_chart(fig)

    # Mostrar estadísticas
    with st.expander(f"Estadísticas descriptivas para {producto_seleccionado}", expanded=False):
        st.write(datos_producto[["Cantidad", "Variación (%)", "Crecimiento Acumulado"]].describe())
