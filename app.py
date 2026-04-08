
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
st.set_page_config(
    page_title="Dengue Ecuador Dashboard",
    layout="wide"
)

# -----------------------------
# ESTILOS (TAILWIND-LIKE)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    .title {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    .subtitle {
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="title">🦠 Sistema de Vigilancia de Dengue - Ecuador</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicción de riesgo de brotes a nivel cantonal</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# SIDEBAR (FILTROS)
# -----------------------------
st.sidebar.title("🎛️ Filtros")

canton = st.sidebar.selectbox(
    "Seleccionar Cantón",
    ["Todos", "Quito", "Guayaquil", "Cuenca", "Machala"]
)

fecha = st.sidebar.date_input("Seleccionar fecha")

riesgo = st.sidebar.selectbox(
    "Nivel de riesgo",
    ["Todos", "Bajo", "Medio", "Alto"]
)

st.sidebar.markdown("---")
st.sidebar.write("🔎 Ajusta los filtros para analizar el riesgo")

# -----------------------------
# KPIs (TARJETAS)
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">🔥 <b>Casos totales</b><br><h2>1,245</h2></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">📈 <b>Riesgo alto</b><br><h2>32%</h2></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">🌧️ <b>Precipitación</b><br><h2>78 mm</h2></div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">🌡️ <b>Temperatura</b><br><h2>27°C</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# MAPA (PLACEHOLDER)
# -----------------------------
st.subheader("🗺️ Mapa de riesgo de dengue")

# Datos dummy para mapa
df_map = pd.DataFrame({
    "lat": [-0.18, -2.17, -3.99, -2.90],
    "lon": [-78.47, -79.92, -79.20, -79.00],
    "ciudad": ["Quito", "Guayaquil", "Cuenca", "Machala"],
    "riesgo": ["Alto", "Medio", "Bajo", "Alto"]
})

fig_map = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    color="riesgo",
    size=[10, 15, 8, 12],
    hover_name="ciudad",
    zoom=5,
    height=500
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# GRÁFICOS
# -----------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("### 📈 Tendencia de casos")
    df_chart = pd.DataFrame({
        "fecha": pd.date_range(start="2022-01-01", periods=10),
        "casos": [10, 20, 15, 30, 50, 40, 60, 80, 70, 90]
    })
    fig = px.line(df_chart, x="fecha", y="casos")
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.markdown("### 📊 Variables climáticas")
    df_bar = pd.DataFrame({
        "variable": ["Temperatura", "Precipitación"],
        "valor": [27, 78]
    })
    fig2 = px.bar(df_bar, x="variable", y="valor")
    st.plotly_chart(fig2, use_container_width=True)