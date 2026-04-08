import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import folium
from streamlit_folium import st_folium

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                            recall_score, precision_score, roc_auc_score, 
                            confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Configuración de página
st.set_page_config(
    page_title="Predictor de Brotes de Dengue - Ecuador",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-header"><h1>🦟 Sistema Predictivo de Brotes de Dengue</h1><p>Ecuador - Predicción a nivel cantonal usando ML y Deep Learning</p></div>', unsafe_allow_html=True)

# ============================================
# GENERADOR DE DATOS SINTÉTICOS REALISTAS
# ============================================

def generar_datos_dengue_ecuador():
    """Genera dataset sintético realista basado en datos epidemiológicos de Ecuador"""
    
    st.info("🔄 Generando dataset sintético de dengue para Ecuador...")
    
    # Cantones de Ecuador (muestra representativa)
    cantones = [
        "Guayaquil", "Quito", "Cuenca", "Santo Domingo", "Machala",
        "Manta", "Portoviejo", "Ambato", "Riobamba", "Loja",
        "Ibarra", "Quevedo", "Milagro", "Esmeraldas", "Babahoyo"
    ]
    
    # Zonas geográficas
    zonas = {
        "Guayaquil": "costa", "Manta": "costa", "Machala": "costa", 
        "Esmeraldas": "costa", "Portoviejo": "costa", "Quevedo": "costa",
        "Quito": "sierra", "Cuenca": "sierra", "Ambato": "sierra",
        "Riobamba": "sierra", "Loja": "sierra", "Ibarra": "sierra",
        "Santo Domingo": "oriente", "Milagro": "oriente", "Babahoyo": "oriente"
    }
    
    # Altitud por cantón (metros)
    altitud = {
        "Guayaquil": 4, "Quito": 2850, "Cuenca": 2550, "Santo Domingo": 600,
        "Machala": 50, "Manta": 6, "Portoviejo": 50, "Ambato": 2577,
        "Riobamba": 2750, "Loja": 2100, "Ibarra": 2225, "Quevedo": 70,
        "Milagro": 12, "Esmeraldas": 15, "Babahoyo": 8
    }
    
    # Generar datos (2020-2024, semanas epidemiológicas)
    fecha_inicio = datetime(2020, 1, 1)
    fecha_fin = datetime(2024, 12, 31)
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq='W')
    
    datos = []
    
    for canton in cantones:
        zona = zonas[canton]
        alt = altitud[canton]
        
        for fecha in fechas:
            # Semana epidemiológica (1-52)
            semana = fecha.isocalendar()[1]
            
            # Estacionalidad: más casos en épocas lluviosas (enero-mayo)
            mes = fecha.month
            estacionalidad = 1 + 0.5 * np.sin(2 * np.pi * (mes - 1) / 12)
            
            # Temperatura según zona y altitud
            if zona == "costa":
                temp_base = 26 + 2 * np.sin(2 * np.pi * (mes - 3) / 12)
                temp = temp_base + np.random.normal(0, 1)
                precipitacion_base = 150 + 100 * np.sin(2 * np.pi * (mes - 3) / 12)
            elif zona == "sierra":
                temp_base = 16 + 2 * np.sin(2 * np.pi * (mes - 3) / 12) - alt/500
                temp = temp_base + np.random.normal(0, 1)
                precipitacion_base = 80 + 60 * np.sin(2 * np.pi * (mes - 3) / 12)
            else:  # oriente
                temp_base = 24 + 2 * np.sin(2 * np.pi * (mes - 3) / 12)
                temp = temp_base + np.random.normal(0, 1)
                precipitacion_base = 200 + 100 * np.sin(2 * np.pi * (mes - 3) / 12)
            
            precipitacion = max(0, precipitacion_base + np.random.normal(0, 30))
            
            # Casos previos (autocorrelación)
            if len(datos) > 0 and datos[-1]['canton'] == canton:
                casos_previos = datos[-1]['casos_semana'] * (0.7 + 0.3 * np.random.random())
            else:
                # Casos base según zona
                if zona == "costa":
                    casos_base = 50
                elif zona == "sierra":
                    casos_base = 10
                else:
                    casos_base = 30
                
                casos_previos = casos_base * estacionalidad
            
            casos_previos = max(0, int(casos_previos + np.random.poisson(5)))
            
            # Riesgo de brote (variable objetivo)
            # Factores que incrementan riesgo:
            riesgo_score = (
                0.3 * (temp > 25) +  # Temperatura alta
                0.2 * (precipitacion > 100) +  # Lluvias
                0.3 * (casos_previos > 30) +  # Casos previos
                0.1 * (semana in [10, 11, 12, 13, 14, 15]) +  # Semanas críticas
                0.1 * (zona == "costa")  # Costa más vulnerable
            )
            
            # Ajuste por altitud (menor riesgo en altura)
            if alt > 2000:
                riesgo_score *= 0.3
            elif alt > 1500:
                riesgo_score *= 0.6
            
            # Variable objetivo binaria (riesgo de brote)
            riesgo_brote = 1 if riesgo_score > 0.5 else 0
            
            # Probabilidad de brote (para regresión)
            prob_brote = np.clip(riesgo_score + np.random.normal(0, 0.1), 0, 1)
            
            datos.append({
                'fecha': fecha,
                'canton': canton,
                'zona': zona,
                'altitud': alt,
                'semana_epidemiologica': semana,
                'temperatura': round(temp, 1),
                'precipitacion': round(precipitacion, 1),
                'casos_previos': int(casos_previos),
                'casos_semana': int(casos_previos * (0.8 + 0.4 * np.random.random())),
                'riesgo_brote': riesgo_brote,
                'probabilidad_brote': round(prob_brote, 3)
            })
    
    df = pd.DataFrame(datos)
    st.success(f"✅ Dataset generado: {len(df)} registros, {df['canton'].nunique()} cantones")
    
    # Mostrar estadísticas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Registros", f"{len(df):,}")
    with col2:
        st.metric("Cantones", df['canton'].nunique())
    with col3:
        st.metric("Brotes (1)", f"{df['riesgo_brote'].sum():,}")
    with col4:
        st.metric("% Brotes", f"{df['riesgo_brote'].mean()*100:.1f}%")
    
    return df

# ============================================
# PIPELINE DE PREPROCESAMIENTO
# ============================================

def preprocesar_datos(df, target_col='riesgo_brote'):
    """Preprocesa datos para modelos ML y DL"""
    
    # Seleccionar características
    feature_cols = ['temperatura', 'precipitacion', 'semana_epidemiologica', 
                    'casos_previos', 'altitud']
    
    # Codificar variables categóricas
    if 'zona' in df.columns:
        zona_map = {'costa': 2, 'sierra': 0, 'oriente': 1}
        df['zona_cod'] = df['zona'].map(zona_map)
        feature_cols.append('zona_cod')
    
    # Preparar X y y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols

# ============================================
# MODELOS MACHINE LEARNING
# ============================================

def entrenar_random_forest(X_train, y_train, X_test, y_test, params=None):
    """Entrena Random Forest con diferentes configuraciones"""
    
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

def entrenar_gradient_boosting(X_train, y_train, X_test, y_test, params=None):
    """Entrena Gradient Boosting"""
    
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

def entrenar_logistic_regression(X_train, y_train, X_test, y_test, params=None):
    """Entrena Regresión Logística"""
    
    if params is None:
        params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

# ============================================
# MODELOS DEEP LEARNING
# ============================================

def crear_mlp(input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.3):
    """Crea modelo MLP (Perceptrón Multicapa)"""
    
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

def entrenar_mlp(X_train, y_train, X_test, y_test, configuracion=1):
    """Entrena MLP con diferentes configuraciones"""
    
    if configuracion == 1:
        hidden_layers = [64, 32, 16]
        dropout_rate = 0.3
        epochs = 50
    elif configuracion == 2:
        hidden_layers = [128, 64, 32, 16]
        dropout_rate = 0.4
        epochs = 100
    else:
        hidden_layers = [32, 16]
        dropout_rate = 0.2
        epochs = 30
    
    model = crear_mlp(X_train.shape[1], hidden_layers, dropout_rate)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return model, y_pred, y_pred_proba, history

# ============================================
# MÉTRICAS DE EVALUACIÓN
# ============================================

def calcular_metricas(y_test, y_pred, y_pred_proba):
    """Calcula todas las métricas de evaluación"""
    
    metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metricas
# ============================================
# MAPA INTERACTIVO DEL ECUADOR
# ============================================

def crear_mapa_ecuador(df_filtrado):
    """Crea un mapa interactivo del Ecuador con datos de brotes"""
    
    # Coordenadas aproximadas de cantones principales
    coordenadas_cantones = {
        "Guayaquil": [-2.1895, -79.8711],
        "Quito": [-0.2299, -78.5249],
        "Cuenca": [-2.8891, -78.9889],
        "Santo Domingo": [0.2469, -79.1733],
        "Machala": [-3.2639, -79.9618],
        "Manta": [-0.9542, -80.7084],
        "Portoviejo": [-1.0545, -80.4544],
        "Ambato": [-1.2290, -78.6341],
        "Riobamba": [-1.6734, -78.6449],
        "Loja": [-4.0344, -79.2061],
        "Ibarra": [0.3520, -78.1197],
        "Quevedo": [0.9430, -79.4729],
        "Milagro": [-2.7331, -79.6063],
        "Esmeraldas": [0.9580, -78.1434],
        "Babahoyo": [-1.8033, -79.5341]
    }
    
    # Calcular estadísticas por cantón
    stats_canton = df_filtrado.groupby('canton').agg({
        'casos_semana': 'sum',
        'riesgo_brote': 'mean',
        'temperatura': 'mean',
        'precipitacion': 'mean'
    }).reset_index()
    
    # Crear mapa centrado en Ecuador
    mapa = folium.Map(
        location=[-1.8312, -78.1834],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Agregar marcadores por cada cantón
    for _, fila in stats_canton.iterrows():
        canton = fila['canton']
        if canton in coordenadas_cantones:
            coords = coordenadas_cantones[canton]
            riesgo = fila['riesgo_brote']
            casos = fila['casos_semana']
            
            # Color según riesgo (rojo = alto, amarillo = medio, verde = bajo)
            if riesgo > 0.6:
                color = 'red'
                icono = '⚠️'
            elif riesgo > 0.3:
                color = 'orange'
                icono = '⚡'
            else:
                color = 'green'
                icono = '✓'
            
            # Popup con información
            popup_text = f"""
            <b>{canton}</b><br>
            Riesgo: {riesgo:.1%}<br>
            Casos: {casos:.0f}<br>
            Temp: {fila['temperatura']:.1f}°C<br>
            Precip: {fila['precipitacion']:.1f}mm
            """
            
            folium.CircleMarker(
                location=coords,
                radius=8 + (riesgo * 10),  # Tamaño proporcional al riesgo
                popup=folium.Popup(popup_text, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(mapa)
    
    return mapa
# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    
    # Inicializar session state
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = False
        st.session_state.resultados = None
    
    # Sidebar único consolidado
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Opción: Usar datos reales o sintéticos
        tipo_datos = st.radio(
            "📊 Fuente de datos",
            ["Generar datos sintéticos", "Cargar archivo propio"],
            key="tipo_datos_radio"
        )
        
        # Cargar datos
        if tipo_datos == "Cargar archivo propio":
            archivo = st.file_uploader("Subir CSV", type=['csv', 'xlsx'])
            if archivo is not None:
                try:
                    if archivo.name.endswith('.csv'):
                        df = pd.read_csv(archivo)
                    else:
                        df = pd.read_excel(archivo)
                    st.success(f"✅ Archivo cargado: {len(df)} registros")
                except Exception as e:
                    st.error(f"Error al cargar: {e}")
                    df = None
            else:
                st.info("Usando datos sintéticos por defecto")
                df = generar_datos_dengue_ecuador()
        else:
            df = generar_datos_dengue_ecuador()
        
        if df is not None:
            # FILTROS
            st.subheader("🔍 Filtros")
            
            # Filtro por zona (si existe)
            if 'zona' in df.columns:
                zonas_unicas = df['zona'].unique().tolist()
                zonas_seleccionadas = st.multiselect(
                    "Zonas geográficas",
                    zonas_unicas,
                    default=zonas_unicas,
                    key="zonas_select"
                )
            else:
                zonas_seleccionadas = None
            
            # Filtro por temperatura (si existe)
            if 'temperatura' in df.columns:
                temp_min, temp_max = st.slider(
                    "Rango de Temperatura (°C)",
                    float(df['temperatura'].min()),
                    float(df['temperatura'].max()),
                    (float(df['temperatura'].min()), float(df['temperatura'].max())),
                    key="temp_slider"
                )
            else:
                temp_min, temp_max = None, None
            
            # Filtro por precipitación (si existe)
            if 'precipitacion' in df.columns:
                precip_min, precip_max = st.slider(
                    "Rango de Precipitación (mm)",
                    float(df['precipitacion'].min()),
                    float(df['precipitacion'].max()),
                    (float(df['precipitacion'].min()), float(df['precipitacion'].max())),
                    key="precip_slider"
                )
            else:
                precip_min, precip_max = None, None
            
            # Filtro por casos previos (si existe)
            if 'casos_previos' in df.columns:
                casos_min, casos_max = st.slider(
                    "Rango de Casos Previos",
                    int(df['casos_previos'].min()),
                    int(df['casos_previos'].max()),
                    (int(df['casos_previos'].min()), int(df['casos_previos'].max())),
                    key="casos_slider"
                )
            else:
                casos_min, casos_max = None, None
            
            # Filtro por riesgo de brote
            if 'riesgo_brote' in df.columns:
                mostrar_brotes = st.checkbox(
                    "Mostrar solo registros con riesgo de brote",
                    value=False,
                    key="brote_checkbox"
                )
            else:
                mostrar_brotes = False
            
            # Aplicar filtros
            df_filtrado = df.copy()
            
            if zonas_seleccionadas is not None:
                df_filtrado = df_filtrado[df_filtrado['zona'].isin(zonas_seleccionadas)]
            
            if temp_min is not None and temp_max is not None:
                df_filtrado = df_filtrado[
                    (df_filtrado['temperatura'] >= temp_min) & 
                    (df_filtrado['temperatura'] <= temp_max)
                ]
            
            if precip_min is not None and precip_max is not None:
                df_filtrado = df_filtrado[
                    (df_filtrado['precipitacion'] >= precip_min) & 
                    (df_filtrado['precipitacion'] <= precip_max)
                ]
            
            if casos_min is not None and casos_max is not None:
                df_filtrado = df_filtrado[
                    (df_filtrado['casos_previos'] >= casos_min) & 
                    (df_filtrado['casos_previos'] <= casos_max)
                ]
            
            if mostrar_brotes and 'riesgo_brote' in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado['riesgo_brote'] == 1]
            
            st.info(f"📊 Registros después de filtrar: {len(df_filtrado)}")
            
            st.markdown("---")
            
            # Selección de modelo
            st.subheader("🤖 Tipo de Modelo")
            tipo_modelo = st.selectbox(
                "Seleccionar modelo",
                ["Random Forest", "Gradient Boosting", "Regresión Logística", "MLP (Deep Learning)"],
                key="modelo_select"
            )
            
            # Configuración de entrenamiento
            st.subheader("🎯 Configuración de Entrenamiento")
            configuracion = st.selectbox(
                "Configuración",
                ["Configuración 1 (Base)", "Configuración 2 (Optimizada)", "Configuración 3 (Rápida)"],
                key="config_select"
            )
            
            # Tamaño de test
            test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2, key="test_slider")
            
            # Botón de entrenamiento
            if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True, key="train_btn"):
                st.session_state.modelo_entrenado = True
                st.session_state.df_filtrado = df_filtrado
                st.session_state.tipo_modelo = tipo_modelo
                st.session_state.configuracion = configuracion
                st.session_state.test_size = test_size
        else:
            st.session_state.modelo_entrenado = False
    
    # Panel principal - Mostrar mapa y resultados si el modelo fue entrenado
    if st.session_state.modelo_entrenado:
        df_filtrado = st.session_state.df_filtrado
        tipo_modelo = st.session_state.tipo_modelo
        configuracion = st.session_state.configuracion
        test_size = st.session_state.test_size
        
        # Mostrar mapa interactivo
        st.header("🗺️ Mapa Interactivo del Ecuador")
        col_map = st.columns(1)
        with col_map[0]:
            mapa = crear_mapa_ecuador(df_filtrado)
            st_folium(mapa, width=1400, height=600)
        
        st.markdown("---")
        
        # Preprocesar datos
        X, y, scaler, features = preprocesar_datos(df_filtrado)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Diccionario de parámetros según configuración
        params_config = {
            "Random Forest": {
                "Configuración 1 (Base)": {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                "Configuración 2 (Optimizada)": {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'random_state': 42},
                "Configuración 3 (Rápida)": {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
            },
            "Gradient Boosting": {
                "Configuración 1 (Base)": {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
                "Configuración 2 (Optimizada)": {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4},
                "Configuración 3 (Rápida)": {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 2}
            },
            "Regresión Logística": {
                "Configuración 1 (Base)": {'C': 1.0},
                "Configuración 2 (Optimizada)": {'C': 0.1},
                "Configuración 3 (Rápida)": {'C': 10.0}
            }
        }
        
        # Entrenar según modelo seleccionado
        st.header("📊 Resultados del Entrenamiento")
        
        # Mostrar información del dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Muestras entrenamiento", len(X_train))
        with col2:
            st.metric("Muestras prueba", len(X_test))
        with col3:
            st.metric("Características", X.shape[1])
        
        st.markdown("---")
        
        # Entrenar modelo
        with st.spinner("🔄 Entrenando modelo..."):
            
            if tipo_modelo == "Random Forest":
                params = params_config["Random Forest"][configuracion]
                model, y_pred, y_pred_proba = entrenar_random_forest(
                    X_train, y_train, X_test, y_test, params
                )
                
            elif tipo_modelo == "Gradient Boosting":
                params = params_config["Gradient Boosting"][configuracion]
                model, y_pred, y_pred_proba = entrenar_gradient_boosting(
                    X_train, y_train, X_test, y_test, params
                )
                
            elif tipo_modelo == "Regresión Logística":
                params = params_config["Regresión Logística"][configuracion]
                model, y_pred, y_pred_proba = entrenar_logistic_regression(
                    X_train, y_train, X_test, y_test, params
                )
                
            else:  # MLP
                config_num = 1 if configuracion == "Configuración 1 (Base)" else (2 if configuracion == "Configuración 2 (Optimizada)" else 3)
                model, y_pred, y_pred_proba, history = entrenar_mlp(
                    X_train, y_train, X_test, y_test, config_num
                )
                
                # Graficar historial de entrenamiento DL
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(y=history.history['loss'], name='Loss entrenamiento'))
                fig_hist.add_trace(go.Scatter(y=history.history['val_loss'], name='Loss validación'))
                fig_hist.update_layout(title='Historial de Entrenamiento', xaxis_title='Época', yaxis_title='Loss')
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Calcular métricas
            metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
            
            # Mostrar métricas
            st.subheader("📈 Métricas de Evaluación")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metricas['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metricas['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metricas['recall']:.2%}")
            with col4:
                st.metric("F1-Score", f"{metricas['f1']:.2%}")
            with col5:
                st.metric("ROC-AUC", f"{metricas['roc_auc']:.2%}")
            
            # Matriz de confusión
            st.subheader("🎯 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, 
                              title="Matriz de Confusión",
                              labels=dict(x="Predicción", y="Real"),
                              x=['Sin Brote', 'Con Brote'],
                              y=['Sin Brote', 'Con Brote'],
                              color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Curva ROC
            st.subheader("📉 Curva ROC")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={metricas["roc_auc"]:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Clasificador Aleatorio', line=dict(dash='dash')))
            fig_roc.update_layout(xaxis_title='Tasa de Falsos Positivos', yaxis_title='Tasa de Verdaderos Positivos')
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Importancia de características (solo para ML)
            if tipo_modelo in ["Random Forest", "Gradient Boosting"]:
                st.subheader("🔍 Importancia de Características")
                importancia = pd.DataFrame({
                    'Característica': features,
                    'Importancia': model.feature_importances_
                }).sort_values('Importancia', ascending=True)
                
                fig_imp = px.bar(importancia, x='Importancia', y='Característica', 
                                orientation='h', title="Importancia de Variables",
                                color='Importancia', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # Guardar modelo
            joblib.dump(model, 'modelo_dengue.pkl')
            joblib.dump(scaler, 'scaler_dengue.pkl')
            st.success("✅ Modelo guardado exitosamente!")
            
            # Sección de predicciones
            st.header("🔮 Predicción de Nuevos Casos")
            st.markdown("Ingrese los datos para predecir riesgo de brote:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temp_input = st.number_input("🌡️ Temperatura (°C)", min_value=10.0, max_value=35.0, value=25.0, key="temp_input")
                precip_input = st.number_input("☔ Precipitación (mm)", min_value=0.0, max_value=500.0, value=100.0, key="precip_input")
                semana_input = st.number_input("📅 Semana Epidemiológica", min_value=1, max_value=52, value=20, key="semana_input")
            
            with col2:
                casos_input = st.number_input("📊 Casos previos", min_value=0, max_value=1000, value=50, key="casos_input")
                altitud_input = st.number_input("⛰️ Altitud (msnm)", min_value=0, max_value=4000, value=500, key="altitud_input")
                zona_input = st.selectbox("🗺️ Zona geográfica", ["costa", "sierra", "oriente"], key="zona_input")
            
            if st.button("🔮 Predecir Riesgo", type="primary", key="predict_btn"):
                # Crear dataframe con entrada
                zona_map = {'costa': 2, 'sierra': 0, 'oriente': 1}
                entrada = pd.DataFrame({
                    'temperatura': [temp_input],
                    'precipitacion': [precip_input],
                    'semana_epidemiologica': [semana_input],
                    'casos_previos': [casos_input],
                    'altitud': [altitud_input],
                    'zona_cod': [zona_map[zona_input]]
                })
                
                # Escalar
                entrada_scaled = scaler.transform(entrada[features])
                
                # Predecir
                if tipo_modelo == "MLP (Deep Learning)":
                    prob = model.predict(entrada_scaled, verbose=0)[0][0]
                else:
                    prob = model.predict_proba(entrada_scaled)[0][1]
                
                riesgo = "ALTO" if prob > 0.5 else "BAJO"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {'#ff6b6b' if prob>0.5 else '#51cf66'} 0%, {'#ee5a24' if prob>0.5 else '#37b24d'} 100%); 
                            padding: 2rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
                    <h2 style="color: white;">Riesgo de Brote: {riesgo}</h2>
                    <p style="color: white; font-size: 24px;">Probabilidad: {prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recomendaciones
                st.subheader("📋 Recomendaciones")
                if prob > 0.7:
                    st.error("🔴 **Alerta Máxima:** Intensificar fumigación, eliminar criaderos, activar protocolos de emergencia")
                elif prob > 0.4:
                    st.warning("🟡 **Alerta Media:** Reforzar vigilancia, campañas de prevención, monitoreo de casos")
                else:
                    st.info("🟢 **Alerta Baja:** Mantener vigilancia rutinaria, preparar equipos de respuesta")
    
    else:
        # Pantalla de inicio
        st.info("👈 Configure los parámetros en el panel lateral y presione 'Entrenar Modelo'")
        
        # Mostrar información del proyecto
        with st.expander("📖 Acerca del Proyecto"):
            st.markdown("""
            ### 🦟 Predicción de Brotes de Dengue en Ecuador
            
            **Contexto:** Ecuador registró 27,838 casos de dengue en 2023, con tendencia al alza y expansión hacia zonas andinas.
            
            **Variables Epidemiológicas:**
            - 🌡️ Temperatura (óptimo 25-30°C para Aedes aegypti)
            - ☔ Precipitación (favorece criaderos)
            - 📅 Semana epidemiológica (estacionalidad)
            - 📊 Casos previos (autocorrelación temporal)
            - ⛰️ Altitud (sobre 2000m menor riesgo)
            - 🗺️ Zona geográfica (costa más vulnerable)
            
            **Modelos Implementados:**
            - Random Forest
            - Gradient Boosting
            - Regresión Logística
            - MLP (Deep Learning)
            
            **Métricas de Evaluación:**
            - **Accuracy:** Proporción de predicciones correctas
            - **Precision:** Exactitud de predicciones positivas
            - **Recall:** Sensibilidad para detectar brotes
            - **F1-Score:** Balance entre precisión y sensibilidad
            - **ROC-AUC:** Capacidad discriminativa del modelo
            
            **Hiperparámetros Ajustables:**
            - Random Forest: n_estimators, max_depth, min_samples_split
            - Gradient Boosting: n_estimators, learning_rate, max_depth
            - MLP: capas ocultas, dropout rate, epochs
            """)

if __name__ == "__main__":
    main()            
