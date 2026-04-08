import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import openpyxl
import os

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                            precision_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# CONFIGURACIÓN
# ============================================
st.set_page_config(
    page_title="Dengue Ecuador Dashboard - ML Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
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

# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================

@st.cache_data
def cargar_datos_xlsx(ruta_archivo):
    """Carga datos desde archivo xlsx"""
    try:
        df = pd.read_excel(ruta_archivo)
        # Preparar columnas para que sean minúsculas sin espacios
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('ñ', 'n')
        
        # Convertir fechas si existen
        date_cols = df.select_dtypes(include=['object']).columns
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        st.success(f"✅ Datos cargados: {len(df)} registros")
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar archivo: {e}")
        return None

# ============================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================

def preprocesar_datos(df, target_col='riesgo_brote', features_auto=True):
    """Preprocesa datos para modelos"""
    try:
        # Seleccionar características numéricas
        if features_auto:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir la columna target
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        else:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Verificar que existe la columna target
        if target_col not in df.columns:
            st.error(f"❌ Columna '{target_col}' no encontrada. Columnas disponibles: {df.columns.tolist()}")
            return None, None, None, None
        
        # Verificar que hay features
        if len(feature_cols) == 0:
            st.error("❌ No hay columnas numéricas para usar como features")
            return None, None, None, None
        
        # Preparar X y y
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col].fillna(0)
        
        # Codificar variables categóricas
        le_dict = {}
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        return X_scaled, y, scaler, feature_cols
    
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        return None, None, None, None

# ============================================
# MODELOS MACHINE LEARNING
# ============================================

def entrenar_random_forest(X_train, y_train, X_test, y_test, params=None):
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

def entrenar_gradient_boosting(X_train, y_train, X_test, y_test, params=None):
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

def entrenar_logistic_regression(X_train, y_train, X_test, y_test, params=None):
    if params is None:
        params = {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

# ============================================
# MODELOS DEEP LEARNING
# ============================================

def crear_mlp(input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.3):
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
    if configuracion == 1:
        hidden_layers = [64, 32, 16]
        dropout_rate = 0.3
        epochs = 30
    elif configuracion == 2:
        hidden_layers = [128, 64, 32, 16]
        dropout_rate = 0.4
        epochs = 50
    else:
        hidden_layers = [32, 16]
        dropout_rate = 0.2
        epochs = 20
    
    model = crear_mlp(X_train.shape[1], hidden_layers, dropout_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return model, y_pred, y_pred_proba, history

# ============================================
# MÉTRICAS
# ============================================

def calcular_metricas(y_test, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

# ============================================
# INTERFAZ PRINCIPAL
# ============================================

def main():
    st.markdown('<div class="main-header"><h1>🦟 Sistema de Vigilancia de Dengue - Ecuador</h1><p>Predicción de riesgo con ML y Deep Learning</p></div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Opción de datos
        tipo_datos = st.radio(
            "📊 Fuente de datos",
            ["Cargar archivo xlsx", "Datos de ejemplo"],
            key="tipo_datos_radio"
        )
        
        df = None
        target_col = None
        
        if tipo_datos == "Cargar archivo xlsx":
            archivo = st.file_uploader("Subir archivo xlsx", type=['xlsx', 'xls'])
            if archivo is not None:
                df = cargar_datos_xlsx(archivo)
            else:
                # Intentar cargar automáticamente
                ruta_default = "Datos_Dengue_MSP_Ene2021_Ago2025.xlsx"
                if os.path.exists(ruta_default):
                    st.info("📂 Cargando archivo por defecto...")
                    df = cargar_datos_xlsx(ruta_default)
                else:
                    st.warning("⚠️ No se encontró archivo")
        else:
            st.info("📊 Usando datos de ejemplo")
        
        # Mostrar info del archivo cargado
        if df is not None:
            with st.expander("📋 Información del Dataset"):
                st.write(f"**Registros:** {len(df)}")
                st.write(f"**Columnas:** {df.shape[1]}")
                st.write("**Columnas disponibles:**")
                for col in df.columns:
                    dtype = df[col].dtype
                    st.write(f"  - {col}: {dtype}")
        
        st.markdown("---")
        
        if df is not None:
            # SELECCIÓN DE COLUMNA TARGET
            st.subheader("🎯 Seleccionar Variable Objetivo")
            
            # Sugerencias automáticas
            colnames_lower = [col.lower() for col in df.columns]
            
            # Palabras clave para detectar target automáticamente
            keywords = ['riesgo', 'brote', 'dengue', 'confirmed', 'positivo', 'caso', 'diagnostico']
            
            suggested_col = None
            for keyword in keywords:
                for i, col_lower in enumerate(colnames_lower):
                    if keyword in col_lower:
                        suggested_col = df.columns[i]
                        break
                if suggested_col:
                    break
            
            # Selectbox de columna target
            if suggested_col:
                idx = list(df.columns).index(suggested_col)
            else:
                idx = 0
            
            target_col = st.selectbox(
                "Columna objetivo (Variable a predecir)",
                df.columns,
                index=idx,
                key="target_col_select"
            )
            
            # FILTROS
            st.subheader("🔍 Filtros")
            
            # Detectar columnas numéricas para filtros
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col_filtro = st.selectbox("Columna para filtrar", numeric_cols, key="col_filtro")
                min_val, max_val = st.slider(
                    f"Rango de {col_filtro}",
                    float(df[col_filtro].min()),
                    float(df[col_filtro].max()),
                    (float(df[col_filtro].min()), float(df[col_filtro].max())),
                    key="range_slider"
                )
                df_filtrado = df[(df[col_filtro] >= min_val) & (df[col_filtro] <= max_val)]
            else:
                df_filtrado = df
            
            st.info(f"📊 Registros después de filtrar: {len(df_filtrado)}")
            
            st.markdown("---")
            
            # MODELO
            st.subheader("🤖 Modelo de ML/DL")
            tipo_modelo = st.selectbox(
                "Seleccionar modelo",
                ["Random Forest", "Gradient Boosting", "Regresión Logística", "MLP (Deep Learning)"],
                key="modelo_select"
            )
            
            configuracion = st.selectbox(
                "Configuración",
                ["Base", "Optimizada", "Rápida"],
                key="config_select"
            )
            
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, key="test_slider")
            
            entrenar_btn = st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True, key="train_btn")
        else:
            entrenar_btn = False
            df_filtrado = None
    
    # CONTENIDO PRINCIPAL
    if df is not None and entrenar_btn and target_col is not None:
        
        # Preprocesar
        X, y, scaler, features = preprocesar_datos(df_filtrado, target_col=target_col)
        
        if X is None:
            st.error("❌ No se pudo preprocesar los datos")
            return
        
        # Verificar que hay datos suficientes
        if len(X) < 10:
            st.error("❌ Datos insuficientes para entrenar (mínimo 10 registros)")
            return
        
        # Dividir datos
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except Exception as e:
            st.error(f"❌ Error al dividir datos: {str(e)}")
            return
        
        st.header("📊 Resultados del Entrenamiento")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train", len(X_train))
        with col2:
            st.metric("Test", len(X_test))
        with col3:
            st.metric("Features", X.shape[1])
        
        st.markdown("---")
        
        # Entrenar
        with st.spinner("🔄 Entrenando modelo..."):
            
            params_config = {
                "Random Forest": {
                    "Base": {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                    "Optimizada": {'n_estimators': 200, 'max_depth': 20, 'random_state': 42},
                    "Rápida": {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
                },
                "Gradient Boosting": {
                    "Base": {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
                    "Optimizada": {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4},
                    "Rápida": {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 2}
                },
                "Regresión Logística": {
                    "Base": {'C': 1.0},
                    "Optimizada": {'C': 0.1},
                    "Rápida": {'C': 10.0}
                }
            }
            
            try:
                if tipo_modelo == "Random Forest":
                    params = params_config["Random Forest"][configuracion]
                    model, y_pred, y_pred_proba = entrenar_random_forest(X_train, y_train, X_test, y_test, params)
                    
                elif tipo_modelo == "Gradient Boosting":
                    params = params_config["Gradient Boosting"][configuracion]
                    model, y_pred, y_pred_proba = entrenar_gradient_boosting(X_train, y_train, X_test, y_test, params)
                    
                elif tipo_modelo == "Regresión Logística":
                    params = params_config["Regresión Logística"][configuracion]
                    model, y_pred, y_pred_proba = entrenar_logistic_regression(X_train, y_train, X_test, y_test, params)
                    
                else:  # MLP
                    config_num = 1 if configuracion == "Base" else (2 if configuracion == "Optimizada" else 3)
                    model, y_pred, y_pred_proba, history = entrenar_mlp(X_train, y_train, X_test, y_test, config_num)
                    
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(y=history.history['loss'], name='Loss entrenamiento'))
                    fig_hist.add_trace(go.Scatter(y=history.history['val_loss'], name='Loss validación'))
                    fig_hist.update_layout(title='Historial de Entrenamiento', xaxis_title='Época', yaxis_title='Loss')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Métricas
                metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
                
                st.subheader("📈 Métricas")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{metricas['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{metricas['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{metricas['recall']:.2%}")
                with col4:
                    st.metric("F1", f"{metricas['f1']:.2%}")
                with col5:
                    st.metric("ROC-AUC", f"{metricas['roc_auc']:.2%}")
                
                st.markdown("---")
                
                # Matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, title="Matriz de Confusión",
                                  labels=dict(x="Predicción", y="Real"),
                                  color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={metricas["roc_auc"]:.3f})'))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Aleatorio', line=dict(dash='dash')))
                fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Importancia (solo ML)
                if tipo_modelo in ["Random Forest", "Gradient Boosting"]:
                    st.subheader("🔍 Importancia de Variables")
                    importancia = pd.DataFrame({
                        'Variable': features,
                        'Importancia': model.feature_importances_
                    }).sort_values('Importancia', ascending=True).tail(10)
                    
                    fig_imp = px.bar(importancia, x='Importancia', y='Variable', orientation='h',
                                    color='Importancia', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Guardar modelo
                joblib.dump(model, 'modelo_dengue.pkl')
                joblib.dump(scaler, 'scaler_dengue.pkl')
                st.success("✅ Modelo guardado")
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")
    
    else:
        st.info("👈 Carga datos en el panel lateral para comenzar")
        
        with st.expander("📖 Información"):
            st.markdown("""
            ### Características
            - **Datos reales:** Carga tu archivo xlsx con datos de dengue
            - **Selección flexible:** Elige qué columna predecir
            - **ML:** Random Forest, Gradient Boosting, Regresión Logística
            - **DL:** MLP (Perceptrón Multicapa)
            - **Métricas:** Accuracy, Precision, Recall, F1, ROC-AUC
            - **Visualizaciones:** Matriz de confusión, Curva ROC, Importancia de variables
            """)

if __name__ == "__main__":
    main()
