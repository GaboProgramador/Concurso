# =========================================
# PREDICCIÓN DE BROTES DE DENGUE - ECUADOR
# =========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score

import streamlit as st

# =========================================
# 1. CARGA DE DATOS
# =========================================

def cargar_datos(ruta):
    df = pd.read_excel(ruta)

    # Ajusta nombres según tu Excel
    df.columns = df.columns.str.lower()

    # Suponiendo columnas: fecha, canton, casos
    df["fecha"] = pd.to_datetime(df["fecha"])

    return df


# =========================================
# 2. FEATURE ENGINEERING
# =========================================

def crear_features(df):
    df = df.sort_values(["canton", "fecha"])

    # Semana epidemiológica
    df["semana_epi"] = df["fecha"].dt.isocalendar().week.astype(int)

    # Lags
    df["casos_prev_1"] = df.groupby("canton")["casos"].shift(1)
    df["casos_prev_2"] = df.groupby("canton")["casos"].shift(2)

    # Media móvil
    df["media_3_sem"] = df.groupby("canton")["casos"].rolling(3).mean().reset_index(0, drop=True)

    # ==========================
    # VARIABLES SINTÉTICAS
    # ==========================
    np.random.seed(42)

    df["temperatura"] = 25 + 5*np.sin(2*np.pi*df["semana_epi"]/52) + np.random.normal(0, 1, len(df))
    df["precipitacion"] = 100 + 80*np.sin(2*np.pi*(df["semana_epi"]-10)/52) + np.random.normal(0, 10, len(df))

    # ==========================
    # VARIABLE OBJETIVO
    # ==========================
    def clasificar_riesgo(x):
        if x < 5:
            return "Bajo"
        elif x < 20:
            return "Medio"
        else:
            return "Alto"

    df["riesgo"] = df["casos"].apply(clasificar_riesgo)

    df = df.dropna()

    return df


# =========================================
# 3. PREPARACIÓN
# =========================================

def preparar_datos(df):
    features = [
        "temperatura", "precipitacion",
        "semana_epi", "casos_prev_1",
        "casos_prev_2", "media_3_sem"
    ]

    X = df[features]
    y = df["riesgo"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================================
# 4. ENTRENAMIENTO
# =========================================

def entrenar_modelos(X_train, X_test, y_train, y_test):

    resultados = {}

    # ==========================
    # EXPERIMENTO 1: BASELINE
    # ==========================
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X_train, y_train)

    pred1 = model1.predict(X_test)

    resultados["baseline"] = {
        "accuracy": accuracy_score(y_test, pred1),
        "f1": f1_score(y_test, pred1, average="weighted"),
        "recall": recall_score(y_test, pred1, average="weighted")
    }

    # ==========================
    # EXPERIMENTO 2: TUNING
    # ==========================
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid,
                        cv=3,
                        scoring="f1_weighted")

    grid.fit(X_train, y_train)

    model2 = grid.best_estimator_

    pred2 = model2.predict(X_test)

    resultados["tuning"] = {
        "accuracy": accuracy_score(y_test, pred2),
        "f1": f1_score(y_test, pred2, average="weighted"),
        "recall": recall_score(y_test, pred2, average="weighted")
    }

    # ==========================
    # EXPERIMENTO 3: MODELO MÁS COMPLEJO
    # ==========================
    model3 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model3.fit(X_train, y_train)

    pred3 = model3.predict(X_test)

    resultados["complejo"] = {
        "accuracy": accuracy_score(y_test, pred3),
        "f1": f1_score(y_test, pred3, average="weighted"),
        "recall": recall_score(y_test, pred3, average="weighted")
    }

    return model2, resultados


# =========================================
# 5. GUARDAR MODELO
# =========================================

def guardar_modelo(modelo):
    joblib.dump(modelo, "modelo_dengue.pkl")


# =========================================
# 6. STREAMLIT APP
# =========================================

def app_streamlit():

    st.title("Predicción de Brotes de Dengue")

    try:
        model = joblib.load("modelo_dengue.pkl")
    except:
        st.warning("Entrena el modelo primero.")
        return

    temp = st.slider("Temperatura", 15, 35, 25)
    prec = st.slider("Precipitación", 0, 300, 100)
    sem = st.slider("Semana epidemiológica", 1, 52, 10)
    prev1 = st.number_input("Casos semana anterior", 0, 100, 5)
    prev2 = st.number_input("Casos hace 2 semanas", 0, 100, 3)
    media = st.number_input("Media 3 semanas", 0.0, 100.0, 4.0)

    if st.button("Predecir"):
        X = [[temp, prec, sem, prev1, prev2, media]]
        pred = model.predict(X)
        st.success(f"Riesgo estimado: {pred[0]}")


# =========================================
# MAIN
# =========================================

def main():

    ruta = "Datos_Dengue_MSP_Ene2021_Ago2025.xlsx"

    df = cargar_datos(ruta)
    df = crear_features(df)

    X_train, X_test, y_train, y_test = preparar_datos(df)

    modelo, resultados = entrenar_modelos(X_train, X_test, y_train, y_test)

    guardar_modelo(modelo)

    print("\nRESULTADOS:")
    for k, v in resultados.items():
        print(f"\n{k.upper()}:")
        for metrica, valor in v.items():
            print(f"{metrica}: {valor:.4f}")

    print("\nModelo guardado como modelo_dengue.pkl")


if __name__ == "__main__":
    main()