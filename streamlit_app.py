import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# ---------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Smartphone Price AI", layout="wide")

CLASSES = {
    0: "Niedrig (Low Cost)",
    1: "Mittel (Medium Cost)",
    2: "Hoch (High Cost)",
    3: "Sehr Hoch (Very High Cost)"
}

# Farben für Plotly
COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

# ---------------------------------------------------------
# BACKEND (Mit Caching für Performance)
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    try:
        # Daten laden
        # HINWEIS: In der Cloud muss die Datei im gleichen Ordner liegen!
        df = pd.read_csv('train.csv')
        df = df[df['px_height'] != 0]
        
        # Feature Engineering
        df['screen_area'] = df['sc_h'] * df['sc_w']
        df['screen_area'] = df['screen_area'].replace(0, df['screen_area'].median())
        df['pixel_density'] = (df['px_height'] * df['px_width']) / df['screen_area']
        df['camera_quality'] = df['fc'] + df['pc']

        X = df.drop('price_range', axis=1)
        y = df['price_range']
        
        feature_names = X.columns
        defaults = df.median()
        
        # Min/Max für Radar Chart
        ranges = {}
        for col in defaults.index:
            if col in df.columns:
                ranges[col] = (df[col].min(), df[col].max())

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, defaults, feature_names, ranges
    except Exception as e:
        return None, None, None, None

# Modell laden
model, defaults, feature_names, ranges = train_model()

# ---------------------------------------------------------
# FRONTEND (Webseite)
# ---------------------------------------------------------
st.title("📱 Smartphone Price Predictor")
st.markdown("**ksterarbeit**")

if model is None:
    st.error("Fehler: 'train.csv' wurde nicht gefunden. Bitte Datei hochladen.")
    st.stop()

# Layout: 2 Spalten (Links Inputs, Rechts Ergebnisse)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("⚙️ Konfiguration")
    
    ram = st.slider("Arbeitsspeicher (RAM)", 256, 4000, 2000, format="%d MB")
    battery = st.slider("Batteriekapazität", 500, 4000, 3000, format="%d mAh")
    px_w = st.slider("Auflösung Breite", 500, 2000, 1080, format="%d px")
    px_h = st.slider("Auflösung Höhe", 500, 2000, 1920, format="%d px")
    int_mem = st.slider("Interner Speicher", 2, 128, 64, format="%d GB")
    pc = st.slider("Kamerauflösung", 0, 20, 10, format="%d MP")

    # Inputs sammeln
    inputs = defaults.copy()
    inputs['ram'] = ram
    inputs['battery_power'] = battery
    inputs['px_width'] = px_w
    inputs['px_height'] = px_h
    inputs['int_memory'] = int_mem
    inputs['pc'] = pc

    # Vorhersage
    df_pred = pd.DataFrame([inputs])
    df_pred['screen_area'] = df_pred['sc_h'] * df_pred['sc_w']
    if df_pred['screen_area'].iloc[0] == 0: df_pred['screen_area'] = 1
    df_pred['pixel_density'] = (df_pred['px_height'] * df_pred['px_width']) / df_pred['screen_area']
    df_pred['camera_quality'] = df_pred['fc'] + df_pred['pc']
    df_pred = df_pred[feature_names]
    
    probs = model.predict_proba(df_pred)[0]
    pred_class = np.argmax(probs)

with col2:
    st.subheader("📊 Analyse & Ergebnis")
    
    # 1. Grosses Ergebnis
    st.metric(label="Vorhergesagte Klasse", value=CLASSES[pred_class])
    
    # Farbiger Balken je nach Klasse
    st.progress(int(probs[pred_class] * 100))
    st.caption(f"Sicherheit des Modells: {probs[pred_class]:.1%}")

    # Tabs für Grafiken
    tab1, tab2 = st.tabs(["Performance Fingerprint", "Wahrscheinlichkeiten"])

    with tab1:
        # RADAR CHART mit Plotly
        # Normalisieren der Werte für 0-1 Skala
        r_norm = (ram - ranges['ram'][0]) / (ranges['ram'][1] - ranges['ram'][0])
        b_norm = (battery - ranges['battery_power'][0]) / (ranges['battery_power'][1] - ranges['battery_power'][0])
        p_norm = (px_h - ranges['px_height'][0]) / (ranges['px_height'][1] - ranges['px_height'][0])
        
        fig = go.Figure(data=go.Scatterpolar(
            r=[r_norm, b_norm, p_norm, r_norm],
            theta=['RAM', 'Battery', 'Display', 'RAM'],
            fill='toself',
            name='Smartphone',
            line_color=COLORS[pred_class]
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=300,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # BALKENDIAGRAMM
        bar_data = pd.DataFrame({
            'Klasse': ["Low", "Medium", "High", "Very High"],
            'Wahrscheinlichkeit': probs
        })
        st.bar_chart(bar_data.set_index('Klasse'), color=COLORS[pred_class])
