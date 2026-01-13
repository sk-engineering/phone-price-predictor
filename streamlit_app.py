import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Smartphone AI Analyst", layout="wide", page_icon="📱")

CLASSES = {
    0: "Niedrig (Low Cost)",
    1: "Mittel (Medium Cost)",
    2: "Hoch (High Cost)",
    3: "Sehr Hoch (Very High Cost)"
}

# Farben passend zu Plotly (Grün, Blau, Orange, Rot)
COLOR_MAP = {
    0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"
}
# Für die Heatmap (0=Grün, 1=Blau, 2=Orange, 3=Rot)
HEATMAP_COLORS = [
    [0.0, "#2ecc71"], [0.25, "#2ecc71"],
    [0.25, "#3498db"], [0.5, "#3498db"],
    [0.5, "#f39c12"], [0.75, "#f39c12"],
    [0.75, "#e74c3c"], [1.0, "#e74c3c"]
]

# ---------------------------------------------------------
# BACKEND (Mit Caching für Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Daten laden
    try:
        df = pd.read_csv('train.csv')
    except:
        return None, None, None, None

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
    
    ranges = {}
    for col in defaults.index:
        if col in df.columns:
            ranges[col] = (df[col].min(), df[col].max())

    # 100 Bäume für das Mosaik
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, defaults, feature_names, ranges

model, defaults, feature_names, ranges = load_model()

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def get_individual_votes(model, input_df):
    """Holt die 100 Einzelmeinungen für das Mosaik"""
    votes = []
    # Wir iterieren durch alle 100 Bäume im Wald
    for tree in model.estimators_:
        vote = tree.predict(input_df.to_numpy())[0]
        votes.append(int(vote))
    return votes

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("📱 Smartphone Price AI")
st.markdown("Semesterarbeit DSBE | **Kiliç & Keller**")

if model is None:
    st.error("⚠️ 'train.csv' fehlt! Bitte laden Sie die Datei in das GitHub Repository hoch.")
    st.stop()

# Spalten-Layout
left_col, right_col = st.columns([1, 1.5])

# --- LINKE SPALTE: INPUTS ---
with left_col:
    st.subheader("⚙️ Konfiguration")
    st.info("Verstellen Sie die Werte, um das KI-Modell live herauszufordern.")
    
    ram = st.slider("Arbeitsspeicher (RAM)", 256, 4000, 2000, format="%d MB")
    battery = st.slider("Batteriekapazität", 500, 4000, 3000, format="%d mAh")
    px_w = st.slider("Auflösung Breite", 500, 2000, 1080, format="%d px")
    px_h = st.slider("Auflösung Höhe", 500, 2000, 1920, format="%d px")
    int_mem = st.slider("Interner Speicher", 2, 128, 64, format="%d GB")
    pc = st.slider("Kamerauflösung", 0, 20, 10, format="%d MP")

    # Daten aufbereiten
    inputs = defaults.copy()
    inputs['ram'] = ram
    inputs['battery_power'] = battery
    inputs['px_width'] = px_w
    inputs['px_height'] = px_h
    inputs['int_memory'] = int_mem
    inputs['pc'] = pc

    df_pred = pd.DataFrame([inputs])
    df_pred['screen_area'] = df_pred['sc_h'] * df_pred['sc_w']
    if df_pred['screen_area'].iloc[0] == 0: df_pred['screen_area'] = 1
    df_pred['pixel_density'] = (df_pred['px_height'] * df_pred['px_width']) / df_pred['screen_area']
    df_pred['camera_quality'] = df_pred['fc'] + df_pred['pc']
    df_pred = df_pred[feature_names]

    # VORHERSAGE
    probs = model.predict_proba(df_pred)[0]
    pred_class = np.argmax(probs)
    votes = get_individual_votes(model, df_pred)

# --- RECHTE SPALTE: VISUALISIERUNG ---
with right_col:
    # 1. Ergebnis Banner
    st.subheader("📊 Analyse-Ergebnis")
    
    res_color = COLOR_MAP[pred_class]
    st.markdown(f"""
    <div style="background-color: {res_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin:0;">{CLASSES[pred_class]}</h2>
        <p style="margin:0;">Modell-Sicherheit: {probs[pred_class]:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Abstand

    # 2. Tabs für Details
    tab1, tab2, tab3 = st.tabs(["🧩 Das KI-Mosaik", "🕸️ Fingerprint", "📈 Wahrscheinlichkeiten"])

    with tab1:
        st.markdown("**Blick in das 'Gehirn' des Random Forest:**")
        st.caption("Jedes Quadrat ist einer von 100 Entscheidungsbäumen. Wenn sich die Farben mischen, ist das Modell unsicher (Konflikt).")
        
        # Mosaik bauen (10x10 Grid)
        grid = np.array(votes).reshape(10, 10)
        
        fig_mosaic = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale=HEATMAP_COLORS,
            zmin=0, zmax=3,
            showscale=False,
            xgap=1, ygap=1 # Gitterlinien
        ))
        fig_mosaic.update_layout(
            width=350, height=350,
            xaxis=dict(showticklabels=False, fixedrange=True),
            yaxis=dict(showticklabels=False, fixedrange=True, autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_mosaic, use_container_width=True)

    with tab2:
        # Radar Chart
        r_norm = (ram - ranges['ram'][0]) / (ranges['ram'][1] - ranges['ram'][0])
        b_norm = (battery - ranges['battery_power'][0]) / (ranges['battery_power'][1] - ranges['battery_power'][0])
        p_norm = (px_h - ranges['px_height'][0]) / (ranges['px_height'][1] - ranges['px_height'][0])
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[r_norm, b_norm, p_norm, r_norm],
            theta=['RAM (Leistung)', 'Battery (Ausdauer)', 'Display (Schärfe)', 'RAM'],
            fill='toself',
            line_color=res_color
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=300,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        # Balkendiagramm
        probs_df = pd.DataFrame({
            "Klasse": ["Low", "Medium", "High", "Very High"],
            "Wahrscheinlichkeit": probs,
            "Farbe": [COLOR_MAP[0], COLOR_MAP[1], COLOR_MAP[2], COLOR_MAP[3]]
        })
        
        fig_bar = px.bar(probs_df, x="Klasse", y="Wahrscheinlichkeit", 
                         color="Klasse", color_discrete_map={
                             "Low": COLOR_MAP[0], "Medium": COLOR_MAP[1], 
                             "High": COLOR_MAP[2], "Very High": COLOR_MAP[3]
                         })
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
