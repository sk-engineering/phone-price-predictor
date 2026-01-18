import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Smartphone Price Predictor", layout="wide", page_icon="📱")

CLASSES = {
    0: "Niedrig (Low Cost)",
    1: "Mittel (Medium Cost)",
    2: "Hoch (High Cost)",
    3: "Sehr Hoch (Very High Cost)"
}

# Farben (Grün, Blau, Orange, Rot)
COLOR_MAP = {
    0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"
}
# Heatmap Farben
HEATMAP_COLORS = [
    [0.0, "#2ecc71"], [0.25, "#2ecc71"],
    [0.25, "#3498db"], [0.5, "#3498db"],
    [0.5, "#f39c12"], [0.75, "#f39c12"],
    [0.75, "#e74c3c"], [1.0, "#e74c3c"]
]

# ---------------------------------------------------------
# BACKEND
# ---------------------------------------------------------
@st.cache_resource
def load_model():
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
    
    # Ranges für ALLE Slider berechnen (für Normalisierung 0-1)
    ranges = {}
    for col in defaults.index:
        if col in df.columns:
            ranges[col] = (df[col].min(), df[col].max())
            
    # Spezielle Ranges für PC (Kamera) und Int_Memory, falls nötig
    ranges['pc'] = (df['pc'].min(), df['pc'].max())
    ranges['int_memory'] = (df['int_memory'].min(), df['int_memory'].max())

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, defaults, feature_names, ranges

model, defaults, feature_names, ranges = load_model()

def get_individual_votes(model, input_df):
    votes = []
    for tree in model.estimators_:
        vote = tree.predict(input_df.to_numpy())[0]
        votes.append(int(vote))
    return votes

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("📱 Smartphone Price Predictor")
st.markdown("Semesterarbeit DSBE")

if model is None:
    st.error("⚠️ 'train.csv' fehlt!")
    st.stop()

left_col, right_col = st.columns([1, 1.5])

# --- INPUTS ---
with left_col:
    st.subheader("⚙️ Konfiguration")
    
    ram = st.slider("Arbeitsspeicher (RAM)", 256, 4000, 2000, format="%d MB")
    battery = st.slider("Batteriekapazität", 500, 4000, 3000, format="%d mAh")
    px_w = st.slider("Auflösung Breite", 500, 2000, 1080, format="%d px")
    px_h = st.slider("Auflösung Höhe", 500, 2000, 1920, format="%d px")
    
    # Diese zwei waren vorher da, aber nicht im Chart. Jetzt nutzen wir sie!
    int_mem = st.slider("Interner Speicher", 2, 64, 32, format="%d GB") 
    pc = st.slider("Kamerauflösung (Main)", 0, 20, 10, format="%d MP")

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

    probs = model.predict_proba(df_pred)[0]
    pred_class = np.argmax(probs)
    votes = get_individual_votes(model, df_pred)

# --- VISUALISIERUNG ---
with right_col:
    st.subheader("📊 Analyse-Ergebnis")
    
    res_color = COLOR_MAP[pred_class]
    st.markdown(f"""
    <div style="background-color: {res_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin:0;">{CLASSES[pred_class]}</h2>
        <p style="margin:0;">Modell-Sicherheit: {probs[pred_class]:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3 = st.tabs(["🕸️ Fingerprint", "🧩 KI-Mosaik", "📈 Wahrscheinlichkeiten"])

    with tab1:
        # RADAR CHART (5-ACHSEN)
        
        # 1. Normalisierung (Werte auf 0.0 bis 1.0 bringen)
        # RAM
        val_ram = (ram - ranges['ram'][0]) / (ranges['ram'][1] - ranges['ram'][0])
        # Battery
        val_bat = (battery - ranges['battery_power'][0]) / (ranges['battery_power'][1] - ranges['battery_power'][0])
        # Display (Nutzt Höhe als Proxy für Qualität)
        val_disp = (px_h - ranges['px_height'][0]) / (ranges['px_height'][1] - ranges['px_height'][0])
        # Kamera
        val_cam = (pc - ranges['pc'][0]) / (ranges['pc'][1] - ranges['pc'][0])
        # Speicher
        val_mem = (int_mem - ranges['int_memory'][0]) / (ranges['int_memory'][1] - ranges['int_memory'][0])
        
        # 2. Daten für Plotly (Wir schließen den Kreis, indem wir RAM am Ende wiederholen)
        r_values = [val_ram, val_bat, val_mem, val_cam, val_disp, val_ram]
        theta_labels = ['<b>Leistung</b><br>(RAM)', '<b>Energie</b><br>(Akku)', '<b>Speicher</b><br>(GB)', '<b>Kamera</b><br>(MP)', '<b>Display</b><br>(Pixel)', '<b>Leistung</b><br>(RAM)']
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=r_values,
            theta=theta_labels,
            fill='toself',
            line_color=res_color,
            marker=dict(size=5),
            name="Spezifikation"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), # Achsen ohne Zahlen, cleaner
                angularaxis=dict(direction="clockwise") # Uhrzeigersinn liest sich besser
            ),
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Das Diagramm zeigt die Balance der Hardware-Komponenten. Je grösser die Fläche, desto besser die Ausstattung.")

    with tab2:
            st.markdown("### Das Gehirn des Modells: 2D vs. 3D")
            st.caption("Vergleich der 100 Entscheidungsbäume. Links die flache Draufsicht, rechts die Topografie der Entscheidung.")
    
            # Grid Daten vorbereiten
            grid = np.array(votes).reshape(10, 10)
    
            # Layout in zwei Spalten teilen
            col_2d, col_3d = st.columns([1, 1]) # 50/50 Aufteilung
    
            # --- LINKE SPALTE: 2D HEATMAP ---
            with col_2d:
                st.markdown("#### 🟦 2D Mosaik")
                fig_2d = go.Figure(data=go.Heatmap(
                    z=grid, 
                    colorscale=HEATMAP_COLORS, 
                    zmin=0, zmax=3, 
                    showscale=False, 
                    xgap=2, ygap=2 # Lücken für Kacheleffekt
                ))
                fig_2d.update_layout(
                    height=350,
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False, autorange="reversed"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_2d, use_container_width=True)
    
            # --- RECHTE SPALTE: 3D SURFACE ---
            with col_3d:
                st.markdown("#### 🏔️ 3D Landschaft")
                
                # Frames für Animation berechnen
                frames = []
                for t in range(0, 360, 5):
                    rad = np.radians(t)
                    x_eye = 1.6 * np.cos(rad)
                    y_eye = 1.6 * np.sin(rad)
                    frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=x_eye, y=y_eye, z=0.6))))))
    
                fig_3d = go.Figure(
                    data=[go.Surface(
                        z=grid,
                        colorscale=HEATMAP_COLORS,
                        cmin=0, cmax=3,
                        opacity=0.9,
                        contours_z=dict(show=True, usecolormap=False, highlightcolor="white", project_z=True)
                    )],
                    layout=go.Layout(
                        height=350,
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(title="", range=[0, 3.5], tickvals=[0,1,2,3], ticktext=["L", "M", "H", "VH"]),
                            aspectmode='manual',
                            aspectratio=dict(x=1, y=1, z=0.7)
                        ),
                        updatemenus=[dict(
                            type='buttons',
                            showactive=False,
                            y=0.05, x=0.05, xanchor='left', yanchor='bottom',
                            pad=dict(t=0, r=10),
                            buttons=[dict(
                                label="🔄", # Kleinerer Button mit Icon spart Platz
                                method="animate",
                                args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]
                            )]
                        )]
                    ),
                    frames=frames
                )
                st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        probs_df = pd.DataFrame({
            "Klasse": ["Low", "Medium", "High", "Very High"],
            "Wahrscheinlichkeit": probs
        })
        fig_bar = px.bar(probs_df, x="Klasse", y="Wahrscheinlichkeit", 
                         color="Klasse", color_discrete_map={
                             "Low": COLOR_MAP[0], "Medium": COLOR_MAP[1], 
                             "High": COLOR_MAP[2], "Very High": COLOR_MAP[3]
                         })
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
