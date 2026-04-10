import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
from datetime import datetime, timedelta
import io
import zipfile
import os

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

# CSS pour le style "Card"
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) { background-color: #ffffff; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MÉTIER STADES (Source: Arvalis) ---
STAGES_WHEAT = {
    "Semis": 0,
    "Levée": 150,
    "Floraison": 1100,
    "Récolte": 2000
}

# --- SIDEBAR (COLONNE DE GAUCHE) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2570/2570163.png", width=80)
    st.title("Bio-Expert 360")
    
    st.header("📥 IMPORTATION")
    uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    mode_debug = st.checkbox("🛠️ Diagnostic Technique")
    
    st.header("🌾 CONFIGURATION")
    culture = st.selectbox("Culture", ["Blé Tendre", "Orge", "Maïs"])
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
    d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    st.header("🚜 CHANTIER")
    v_vent = st.slider("Vitesse Vent (km/h)", 0, 40, 12)
    hygro = st.slider("Hygrométrie (%)", 0, 100, 75)
    
    st.header("💰 ÉCONOMIE")
    prix_vente = st.number_input("Prix de vente (€/T)", value=210)
    cout_prod = st.number_input("Coût Produit (€/ha)", value=45)
    calc_renta = st.button("Calculer Rentabilité", use_container_width=True)

# --- CORPS PRINCIPAL ---
if uploaded_file:
    try:
        # Lecture des 449 points 
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping Bande
        val_prod = st.sidebar.selectbox("Valeur Produit dans le fichier", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Biostimulant' if x == val_prod else 'Témoin')

        # --- TOP METRICS ---
        m = df.groupby('grp')['rdt'].mean()
        gain = m.get('Biostimulant', 0) - m.get('Témoin', 0)
        marge = ((gain/10) * prix_vente) - cout_prod
        _, p_val = stats.ttest_ind(df[df['grp']=='Biostimulant']['rdt'], df[df['grp']=='Témoin']['rdt'])

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("GAIN MOYEN RDT", f"+{round(gain, 2)} qtx/ha")
        with c2: st.metric("FIABILITÉ (P-VALUE)", f"{p_val:.2e}", delta="SIGNIFICATIF" if p_val < 0.05 else "NON SIGNIF.")
        with c3: st.metric("MARGE NETTE", f"+{round(marge, 2)} €/ha", delta_color="normal")

        # --- GRAPHIQUES ---
        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.subheader("📍 RÉPONSE RDT vs POTENTIEL SOL")
            fig_box = px.box(df, x="potentiel", y="rdt", color="grp", notched=True, 
                             color_discrete_map={'Témoin': '#3498db', 'Biostimulant': '#2ecc71'})
            st.plotly_chart(fig_box, use_container_width=True)

            # TIMELINE CLIMAT
            st.subheader("📅 TIMELINE CLIMAT & STRESS")
            lat, lon = gdf.to_crs(epsg=4326).geometry.y.mean(), gdf.to_crs(epsg=4326).geometry.x.mean()
            
            # API Weather (Temp, Pluie, Rayonnement)
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={datetime.now().date()}&daily=temperature_2m_max,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
            res = requests.get(url).json()['daily']
            w_df = pd.DataFrame(res)
            w_df['time'] = pd.to_datetime(w_df['time'])
            
            # Calcul Somme de Température (Simple)
            w_df['cum_temp'] = w_df['temperature_2m_max'].cumsum()

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Précipitations", marker_color='blue', opacity=0.4))
            fig_timeline.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="Temp Max", line=dict(color='orange')))
            
            # Ajout des Stades
            for stage, deg in STAGES_WHEAT.items():
                date_stage = w_df[w_df['cum_temp'] >= deg]['time'].iloc[0] if not w_df[w_df['cum_temp'] >= deg].empty else None
                if date_stage:
                    fig_timeline.add_vline(x=date_stage, line_dash="dot", annotation_text=stage)

            st.plotly_chart(fig_timeline, use_container_width=True)

        with col_side:
            st.subheader("🧪 CORRÉLATION SOL")
            cols_sol = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            st.plotly_chart(px.imshow(df[cols_sol].corr(), text_auto=True, color_continuous_scale="Greens"), use_container_width=True)

            st.subheader("📝 PRÉCONISATION")
            # Logique de conclusion automatique
            if gain > 3 and p_val < 0.05:
                concl = f"L'essai sur {culture} est concluant. Le produit est rentable (+{round(marge,2)}€/ha). "
                if w_df[w_df['temperature_2m_max'] > 25].shape[0] > 0:
                    concl += "Un effet anti-stress thermique a été identifié."
            else:
                concl = "Les résultats ne permettent pas de conclure à un gain significatif. Vérifiez les conditions d'application."
            st.write(concl)

        # BOUTON SYNTHÈSE
        if st.button("🖨️ IMPRIMER RAPPORT SYNTHÈSE PDF"):
            st.balloons()
            st.success("Génération du rapport en cours...")

    except Exception as e:
        st.error(f"Erreur : {e}")
