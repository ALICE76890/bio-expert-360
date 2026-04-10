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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

# Paramètres agronomiques par défaut
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- 2. BARRE LATÉRALE (TON MENU) ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    
    with st.expander("📥 IMPORTATION", expanded=True):
        uploaded_file = st.file_uploader("Fichier ZIP (449 pts)", type=["zip"])
    
    with st.expander("🌾 CONFIGURATION", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    with st.expander("🧹 NETTOYAGE", expanded=False):
        clean_data = st.checkbox("Filtre Outliers (IQR)", value=True)
    
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix Vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- 3. LOGIQUE PRINCIPALE ---
if uploaded_file:
    try:
        # Lecture et extraction
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping des groupes
        val_prod = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_prod else 'Témoin')
        
        # Coordonnées moyennes
        lat_m = gdf.to_crs(epsg=4326).geometry.y.mean()
        lon_m = gdf.to_crs(epsg=4326).geometry.x.mean()

        # Nettoyage IQR
        if clean_data:
            q1, q3 = df['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df['rdt'] >= q1 - 1.5*iqr) & (df['rdt'] <= q3 + 1.5*iqr)]

        # --- CALCULS ---
        data_p = df[df['grp'] == 'Produit']['rdt'].dropna()
        data_t = df[df['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()
        _, p_val = stats.ttest_ind(data_p, data_t)

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        c2.metric("FIABILITÉ", f"{p_val:.2e}")
        c3.metric("COORDONNÉES", f"{round(lat_m, 3)}, {round(lon_m, 3)}")

        # --- ONGLETS ---
        t_rdt, t_meteo, t_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Rayonnement", "🔬 Preuve Stat"])

        with t_rdt:
            st.plotly_chart(px.box(df, x="potentiel", y="rdt", color="grp", notched=True), use_container_width=True)

        with t_meteo:
            st.subheader(f"Données Météo (Station Virtuelle)")
            # Fix date pour l'API (max J-2)
            end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_m}&longitude={lon_m}&start_date={d_semis}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
            
            try:
                r = requests.get(url).json()
                w_df = pd.DataFrame(r['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])

                fig = go.Figure()
                # Rayonnement
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], fill='tozeroy', name="Rayonnement", line_color='rgba(255,215,0,0.2)'))
                # Pluie
                fig.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='blue', opacity=0.5))
                # Températures
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
                
                fig.add_vline(x=pd.Timestamp(d_appli), line_color="green", line_dash="dash", annotation_text="Appli")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Données météo momentanément indisponibles.")

        with t_stats:
            st.subheader("Distribution & Robustesse")
            st.plotly_chart(px.histogram(df, x="rdt", color="grp", barmode="overlay"), use_container_width=True)
            st.write(f"Nombre de points analysés : **{len(df)}**")

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse : {e}")
