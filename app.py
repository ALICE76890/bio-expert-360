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

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Bio-Expert 360 Pro", layout="wide", page_icon="🌱")

# --- PARAMÈTRES ARVALIS ---
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- SIDEBAR (COLONNE GAUCHE) ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    with st.expander("🧹 NETTOYAGE DATA", expanded=True):
        clean_outliers = st.checkbox("Filtre anti-aberrants (IQR)", value=True)
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    with st.expander("🚜 CHANTIER", expanded=True):
        v_vent = st.slider("Vitesse Vent (km/h)", 0, 40, 12)
        hygro = st.slider("Hygrométrie (%)", 0, 100, 75)
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- ANALYSE ---
if uploaded_file:
    try:
        # 1. Chargement & Nettoyage
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping Groupe
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Calcul Stats
        data_p = df[df['grp']=='Produit']['rdt'].dropna()
        data_t = df[df['grp']=='Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()
        _, p_val = stats.ttest_ind(data_p, data_t)

        # 2. Affichage KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("P-VALUE", f"{p_val:.2e}")
        k3.metric("COORDONNÉES CHAMP", f"{round(gdf.to_crs(epsg=4326).geometry.y.mean(),4)}, {round(gdf.to_crs(epsg=4326).geometry.x.mean(),4)}")

        # 3. ONGLET MÉTÉO DÉTAILLÉ
        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Rayonnement", "🔬 Preuve Statistique"])

        with tab_climat:
            lat = gdf.to_crs(epsg=4326).geometry.y.mean()
            lon = gdf.to_crs(epsg=4326).geometry.x.mean()
            
            # Appel API avec toutes les variables demandées
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={datetime.now().date() - timedelta(days=2)}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
            r = requests.get(url).json()
            w_df = pd.DataFrame(r['daily'])
            w_df['time'] = pd.to_datetime(w_df['time'])

            st.subheader(f"📍 Station Virtuelle : Lat {round(lat,3)} / Lon {round(lon,3)}")
            st.caption(f"Dernière mise à jour des données : {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

            # Graphique Complexe Plotly
            fig = go.Figure()

            # 1. Rayonnement Solaire (Fond)
            fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], 
                                     fill='tozeroy', name="Rayonnement (MJ/m²)", 
                                     line_color='rgba(255, 215, 0, 0.3)', yaxis="y3"))

            # 2. Précipitations (Barres)
            fig.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], 
                                 name="Pluie (mm)", marker_color='rgba(0, 0, 255, 0.4)', yaxis="y2"))

            # 3. Températures Max et Min
            fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], 
                                     name="T° Max", line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], 
                                     name="T° Min", line=dict(color='blue', width=1, dash='dot')))

            # Mise en page multi-axes
            fig.update_layout(
                height=600,
                xaxis=dict(title="Cycle de culture"),
                yaxis=dict(title="Température (°C)", side="left"),
                yaxis2=dict(title="Pluie (mm)", overlaying="y", side="right", showgrid=False),
                yaxis3=dict(title="Rayonnement", overlaying="y", side="right", anchor="free", position=1, showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Marquage Application
            fig.add_vline(x=pd.Timestamp(d_appli), line_width=3, line_dash="dash", line_color="green", annotation_text="APPLI")

            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Analyse de l'amplitude :** L'écart moyen Tmax/Tmin sur la période est de {round((w_df['temperature_2m_max'] - w_df['temperature_2m_min']).mean(), 1)}°C. "
                    f"Le rayonnement cumulé depuis le semis est de {round(w_df['shortwave_radiation_sum'].sum() / 1000, 2)} GJ/m².")

        # Onglets RDT et Stats (Restent inchangés pour la stabilité)
        with tab_rdt:
            st.plotly_chart(px.box(df, x="potentiel", y="rdt", color="grp", notched=True), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur technique : {e}")
