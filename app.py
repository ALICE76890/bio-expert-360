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

# --- STYLE & CONFIG ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")
st.markdown("<style>.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }</style>", unsafe_allow_html=True)

PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION", expanded=True):
        uploaded_file = st.file_uploader("Fichier ZIP", type=["zip"])
    with st.expander("🌾 CONFIGURATION", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    with st.expander("🚜 CHANTIER", expanded=True):
        v_vent = st.slider("Vitesse Vent (km/h)", 0, 40, 12)
        hygro = st.slider("Hygrométrie (%)", 0, 100, 75)
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix Vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- ANALYSE ---
if uploaded_file:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        val_p = st.sidebar.selectbox("Valeur Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Calculs rapides
        m = df.groupby('grp')['rdt'].mean()
        gain = m.get('Produit', 0) - m.get('Témoin', 0)
        lat_m, lon_m = gdf.geometry.y.mean(), gdf.geometry.x.mean()

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("GAIN MOYEN", f"+{round(gain, 2)} qtx/ha")
        c2.metric("COORDONNÉES", f"{round(lat_m, 4)}, {round(lon_m, 4)}")
        c3.metric("STATUT", "Analyse Terminée ✅")

        tabs = st.tabs(["📊 Rendement", "🌦️ Climat & Rayonnement", "🔬 Stats"])

        with tabs[1]:
            st.subheader("🌦️ Station Virtuelle & Timeline")
            
            # Utilisation de l'API Forecast (plus fiable pour les dates récentes)
            # On demande les 90 derniers jours pour couvrir l'application
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_m}&longitude={lon_m}&past_days=90&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=auto"
            
            res = requests.get(url).json()
            if "daily" in res:
                w_df = pd.DataFrame(res['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                
                fig = go.Figure()
                # Rayonnement (MJ/m²)
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], fill='tozeroy', name="Rayonnement", line_color='rgba(255, 215, 0, 0.2)', yaxis="y3"))
                # Pluie
                fig.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='blue', opacity=0.4, yaxis="y2"))
                # Températures
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                fig.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
                
                # Axe Application
                fig.add_vline(x=pd.Timestamp(d_appli), line_width=3, line_dash="dash", line_color="green", annotation_text="APPLI")

                fig.update_layout(
                    height=500,
                    yaxis=dict(title="Température (°C)"),
                    yaxis2=dict(title="Pluie (mm)", overlaying="y", side="right", showgrid=False),
                    yaxis3=dict(title="Rayonnement", overlaying="y", side="right", anchor="free", position=1, showgrid=False),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Données extraites le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
            else:
                st.error("Erreur API Météo : Impossible de récupérer les données.")

        with tabs[0]:
            st.plotly_chart(px.box(df, x="potentiel", y="rdt", color="grp", notched=True), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur technique : {e}")
