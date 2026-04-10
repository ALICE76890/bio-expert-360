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
import numpy as np

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Bio-Expert 360 Pro", layout="wide", page_icon="🌱")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    with st.expander("🚜 CHANTIER & APPLICATION", expanded=True):
        v_vent = st.slider("Vitesse Vent (km/h)", 0, 40, 12)
        hygro = st.slider("Hygrométrie (%)", 0, 100, 75)
        if v_vent > 19: st.error("⚠️ Vent trop fort")
        elif hygro < 60: st.warning("⚠️ Hygro faible")
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT ---
if uploaded_file:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # --- CALCULS STATS DE BASE ---
        data_p = df[df['grp'] == 'Produit']['rdt'].dropna()
        data_t = df[df['grp'] == 'Témoin']['rdt'].dropna()
        
        m_p, m_t = data_p.mean(), data_t.mean()
        gain = m_p - m_t
        marge = ((gain/10) * prix_vente) - cout_prod
        t_stat, p_val = stats.ttest_ind(data_p, data_t)

        # --- BANDEAU KPIs ---
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN MOYEN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_val:.2e}")
        k3.metric("MARGE NETTE", f"{round(marge, 2)} €/ha")

        # --- LES ONGLETS ---
        tab_rdt, tab_climat, tab_stats, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat", "🔬 Preuve Statistique", "🧪 Éléments du Sol"])

        with tab_rdt:
            st.subheader("📍 Réponse RDT vs Potentiel Sol")
            fig_rdt = px.box(df, x="potentiel", y="rdt", color="grp", notched=True, 
                             color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.subheader("📅 Timeline Climatique")
            lat, lon = gdf.to_crs(epsg=4326).geometry.y.mean(), gdf.to_crs(epsg=4326).geometry.x.mean()
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={datetime.now().date() - timedelta(days=2)}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FBerlin"
            r = requests.get(url).json()
            w_df = pd.DataFrame(r['daily'])
            w_df['time'] = pd.to_datetime(w_df['time'])
            
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='blue', yaxis="y2"))
            fig_t.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
            fig_t.update_layout(yaxis2=dict(overlaying="y", side="right"))
            st.plotly_chart(fig_t, use_container_width=True)

        with tab_stats:
            st.header("🔬 Rigueur Scientifique du Test")
            
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Distribution des rendements")
                fig_hist = px.histogram(df, x="rdt", color="grp", barmode="overlay", marginal="box",
                                        title="Répartition des 449 points de mesure",
                                        color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
                st.plotly_chart(fig_hist, use_container_width=True)

            with c_right:
                st.subheader("Indicateurs de robustesse")
                # Test de normalité (Shapiro)
                _, p_norm_p = stats.shapiro(data_p)
                _, p_norm_t = stats.shapiro(data_t)
                
                st.write(f"**Nombre de points (N) :** {len(df)}")
                st.write(f"**Normalité (Produit) :** {'✅ OK' if p_norm_p > 0.05 else '⚠️ Distribution asymétrique'}")
                st.write(f"**Normalité (Témoin) :** {'✅ OK' if p_norm_t > 0.05 else '⚠️ Distribution asymétrique'}")
                
                # Intervalle de Confiance à 95%
                sem = stats.sem(data_p) - stats.sem(data_t)
                ic = sem * stats.t.ppf((1 + 0.95) / 2., len(data_p) + len(data_t) - 2)
                
                st.info(f"**Intervalle de confiance (95%) :** Le gain réel se situe entre **{round(gain-ic, 2)}** et **{round(gain+ic, 2)}** qtx/ha.")

        with tab_sol:
            st.subheader("🧪 Corrélation Éléments du Sol")
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            st.plotly_chart(px.imshow(df[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")
