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

# --- 1. CONFIGURATION VISUELLE (STYLE PHOTO) ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.02); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; padding-top: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #2ecc71; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PARAMÈTRES AGRONOMIQUES (ARVALIS) ---
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- 3. COLONNE DE GAUCHE (SIDEBAR) ---
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
        st.write(f"**Conditions :** {'✅ Optimales' if v_vent < 19 and hygro > 60 else '⚠️ Sous-optimales'}")
    
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix Vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- 4. ANALYSE ET AFFICHAGE ---
if uploaded_file:
    try:
        # Lecture
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping Groupes
        val_p = st.sidebar.selectbox("Valeur 'Bande' = Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Calculs Rapides
        data_p = df[df['grp'] == 'Produit']['rdt'].dropna()
        data_t = df[df['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()
        marge = ((gain/10) * prix_vente) - cout_prod
        _, p_val = stats.ttest_ind(data_p, data_t)

        # KPIs HAUT
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN MOYEN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_val:.2e}", delta="SIGNIFICATIF" if p_val < 0.05 else "NON SIGNIF.")
        k3.metric("MARGE NETTE", f"+{round(marge, 2)} €/ha")

        st.divider()

        # --- ONGLETS ---
        tab_rdt, tab_climat, tab_stats, tab_sol = st.tabs([
            "📊 Réponse RDT & Potentiel", 
            "🌦️ Timeline Climat & Stress", 
            "🔬 Preuve Statistique", 
            "🧪 Corrélation Sol"
        ])

        with tab_rdt:
            st.subheader("📍 Analyse du Rendement par Potentiel de Sol")
            fig_rdt = px.box(df, x="potentiel", y="rdt", color="grp", notched=True,
                             color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.subheader("📅 Timeline Climatique, Rayonnement & Stades")
            lat, lon = gdf.to_crs(epsg=4326).geometry.y.mean(), gdf.to_crs(epsg=4326).geometry.x.mean()
            st.caption(f"📍 Station Virtuelle : {round(lat,3)}, {round(lon,3)} | Mise à jour : {datetime.now().strftime('%d/%m %H:%M')}")
            
            # --- SOLUTION MÉTÉO ROBUSTE ---
            # On utilise l'API Forecast avec past_days pour éviter les plantages d'archive
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=92&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=auto"
            
            try:
                r = requests.get(url).json()['daily']
                w_df = pd.DataFrame(r)
                w_df['time'] = pd.to_datetime(w_df['time'])
                
                # On filtre pour ne garder que depuis le semis
                w_df = w_df[w_df['time'] >= pd.to_datetime(d_semis)]
                w_df['cum_t'] = w_df['temperature_2m_max'].cumsum()

                fig_climat = go.Figure()
                # Rayonnement en fond
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], name="Rayonnement", fill='tozeroy', line_color='rgba(255,215,0,0.15)', yaxis="y3"))
                # Pluie
                fig_climat.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='rgba(0,0,255,0.4)', yaxis="y2"))
                # Températures
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
                
                # Stades Arvalis basés sur somme de T°
                p_c = PARAM_CULTURES[culture]
                for stage, deg in {"Levée": 150, "Floraison": p_c['flo'], "Récolte": p_c['rec']}.items():
                    stade_date = w_df[w_df['cum_t'] >= deg]['time'].iloc[0] if not w_df[w_df['cum_t'] >= deg].empty else None
                    if stade_date:
                        fig_climat.add_vline(x=stade_date, line_dash="dot", annotation_text=stage)

                fig_climat.update_layout(
                    height=550,
                    yaxis=dict(title="Température (°C)"),
                    yaxis2=dict(title="Pluie (mm)", overlaying="y", side="right", showgrid=False),
                    yaxis3=dict(title="Rayonnement", overlaying="y", side="right", anchor="free", position=1, showgrid=False),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_climat, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur météo : {e}")

        with tab_stats:
            st.subheader("🔬 Robustesse des données")
            st.plotly_chart(px.histogram(df, x="rdt", color="grp", barmode="overlay"), use_container_width=True)
            st.write(f"Nombre de points analysés : **{len(df)}**")

        with tab_sol:
            st.subheader("🧪 Corrélation Éléments du Sol")
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            if len(sol_cols) > 1:
                st.plotly_chart(px.imshow(df[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

        # SYNTHÈSE FINALE
        st.divider()
        st.subheader("💬 PRÉCONISATION & SYNTHÈSE AGRONOMIQUE")
        st.info(f"Analyse terminée pour le **{culture}**. Le gain mesuré est de **{round(gain,2)} qtx/ha**, générant une marge de **{round(marge,2)} €/ha**. Les conditions de chantier étaient **{'optimales' if v_vent < 19 and hygro > 60 else 'sous-optimales'}**.")
        st.button("📄 IMPRIMER RAPPORT PDF")

    except Exception as e:
        st.error(f"Erreur d'analyse : {e}")
