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
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #2ecc71; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PARAMÈTRES AGRONOMIQUES ---
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
    
    with st.expander("🧹 NETTOYAGE & STATS", expanded=False):
        clean_outliers = st.checkbox("Filtre Outliers (IQR)", value=True)
        mode_debug = st.checkbox("Mode Diagnostic")
    
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

        # Nettoyage IQR
        n_init = len(df)
        if clean_outliers:
            q1, q3 = df['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df['rdt'] >= q1 - 1.5*iqr) & (df['rdt'] <= q3 + 1.5*iqr)]
        n_removed = n_init - len(df)

        # Calculs
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
            st.caption(f"📍 Station : {round(lat,3)}, {round(lon,3)} | Mise à jour : {datetime.now().strftime('%d/%m %H:%M')}")
            
            # API Corrigée (Fin à J-2 pour éviter les erreurs)
            end_d = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={end_d}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
            
            try:
                r = requests.get(url).json()['daily']
                w_df = pd.DataFrame(r)
                w_df['time'] = pd.to_datetime(w_df['time'])
                w_df['cum_t'] = w_df['temperature_2m_max'].cumsum()

                fig_climat = go.Figure()
                # Rayonnement en fond
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], name="Rayonnement", fill='tozeroy', line_color='rgba(255,215,0,0.15)', yaxis="y3"))
                # Pluie
                fig_climat.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='rgba(0,0,255,0.4)', yaxis="y2"))
                # Températures
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                fig_climat.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
                
                # Stades Arvalis
                p_c = PARAM_CULTURES[culture]
                for stage, deg in {"Levée": 150, "Floraison": p_c['flo'], "Récolte": p_c['rec']}.items():
                    stade_date = w_df[w_df['cum_t'] >= deg]['time'].iloc[0] if not w_df[w_df['cum_t'] >= deg].empty else None
                    if stade_date:
                        fig_climat.add_vline(x=stade_date, line_dash="dot", annotation_text=stage)

                fig_climat.update_layout(yaxis2=dict(overlaying="y", side="right"), yaxis3=dict(overlaying="y", side="right", anchor="free", position=1))
                st.plotly_chart(fig_climat, use_container_width=True)
            except:
                st.error("Données météo indisponibles pour ces dates.")

        with tab_stats:
            st.subheader("🔬 Robustesse des données")
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.plotly_chart(px.histogram(df, x="rdt", color="grp", barmode="overlay", title="Distribution après nettoyage"), use_container_width=True)
            with c_s2:
                st.write(f"**Analyse de qualité :**")
                st.write(f"- Points initiaux : {n_init}")
                st.write(f"- Points aberrants supprimés : {n_removed}")
                st.write(f"- Intervalle de confiance (95%) : +/- {round(stats.sem(data_p)*1.96, 2)} qtx")
                if p_val < 0.05: st.success("Essai Validé Scientifiquement")

        with tab_sol:
            st.subheader("🧪 Corrélation Éléments du Sol")
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            if len(sol_cols) > 1:
                st.plotly_chart(px.imshow(df[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

        # SYNTHÈSE FINALE
        st.divider()
        st.subheader("💬 PRÉCONISATION & SYNTHÈSE AGRONOMIQUE")
        st.info(f"L'essai sur **{culture}** est concluant avec un gain de **{round(gain,2)} qtx/ha**. "
                f"La marge nette générée est de **{round(marge,2)} €/ha**. Les conditions de chantier étaient **{'optimales' if v_vent < 19 else 'dégradées (vent)'}**.")
        st.button("📄 IMPRIMER RAPPORT PDF")

    except Exception as e:
        st.error(f"Erreur : {e}")
