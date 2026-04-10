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
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

# Style CSS pour coller à la photo (Cards & Sidebar grise)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.02); }
    .card { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- PARAMÈTRES STADES & STRESS (Sources Arvalis/Bayer) ---
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- COLONNE DE GAUCHE (CONFIGURATION) ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip/.shp)", type=["zip"])
        st.button("🔍 ANALYSER LA PARCELLE", use_container_width=True)
    
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    with st.expander("🚜 CHANTIER & APPLICATION", expanded=True):
        v_vent = st.slider("Vitesse Vent (km/h)", 0, 40, 12)
        hygro = st.slider("Hygrométrie (%)", 0, 100, 75)
        produit = st.text_input("Produit & Dose", "Expert-Grow 2L/ha")
        # Diagnostic immédiat du chantier
        if v_vent > 19: st.error("⚠️ Vent trop fort (Dérive)")
        elif hygro < 60: st.warning("⚠️ Hygro faible (Évaporation)")
        else: st.success("✅ Conditions Optimales")
    
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)
        mode_debug = st.checkbox("🛠️ Mode Diagnostic")

# --- ANALYSE & RÉSULTATS (PARTIE DROITE) ---
if uploaded_file:
    try:
        # Lecture des 449 points de ton fichier binaire [cite: 1-49, 13]
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Identification bande (ton '1' vs '0')
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        if mode_debug:
            st.write("Colonnes détectées :", list(df.columns))

        # --- 1. BANDEAU DE KPIs (HAUT) ---
        m = df.groupby('grp')['rdt'].mean()
        gain = m.get('Produit', 0) - m.get('Témoin', 0)
        marge = ((gain/10) * prix_vente) - cout_prod
        _, p_val = stats.ttest_ind(df[df['grp']=='Produit']['rdt'], df[df['grp']=='Témoin']['rdt'])

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("GAIN MOYEN RDT", f"+{round(gain, 2)} qtx/ha")
        kpi2.metric("FIABILITÉ STATISTIQUE", f"{round((1-p_val)*100, 1)}%", delta="SIGNIFICATIF" if p_val < 0.05 else "NON SIGNIF.")
        kpi3.metric("MARGE NETTE", f"+{round(marge, 2)} €/ha", delta="RECOMMANDÉ" if marge > 0 else "NON RENTABLE")

        st.divider()

        # --- 2. GRAPHIQUES CENTRAUX ---
        col_rdt, col_sol = st.columns([2, 1])
        
        with col_rdt:
            st.markdown("### 📍 RÉPONSE RDT vs POTENTIEL SOL")
            fig_rdt = px.box(df, x="potentiel", y="rdt", color="grp", notched=True,
                             color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with col_sol:
            st.markdown("### 🧪 CORRÉLATION SOL")
            c_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            if len(c_cols) > 1:
                st.plotly_chart(px.imshow(df[c_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

        # --- 3. TIMELINE CLIMAT & STRESS ---
        st.markdown("### 📅 TIMELINE CLIMAT, STRESS & STADES PHYSIOLOGIQUES")
        lat, lon = gdf.to_crs(epsg=4326).geometry.y.mean(), gdf.to_crs(epsg=4326).geometry.x.mean()
        
        # Appel API Météo (Temp, Pluie, Rayonnement)
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={datetime.now().date()}&daily=temperature_2m_max,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
        w_data = requests.get(url).json()['daily']
        w_df = pd.DataFrame(w_data)
        w_df['time'] = pd.to_datetime(w_df['time'])
        w_df['cum_t'] = w_df['temperature_2m_max'].cumsum() # Somme de T° simple

        fig_t = go.Figure()
        # Rayonnement (Radiation) en fond
        fig_t.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], name="Rayonnement", fill='tozeroy', line_color='rgba(255,200,0,0.2)'))
        # Précipitations
        fig_t.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Précipitations", marker_color='blue'))
        # Temp Max
        fig_t.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="Temp Max", line_color='red'))

        # Marquer les Stades selon Arvalis
        p_c = PARAM_CULTURES[culture]
        for label, deg in {"Levée": 150, "Floraison": p_c['flo'], "Récolte": p_c['rec']}.items():
            stade_date = w_df[w_df['cum_t'] >= deg]['time'].iloc[0] if not w_df[w_df['cum_t'] >= deg].empty else None
            if stade_date:
                fig_t.add_vline(x=stade_date, line_dash="dot", annotation_text=label)
        
        # Zone de Stress Thermique (Hachurée)
        stress_dates = w_df[w_df['temperature_2m_max'] > p_c['max']]
        if not stress_dates.empty:
            fig_t.add_vrect(x0=stress_dates['time'].min(), x1=stress_dates['time'].max(), fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="STRESS THERMIQUE")

        st.plotly_chart(fig_t, use_container_width=True)

        # --- 4. SYNTHÈSE & PRÉCONISATION ---
        st.markdown("### 💬 PRÉCONISATION & SYNTHÈSE AGRONOMIQUE")
        c_diag, c_btn = st.columns([3, 1])
        with c_diag:
            diag = f"**Analyse automatique :** L'essai sur **{culture}** montre un gain de **{round(gain,1)} qtx/ha**. "
            if p_val < 0.05: diag += "Ce résultat est **statistiquement prouvé**. "
            if v_vent > 19: diag += "Attention : les conditions de vent lors du chantier étaient défavorables. "
            if not stress_dates.empty: diag += f"Une zone de stress thermique (> {p_c['max']}°C) a été détectée durant le cycle, justifiant l'intérêt du biostimulant."
            st.info(diag)
            
        with c_btn:
            st.button("📄 IMPRIMER RAPPORT PDF", use_container_width=True)
            st.button("📊 GÉNÉRER SCÉNARIOS", use_container_width=True)

    except Exception as e:
        st.error(f"Erreur d'analyse : {e}")
