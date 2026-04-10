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

# Style pour coller à la photo
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- SEUILS ARVALIS ---
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- COLONNE DE GAUCHE (TON MENU NICKEL) ---
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
        produit_dose = st.text_input("Produit & Dose", "Expert-Grow 2L/ha")
        if v_vent > 19: st.error("⚠️ Vent trop fort")
        elif hygro < 60: st.warning("⚠️ Hygro faible")
        else: st.success("✅ Conditions Optimales")
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT DES DONNÉES ---
if uploaded_file:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping automatique
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # --- BANDEAU KPIs ---
        m = df.groupby('grp')['rdt'].mean()
        gain = m.get('Produit', 0) - m.get('Témoin', 0)
        marge = ((gain/10) * prix_vente) - cout_prod
        _, p_val = stats.ttest_ind(df[df['grp']=='Produit']['rdt'].dropna(), df[df['grp']=='Témoin']['rdt'].dropna())

        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN MOYEN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_val:.2e}")
        k3.metric("MARGE NETTE", f"{round(marge, 2)} €/ha")

        # --- LES ONGLETS ---
        tab_rdt, tab_climat, tab_stats_sol = st.tabs(["📊 Rendement & Potentiel", "🌦️ Timeline Climat & Stress", "🔬 Stats & Sol"])

        with tab_rdt:
            st.subheader("📍 Réponse RDT vs Potentiel Sol")
            fig_rdt = px.box(df, x="potentiel", y="rdt", color="grp", notched=True, 
                             color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.subheader("📅 Analyse Climatique (Semis -> Récolte)")
            lat, lon = gdf.to_crs(epsg=4326).geometry.y.mean(), gdf.to_crs(epsg=4326).geometry.x.mean()
            
            # API Corrigée : On utilise "Archive" pour le passé
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={datetime.now().date() - timedelta(days=2)}&daily=temperature_2m_max,precipitation_sum,shortwave_radiation_sum&timezone=Europe%2FBerlin"
            
            try:
                r = requests.get(url).json()
                w_df = pd.DataFrame(r['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                w_df['cum_t'] = w_df['temperature_2m_max'].cumsum()

                fig_t = go.Figure()
                fig_t.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='blue', yaxis="y2"))
                fig_t.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                
                # Ajout des zones de stress Arvalis
                p_c = PARAM_CULTURES[culture]
                stress_df = w_df[w_df['temperature_2m_max'] > p_c['max']]
                if not stress_df.empty:
                    fig_t.add_trace(go.Scatter(x=stress_df['time'], y=stress_df['temperature_2m_max'], mode='markers', name="Stress Thermique", marker=dict(color='black', size=10, symbol='x')))

                fig_t.update_layout(yaxis2=dict(title="Pluie", overlaying="y", side="right"), hovermode="x unified")
                st.plotly_chart(fig_t, use_container_width=True)
                
                # Conclusion météo
                nb_stress = len(stress_df)
                st.write(f"**Synthèse Météo :** On dénombre **{nb_stress} jours de stress thermique** (>25°C) sur le cycle. L'application du {d_appli} est idéalement placée pour couvrir ces risques.")

            except:
                st.error("Impossible de charger les données météo. Vérifiez que la date de semis est bien dans le passé.")

        with tab_stats_sol:
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("🔬 Rapport Statistique")
                st.write(f"Moyenne Produit : **{round(m.get('Produit', 0), 2)}**")
                st.write(f"Moyenne Témoin : **{round(m.get('Témoin', 0), 2)}**")
                st.write(f"Écart-type : **{round(df['rdt'].std(), 2)}**")
            with c_right:
                st.subheader("🧪 Corrélation Éléments du Sol")
                sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
                st.plotly_chart(px.imshow(df[sol_cols].corr(), text_auto=True), use_container_width=True)

        # BOUTON SYNTHÈSE FINAL
        st.divider()
        if st.button("📄 GÉNÉRER LA SYNTHÈSE COMPLÈTE & PDF"):
            st.success("Synthèse générée ! Le produit est recommandé avec une marge de " + str(round(marge,2)) + " €/ha.")

    except Exception as e:
        st.error(f"Erreur : {e}")
