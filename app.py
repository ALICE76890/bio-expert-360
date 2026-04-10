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

# --- PARAMÈTRES ---
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
    
    with st.expander("🧹 NETTOYAGE DES DONNÉES", expanded=True):
        clean_outliers = st.checkbox("Activer le filtre anti-aberrants", value=True)
        st.info("Méthode : Interquartile Range (IQR) 1.5x")

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
        
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT PRINCIPAL ---
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

        # --- ÉTAPE DE NETTOYAGE (OUTLIERS) ---
        n_initial = len(df)
        df_cleaned = df.copy()
        
        if clean_outliers:
            # Calcul par groupe pour ne pas écraser les vraies différences
            for g in ['Produit', 'Témoin']:
                subset = df[df['grp'] == g]['rdt']
                q1 = subset.quantile(0.25)
                q3 = subset.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                # On ne garde que ce qui est dans les bornes
                df_cleaned = df_cleaned.drop(df_cleaned[(df_cleaned['grp'] == g) & ((df_cleaned['rdt'] < lower) | (df_cleaned['rdt'] > upper))].index)
        
        n_final = len(df_cleaned)
        n_removed = n_initial - n_final

        # --- CALCULS AVEC DONNÉES PROPRES ---
        data_p = df_cleaned[df_cleaned['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_cleaned[df_cleaned['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()
        marge = ((gain/10) * prix_vente) - cout_prod
        t_stat, p_val = stats.ttest_ind(data_p, data_t)

        # --- AFFICHAGE ---
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN (NETTOYÉ)", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ", f"{p_val:.2e}")
        k3.metric("MARGE NETTE", f"{round(marge, 2)} €/ha")

        tab_rdt, tab_climat, tab_stats, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat", "🔬 Preuve & Nettoyage", "🧪 Sol"])

        with tab_stats:
            st.header("🔬 Rapport de Qualité des Données")
            
            # Rapport de nettoyage
            c1, c2, c3 = st.columns(3)
            c1.metric("Points initiaux", n_initial)
            c2.metric("Points supprimés", n_removed, delta_color="inverse")
            c3.metric("Qualité data", f"{round((n_final/n_initial)*100, 1)}%")
            
            st.info(f"**Méthode utilisée :** Filtre de Tukey (IQR 1.5). Les valeurs aberrantes sont les points situés en dehors de l'intervalle $[Q1 - 1.5 \times IQR ; Q3 + 1.5 \times IQR]$.")
            
            # Histogrammes Comparatifs
            fig_dist = px.histogram(df_cleaned, x="rdt", color="grp", marginal="rug", barmode="overlay",
                                    title="Distribution finale après nettoyage",
                                    color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_dist, use_container_width=True)
            
            with st.expander("📝 Détails mathématiques du test"):
                st.write(f"**Test de normalité (p-value) :** {round(stats.shapiro(data_p)[1], 4)}")
                st.write(f"**Écart-type Produit :** {round(data_p.std(), 2)}")
                st.write(f"**Écart-type Témoin :** {round(data_t.std(), 2)}")

        with tab_rdt:
            st.subheader("📍 Comparatif Rendement (Données filtrées)")
            st.plotly_chart(px.box(df_cleaned, x="potentiel", y="rdt", color="grp", notched=True), use_container_width=True)

        # (Les autres onglets restent identiques...)
        # ... [Code Climat et Sol ici] ...

    except Exception as e:
        st.error(f"Erreur : {e}")
