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
import shutil

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Bio-Expert 360 Pro", layout="wide", page_icon="🌱")

# --- FONCTION DE NETTOYAGE ---
def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

# --- SEUILS DE STRESS ARVALIS ---
PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100, "base_t": 0},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700, "base_t": 6},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950, "base_t": 0}
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🧹 NETTOYAGE IQR (Tukey 1.5x)", expanded=True):
        clean_outliers = st.checkbox("Filtrer les points aberrants", value=True)

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2024, 10, 20)) # Ajusté pour l'archive
        d_appli = st.date_input("Date d'Application", datetime(2025, 3, 10))
        
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT PRINCIPAL ---
if uploaded_file:
    try:
        clear_temp()
        # 1. Lecture du ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        
        # Trouver le fichier .shp
        shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
        if not shp_files:
            st.error("Aucun fichier .shp trouvé dans le ZIP")
            st.stop()
            
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Vérification des colonnes critiques
        if 'bande' not in df.columns or 'rdt' not in df.columns:
            st.error(f"Colonnes manquantes. Colonnes détectées : {list(df.columns)}")
            st.stop()

        val_p = st.sidebar.selectbox("Quelle bande est le 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # 2. Nettoyage & Stats
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list)
        else:
            df_final = df.copy()

        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()

        # Tests Stats
        _, p_norm = stats.shapiro(data_p) if len(data_p) > 3 else (0, 0)
        t_stat, p_student = stats.ttest_ind(data_p, data_t)
        _, p_wilcoxon = stats.mannwhitneyu(data_p, data_t)

        gain = data_p.mean() - data_t.mean()

        # 3. Affichage KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-STUDENT)", f"{p_student:.4f}")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        tab_rdt, tab_climat, tab_stats_expert, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat", "🔬 Stat Expert", "🧪 Sol"])

        with tab_rdt:
            st.plotly_chart(px.box(df_final, x="grp", y="rdt", color="grp", points="all", title="Répartition du Rendement"), use_container_width=True)

        with tab_climat:
            st.subheader("🌦️ Analyse Climatique (Open-Meteo)")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            d_start = d_semis.strftime("%Y-%m-%d")
            d_end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_start}&end_date={d_end}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            r = requests.get(url).json()
            if 'daily' in r:
                w_df = pd.DataFrame(r['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                p_c = PARAM_CULTURES[culture]
                w_df['stress'] = w_df['temperature_2m_max'] >= p_c['max']
                
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)"))
                fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                st.plotly_chart(fig_w, use_container_width=True)
                st.write(f"⚠️ Jours de stress thermique (> {p_c['max']}°C) : {w_df['stress'].sum()}")

        with tab_stats_expert:
            st.header("🔬 Analyse de Robustesse")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Normalité :** {'Valide ✅' if p_norm > 0.05 else 'Hétérogène ⚠️'}")
                st.write(f"**P-Value Student :** `{p_student:.4f}`")
                st.write(f"**P-Value Wilcoxon :** `{p_wilcoxon:.4f}`")
            with col2:
                st.plotly_chart(px.histogram(df_final, x="rdt", color="grp", barmode="overlay"), use_container_width=True)

        with tab_sol:
            st.subheader("🧪 Corrélations Sol-Rendement")
            # Filtrer uniquement les colonnes numériques existantes
            num_cols = df_final.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 1:
                corr = df_final[num_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur système : {e}")
