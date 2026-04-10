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
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
        
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT PRINCIPAL ---
if uploaded_file:
    try:
        # 1. Lecture
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # 2. Nettoyage & Stats Robustes
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                iqr = q3 - q1
                sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                clean_list.append(sub)
            df_final = pd.concat(clean_list)
        else:
            df_final = df.copy()

        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()

        # --- TESTS STATISTIQUES ---
        # Test A : Normalité (Shapiro)
        _, p_norm = stats.shapiro(data_p)
        
        # Test B : Student (Paramétrique)
        t_stat, p_student = stats.ttest_ind(data_p, data_t)
        
        # Test C : Wilcoxon (Non-paramétrique, plus robuste si données bruitées)
        _, p_wilcoxon = stats.mannwhitneyu(data_p, data_t)

        gain = data_p.mean() - data_t.mean()

        # 3. Affichage KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-STUDENT)", f"{p_student:.2e}")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        tab_rdt, tab_climat, tab_stats_expert, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat & Stress Arvalis", "🔬 Preuve Statistique Expert", "🧪 Sol"])

        with tab_rdt:
            st.plotly_chart(px.box(df_final, x="potentiel", y="rdt", color="grp", notched=True), use_container_width=True)

        with tab_climat:
            st.subheader("🌦️ Analyse Climatique Multi-Annuelle (2025-2026)")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            
            # API Archive pour gérer 2025 et début 2026
            d_start = d_semis.strftime("%Y-%m-%d")
            d_end = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
            
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_start}&end_date={d_end}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=auto"
            
            try:
                r = requests.get(url).json()
                w_df = pd.DataFrame(r['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                
                # Calcul Stress Arvalis
                p_c = PARAM_CULTURES[culture]
                w_df['stress_thermique'] = w_df['temperature_2m_max'] >= p_c['max']
                
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='blue', opacity=0.3))
                fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                
                # Zone de Stress (Rouge)
                stress_only = w_df[w_df['stress_thermique']]
                fig_w.add_trace(go.Scatter(x=stress_only['time'], y=stress_only['temperature_2m_max'], mode='markers', name="Alerte Stress Arvalis", marker=dict(color='orange', size=8)))

                fig_w.add_vline(x=pd.to_datetime(d_appli).timestamp() * 1000, line_dash="dash", line_color="green", annotation_text="Appli")
                st.plotly_chart(fig_w, use_container_width=True)
                st.write(f"⚠️ Nombre de jours de stress (> {p_c['max']}°C) détectés : **{len(stress_only)} jours**.")
            except:
                st.error("Erreur de récupération des archives climatiques.")

        with tab_stats_expert:
            st.header("🔬 Analyse de Robustesse Statistique")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**1. Test de Normalité (Shapiro-Wilk)**")
                if p_norm > 0.05:
                    st.success(f"p={round(p_norm,4)} : Les données suivent une loi normale. Le test de Student est valide.")
                else:
                    st.warning(f"p={round(p_norm,4)} : Les données sont hétérogènes. Le test de Wilcoxon est plus fiable ici.")
                
                st.write("**2. Comparaison des tests**")
                st.write(f"- Test de Student (Paramétrique) : p = `{p_student:.2e}`")
                st.write(f"- Test de Wilcoxon (Non-paramétrique) : p = `{p_wilcoxon:.2e}`")
                
            with col2:
                st.write("**3. Histogramme de Distribution**")
                st.plotly_chart(px.histogram(df_final, x="rdt", color="grp", barmode="overlay"), use_container_width=True)

        with tab_sol:
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df_final.columns]
            st.plotly_chart(px.imshow(df_final[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur système : {e}")
