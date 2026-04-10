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

# --- CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert R&D", layout="wide", page_icon="🌾")
st.title("🌾 BIO-EXPERT 360 - Analyse de Stress & Statistique")

# --- PARAMÈTRES DE STRESS (Source: Arvalis/Techniques culturales) ---
STRESS_PARAMS = {
    "Blé": {"zero": 0, "max": 25, "echau": 30, "base_t": 600}, # Somme de T° indicative
    "Maïs": {"zero": 6, "max": 35, "echau": 38, "base_t": 1500},
    "Colza": {"zero": 0, "max": 25, "echau": 28, "base_t": 1100}
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Fiche Technique")
    uploaded_file = st.file_uploader("Fichier ZIP (449 points détectés)", type=["zip"])
    culture = st.selectbox("Culture", list(STRESS_PARAMS.keys()))
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
    d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    st.divider()
    produit_nom = st.text_input("Produit", "Expert-Grow")
    prix_vente = st.number_input("Prix Vente (€/T)", value=210)
    cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- ANALYSE ---
if uploaded_file:
    try:
        # Extraction & Lecture [cite: 1, 13]
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp_shp")
        shp = [f for f in os.listdir("temp_shp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp_shp", shp))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        df['lat'], df['lon'] = gdf.to_crs(epsg=4326).geometry.y, gdf.to_crs(epsg=4326).geometry.x

        # Mapping Bande
        liste_b = df['bande'].unique().tolist()
        val_prod = st.sidebar.selectbox("Valeur = Produit ?", liste_b)
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_prod else 'Temoin')

        # --- TABS ---
        tab_stats, tab_stress, tab_sol = st.tabs(["🔬 Rapport Statistique", "🌦️ Analyse Stress Climat", "🧪 Éléments du Sol"])

        with tab_stats:
            st.header("🔬 Détails des Tests Statistiques")
            
            data_p = df[df['grp'] == 'Produit']['rdt'].dropna()
            data_t = df[df['grp'] == 'Temoin']['rdt'].dropna()
            
            # Calculs détaillés
            t_stat, p_val = stats.ttest_ind(data_p, data_t)
            std_p, std_t = data_p.std(), data_t.std()
            
            with st.expander("👁️ Cliquer pour voir les résultats bruts du test"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Groupe Produit**")
                    st.write(f"Moyenne : {round(data_p.mean(), 2)} qtx")
                    st.write(f"Écart-type : {round(std_p, 2)}")
                    st.write(f"Effectif (N) : {len(data_p)} points [cite: 31]")
                with c2:
                    st.write("**Groupe Témoin**")
                    st.write(f"Moyenne : {round(data_t.mean(), 2)} qtx")
                    st.write(f"Écart-type : {round(std_t, 2)}")
                    st.write(f"Effectif (N) : {len(data_t)} points [cite: 49]")
                
                st.divider()
                st.write(f"**Test de Student (T-Test) :**")
                st.write(f"Valeur T : {round(t_stat, 4)}")
                st.info(f"P-Value : {round(p_val, 6)}")
                if p_val < 0.05:
                    st.success("Verdict : Différence statistiquement significative (95% de confiance).")
                else:
                    st.warning("Verdict : Différence non significative (probablement due au hasard).")

        with tab_stress:
            st.header(f"🌦️ Analyse du Stress Climatique ({culture})")
            
            # API Météo de Semis à Aujourd'hui
            lat_m, lon_m = df['lat'].mean(), df['lon'].mean()
            # On cherche les données réelles jusqu'en avril 2026 [cite: 31]
            end_date = datetime.now().strftime("%Y-%m-%d")
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_m}&longitude={lon_m}&start_date={d_semis}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FBerlin"
            
            r = requests.get(url).json()
            m_df = pd.DataFrame(r['daily'])
            m_df['time'] = pd.to_datetime(m_df['time'])

            # Zones de stress Arvalis
            p = STRESS_PARAMS[culture]
            m_df['Stress_Thermique'] = m_df['temperature_2m_max'].apply(lambda x: x if x > p['max'] else None)
            m_df['Gel'] = m_df['temperature_2m_min'].apply(lambda x: x if x < p['zero'] else None)

            # Graphique de Stress
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_df['time'], y=m_df['temperature_2m_max'], name="T° Max", line=dict(color='gray', width=1)))
            
            # Marquage Stress Thermique
            fig.add_trace(go.Scatter(x=m_df['time'], y=m_df['Stress_Thermique'], mode='markers', name="Zone Stress >25°C", marker=dict(color='red', size=8)))
            
            # Position de l'application
            fig.add_vline(x=pd.Timestamp(d_appli), line_width=3, line_dash="dash", line_color="green", annotation_text="Date Application")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Diagnostic
            appli_temp = m_df[m_df['time'] == pd.Timestamp(d_appli)]['temperature_2m_max'].values[0]
            st.write(f"🔎 Au moment de l'application, la température max était de **{appli_temp}°C**.")
            if appli_temp > p['max']:
                st.error(f"⚠️ Alerte : Application en zone de stress thermique pour le {culture} !")
            else:
                st.success(f"✅ Conditions thermiques favorables le jour de l'application.")

        with tab_sol:
            st.header("🧪 Corrélation avec les éléments du sol")
            cols_sol = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca', 'potentiel'] if c in df.columns]
            if len(cols_sol) > 1:
                corr = df[cols_sol].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, title="Impact du sol sur le RDT"), use_container_width=True)
            else:
                st.info("Données de sol insuffisantes.")

    except Exception as e:
        st.error(f"Erreur : {e}")
