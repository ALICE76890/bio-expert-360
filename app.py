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
st.set_page_config(page_title="Bio-Expert 360 R&D", layout="wide", page_icon="🌾")
st.title("🌾 BIO-EXPERT 360 - Analyse de Précision R&D")

# --- PARAMÈTRES DE STRESS (Sources Arvalis) ---
STRESS_PARAMS = {
    "Blé": {"zero": 0, "max": 25, "echau": 30, "base_t": 600},
    "Maïs": {"zero": 6, "max": 35, "echau": 38, "base_t": 1500},
    "Orge": {"zero": 0, "max": 25, "echau": 28, "base_t": 550},
    "Colza": {"zero": 0, "max": 25, "echau": 28, "base_t": 1100}
}

# --- BARRE LATÉRALE COMPLÈTE ---
with st.sidebar:
    st.header("📂 Données & Diagnostic")
    uploaded_file = st.file_uploader("Charger ZIP (shp, dbf, shx)", type=["zip"])
    mode_debug = st.checkbox("🛠️ Activer le mode diagnostic")
    
    st.divider()
    st.header("🚜 Itinéraire Technique")
    culture = st.selectbox("Culture", list(STRESS_PARAMS.keys()))
    produit_nom = st.selectbox("Produit appliqué", ["Bio-Stim 1", "Expert-Grow", "Nitro-Plus", "Autre..."])
    dose = st.text_input("Dose (ex: 2L/ha)", "2.0 L/ha")
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
    d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    st.divider()
    st.header("💰 Économie")
    prix_vente = st.number_input("Prix Vente (€/T)", value=210)
    cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- LOGIQUE PRINCIPALE ---
if uploaded_file:
    try:
        # Extraction & Lecture
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp_shp")
        shp = [f for f in os.listdir("temp_shp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp_shp", shp))
        df = pd.DataFrame(gdf.to_crs(epsg=4326).drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Coordonnées pour météo
        gdf_wgs = gdf.to_crs(epsg=4326)
        df['lat'], df['lon'] = gdf_wgs.geometry.y, gdf_wgs.geometry.x

        # Mapping des bandes
        st.sidebar.warning("🎯 Configuration de la comparaison")
        liste_valeurs = df['bande'].unique().tolist()
        val_prod = st.sidebar.selectbox("Quelle valeur dans 'bande' = Produit ?", liste_valeurs)
        df['label_bande'] = df['bande'].apply(lambda x: 'Produit' if x == val_prod else 'Temoin')

        if mode_debug:
            st.info("🛠️ Diagnostic : Colonnes lues")
            st.write(list(df.columns))
            st.dataframe(df.head(3))

        # --- CALCULS STATISTIQUES ---
        if 'rdt' in df.columns:
            # Stats Globales
            m = df.groupby('label_bande')['rdt'].mean()
            gain_global = m.get('Produit', 0) - m.get('Temoin', 0)
            
            # Onglets
            t_rdt, t_stat, t_stress, t_sol = st.tabs(["📊 Rendement", "🔬 Rapport Statistique", "🌦️ Stress Climat", "🧪 Éléments du Sol"])

            with t_rdt:
                st.header("Comparaison par Potentiel de Sol")
                # Boxplot groupé par potentiel
                fig_box = px.box(df, x="potentiel" if "potentiel" in df.columns else "label_bande", 
                                 y="rdt", color="label_bande", 
                                 title="Rendement : Produit vs Témoin à potentiel équivalent",
                                 points="all")
                st.plotly_chart(fig_box, use_container_width=True)
                
                if 'potentiel' in df.columns:
                    st.write("**Analyse par zone de potentiel :**")
                    summary = df.groupby(['potentiel', 'label_bande'])['rdt'].mean().unstack()
                    summary['Gain'] = summary['Produit'] - summary['Temoin']
                    st.table(summary.style.format("{:.2f} qtx"))

            with t_stat:
                st.header("🔬 Détails des Tests Scientifiques")
                
                # Test Global
                d_p = df[df['label_bande'] == 'Produit']['rdt'].dropna()
                d_t = df[df['label_bande'] == 'Temoin']['rdt'].dropna()
                t_val, p_val = stats.ttest_ind(d_p, d_t)

                st.subheader("1. Test Global (Student)")
                c1, c2, c3 = st.columns(3)
                c1.metric("P-Value (Fiabilité)", f"{p_val:.2e}")
                c2.metric("Significatif ?", "✅ OUI" if p_val < 0.05 else "❌ NON")
                c3.metric("Gain Net", f"{round(gain_global, 2)} qtx")

                with st.expander("📝 Comprendre les résultats"):
                    st.write(f"**Moyenne Produit :** {round(d_p.mean(), 2)} | **Écart-type :** {round(d_p.std(), 2)}")
                    st.write(f"**Moyenne Témoin :** {round(d_t.mean(), 2)} | **Écart-type :** {round(d_t.std(), 2)}")
                    st.info("Une P-Value inférieure à 0.05 (5e-2) signifie qu'il y a moins de 5% de chances que ce résultat soit dû au hasard.")

                if 'potentiel' in df.columns:
                    st.subheader("2. Tests par niveau de Potentiel")
                    for pot in df['potentiel'].unique():
                        sub = df[df['potentiel'] == pot]
                        p_p = sub[sub['label_bande'] == 'Produit']['rdt'].dropna()
                        p_t = sub[sub['label_bande'] == 'Temoin']['rdt'].dropna()
                        if len(p_p) > 1 and len(p_t) > 1:
                            _, p_loc = stats.ttest_ind(p_p, p_t)
                            st.write(f"Zone Potentiel **{pot}** : P-Value = `{p_loc:.2e}` ({'✅ OK' if p_loc < 0.05 else '❌ Incertain'})")

            with t_stress:
                st.header(f"🌦️ Analyse Stress Thermique ({culture})")
                date_start = pd.to_datetime(d_semis).strftime("%Y-%m-%d")
                date_end = datetime.now().strftime("%Y-%m-%d")
                lat_m, lon_m = df['lat'].mean(), df['lon'].mean()
                
                url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_m}&longitude={lon_m}&start_date={date_start}&end_date={date_end}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FBerlin"
                r = requests.get(url).json()
                m_df = pd.DataFrame(r['daily'])
                m_df['time'] = pd.to_datetime(m_df['time'])
                
                # Seuils Arvalis
                p = STRESS_PARAMS[culture]
                m_df['Stress'] = m_df['temperature_2m_max'].apply(lambda x: x if x > p['max'] else None)

                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(x=m_df['time'], y=m_df['temperature_2m_max'], name="Temp Max", line=dict(color='silver')))
                fig_m.add_trace(go.Scatter(x=m_df['time'], y=m_df['Stress'], mode='markers', name="Alerte Stress", marker=dict(color='red', size=8)))
                
                # Date appli
                d_app_ts = pd.to_datetime(d_appli)
                fig_m.add_vline(x=d_app_ts.timestamp()*1000, line_dash="dash", line_color="green", annotation_text="Application")
                
                st.plotly_chart(fig_m, use_container_width=True)
                st.info(f"Seuil de stress {culture} : {p['max']}°C. Vérifiez si l'application a eu lieu avant une zone rouge.")

            with t_sol:
                st.header("🧪 Impact des éléments du sol")
                sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca', 'potentiel'] if c in df.columns]
                if len(sol_cols) > 1:
                    corr = df[sol_cols].corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
                else:
                    st.warning("Données de sol absentes.")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
