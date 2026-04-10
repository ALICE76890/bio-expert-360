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

# --- PARAMÈTRES AGRONOMIQUES ---
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
        st.info("Méthode : IQR 1.5x")

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
        # 1. Extraction et Lecture
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp_name = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp_name)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # 2. Mapping Produit/Témoin
        val_p = st.sidebar.selectbox("Ligne Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # 3. Nettoyage IQR (Méthode de Tukey)
        n_initial = len(df)
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                iqr = q3 - q1
                sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                clean_list.append(sub)
            df_cleaned = pd.concat(clean_list)
        else:
            df_cleaned = df.copy()
        
        n_removed = n_initial - len(df_cleaned)

        # 4. Calculs Statistiques
        data_p = df_cleaned[df_cleaned['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_cleaned[df_cleaned['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()
        marge = ((gain/10) * prix_vente) - cout_prod
        t_stat, p_val = stats.ttest_ind(data_p, data_t)

        # 5. Affichage des KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN (NETTOYÉ)", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_val:.2e}")
        k3.metric("MARGE NETTE", f"{round(marge, 2)} €/ha")

        # 6. Onglets
        tab_rdt, tab_climat, tab_stats, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat", "🔬 Preuve & Nettoyage", "🧪 Sol"])

        with tab_rdt:
            st.subheader("📍 Réponse RDT vs Potentiel Sol")
            fig_rdt = px.box(df_cleaned, x="potentiel", y="rdt", color="grp", notched=True,
                             color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.subheader("📅 Timeline Climatique & Stress")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            
            # Correction Météo : Utilisation de l'API Forecast (plus stable pour le récent)
            # On calcule le nombre de jours entre semis et aujourd'hui
            start_date = pd.to_datetime(d_semis)
            past_days = (pd.Timestamp.now() - start_date).days
            past_days = max(7, min(92, past_days)) # Sécurité API entre 7 et 92 jours
            
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days={past_days}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=auto"
            
            try:
                r = requests.get(url).json()
                if "daily" in r:
                    w_df = pd.DataFrame(r['daily'])
                    w_df['time'] = pd.to_datetime(w_df['time'])
                    
                    fig_w = go.Figure()
                    # Rayonnement (MJ/m²)
                    fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], fill='tozeroy', name="Rayonnement", line_color='rgba(255,215,0,0.2)', yaxis="y3"))
                    # Pluie
                    fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='blue', opacity=0.4, yaxis="y2"))
                    # Températures
                    fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                    fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
                    
                    # Ligne Application
                    fig_w.add_vline(x=pd.Timestamp(d_appli).timestamp() * 1000, line_dash="dash", line_color="green", annotation_text="Appli")

                    fig_w.update_layout(yaxis2=dict(overlaying="y", side="right"), yaxis3=dict(overlaying="y", side="right", anchor="free", position=1))
                    st.plotly_chart(fig_w, use_container_width=True)
                else:
                    st.warning("Données météo momentanément indisponibles sur cette zone.")
            except:
                st.error("Erreur lors de l'appel météo.")

        with tab_stats:
            st.header("🔬 Qualité des données")
            c1, c2, c3 = st.columns(3)
            c1.metric("Points initiaux", n_initial)
            c2.metric("Points supprimés", n_removed, delta_color="inverse")
            c3.metric("Conservation", f"{round((len(df_cleaned)/n_initial)*100, 1)}%")
            
            st.plotly_chart(px.histogram(df_cleaned, x="rdt", color="grp", barmode="overlay"), use_container_width=True)

        with tab_sol:
            st.subheader("🧪 Corrélation Éléments du Sol")
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df_cleaned.columns]
            if len(sol_cols) > 1:
                st.plotly_chart(px.imshow(df_cleaned[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur système : {e}")
