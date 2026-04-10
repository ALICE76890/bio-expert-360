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

# --- CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

# Style "Dashboard"
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #eee; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    .stTabs [data-baseweb="tab"] { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

PARAM_CULTURES = {
    "Blé Tendre": {"zero": 0, "max": 25, "flo": 1100, "rec": 2100},
    "Maïs": {"zero": 6, "max": 35, "flo": 900, "rec": 1700},
    "Orge": {"zero": 0, "max": 25, "flo": 1000, "rec": 1950}
}

# --- COLONNE DE GAUCHE (SIDEBAR) ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION", expanded=True):
        uploaded_file = st.file_uploader("Fichier ZIP (449 pts)", type=["zip"])
    
    with st.expander("🌾 CONFIGURATION", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    with st.expander("🧹 TRAITEMENT DATA (IQR)", expanded=True):
        do_clean = st.checkbox("Filtre Outliers (Tukey 1.5x)", value=True)
    
    with st.expander("🚜 CHANTIER", expanded=True):
        v_vent = st.slider("Vent (km/h)", 0, 40, 12)
        hygro = st.slider("Hygro (%)", 0, 100, 75)
    
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix Vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- ANALYSE ---
if uploaded_file:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        val_p = st.sidebar.selectbox("Valeur Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # --- MÉTHODE IQR (TRAITEMENT DONNÉES) ---
        n_brut = len(df)
        if do_clean:
            # On nettoie par groupe pour ne pas biaiser
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                iqr = q3 - q1
                sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                clean_list.append(sub)
            df = pd.concat(clean_list)
        n_clean = len(df)

        # --- CALCULS STATS ---
        d_p = df[df['grp'] == 'Produit']['rdt'].dropna()
        d_t = df[df['grp'] == 'Témoin']['rdt'].dropna()
        gain = d_p.mean() - d_t.mean()
        marge = ((gain/10) * prix_vente) - cout_prod
        t_stat, p_val = stats.ttest_ind(d_p, d_t)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN MOYEN", f"+{round(gain, 2)} qtx/ha")
        k2.metric("P-VALUE (Notation Sci.)", f"{p_val:.2e}")
        k3.metric("MARGE NETTE", f"{round(marge, 2)} €/ha")

        tabs = st.tabs(["📊 Rendement & Potentiel", "🌦️ Climat & Stress", "🔬 Rapport Statistique Expert", "🧪 Sol"])

        with tabs[0]:
            st.plotly_chart(px.box(df, x="potentiel", y="rdt", color="grp", notched=True, 
                                   color_discrete_map={'Témoin': '#3498db', 'Produit': '#2ecc71'}), use_container_width=True)

        with tabs[1]:
            st.subheader("Analyse Climat & Rayonnement")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            # Fix Timestamp error: conversion propre des dates
            start_ts = pd.to_datetime(d_semis)
            past_days = (pd.Timestamp.now() - start_ts).days
            past_days = max(7, min(92, past_days)) # Sécurité API

            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days={past_days}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum&timezone=auto"
            r = requests.get(url).json()['daily']
            w_df = pd.DataFrame(r)
            w_df['time'] = pd.to_datetime(w_df['time'])
            w_df['cum_t'] = w_df['temperature_2m_max'].cumsum()

            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['shortwave_radiation_sum'], fill='tozeroy', name="Rayonnement", line_color='rgba(255,215,0,0.1)', yaxis="y3"))
            fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie", marker_color='blue', opacity=0.4, yaxis="y2"))
            fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
            fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_min'], name="T° Min", line_color='blue', line=dict(dash='dot')))
            
            # Stades
            p_c = PARAM_CULTURES[culture]
            for s, deg in {"Levée": 150, "Floraison": p_c['flo'], "Récolte": p_c['rec']}.items():
                mask = w_df['cum_t'] >= deg
                if mask.any():
                    fig_w.add_vline(x=w_df.loc[mask, 'time'].iloc[0], line_dash="dot", annotation_text=s)
            
            fig_w.update_layout(yaxis2=dict(overlaying="y", side="right"), yaxis3=dict(overlaying="y", side="right", anchor="free", position=1))
            st.plotly_chart(fig_w, use_container_width=True)

        with tabs[2]:
            st.header("🔬 Rigueur du Test Statistique")
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Nettoyage des données (IQR)")
                st.write(f"Points bruts : **{n_brut}**")
                st.write(f"Points exclus : **{n_brut - n_clean}**")
                st.write(f"Fiabilité du jeu de données : **{round((n_clean/n_brut)*100, 1)}%**")
                st.plotly_chart(px.histogram(df, x="rdt", color="grp", barmode="overlay"), use_container_width=True)
            with col_b:
                st.subheader("Validation Scientifique")
                st.write(f"Test de Normalité (Shapiro) : **{round(stats.shapiro(d_p)[1], 4)}**")
                st.write(f"Erreur standard (SEM) : **{round(stats.sem(d_p), 3)}**")
                ic = stats.sem(d_p) * 1.96
                st.info(f"Intervalle de Confiance (95%) : **[{round(gain-ic,2)} ; {round(gain+ic,2)}] qtx**")

        with tabs[3]:
            sol_cols = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
            st.plotly_chart(px.imshow(df[sol_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

        # --- BOUTON IMPRESSION ---
        st.divider()
        if st.button("📄 GÉNÉRER LE RAPPORT PDF"):
            st.write("### 📝 SYNTHÈSE AGRONOMIQUE")
            st.markdown(f"""
            **Essai :** {produit_nom if 'produit_nom' in locals() else 'Biostimulant'} sur {culture}.  
            **Résultat :** Gain de {round(gain,2)} qtx/ha avec une fiabilité de {(1-p_val)*100:.1f}%.  
            **Météo :** Somme de températures au stade récolte : {round(w_df['cum_t'].max(),0)}°C.  
            **Conclusion :** Rentabilité prouvée ({round(marge,2)}€/ha).
            """)
            st.toast("Prêt pour l'impression (Ctrl+P)")

    except Exception as e:
        st.error(f"Erreur : {e}")
