import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from scipy import stats
import requests
from datetime import datetime, timedelta
import io
import zipfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert Rapide", layout="wide")
st.title("🌱 Bio-Expert 360 (Mode Stable)")

# --- 2. MENU GAUCHE ---
with st.sidebar:
    st.header("📥 Données")
    uploaded_file = st.file_uploader("Charger ZIP", type=["zip"])
    
    st.header("🌾 Essai")
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
    d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    st.header("💰 Économie")
    prix_t = st.number_input("Prix Vente (€/T)", value=210)
    cout_ha = st.number_input("Coût Produit (€/ha)", value=45)

# --- 3. TRAITEMENT ---
if uploaded_file:
    try:
        # Extraction simple
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping Produit vs Témoin
        val_p = st.selectbox("Quelle valeur = Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # --- NETTOYAGE IQR (Simple & Efficace) ---
        q1 = df['rdt'].quantile(0.25)
        q3 = df['rdt'].quantile(0.75)
        iqr = q3 - q1
        df_clean = df[(df['rdt'] >= q1 - 1.5*iqr) & (df['rdt'] <= q3 + 1.5*iqr)]
        points_enleves = len(df) - len(df_clean)

        # --- CALCULS ---
        m_p = df_clean[df_clean['grp'] == 'Produit']['rdt'].mean()
        m_t = df_clean[df_clean['grp'] == 'Témoin']['rdt'].mean()
        gain = m_p - m_t
        marge = ((gain/10) * prix_t) - cout_ha
        _, p_val = stats.ttest_ind(df_clean[df_clean['grp']=='Produit']['rdt'], 
                                   df_clean[df_clean['grp']=='Témoin']['rdt'])

        # --- AFFICHAGE ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Gain RDT", f"+{round(gain, 2)} qtx")
        c2.metric("Marge Nette", f"{round(marge, 2)} €/ha")
        c3.metric("Fiabilité (p)", f"{p_val:.4f}")

        st.write(f"ℹ️ {points_enleves} points aberrants ont été supprimés par le filtre IQR.")

        tabs = st.tabs(["📊 Graphique", "🌦️ Météo"])

        with tabs[0]:
            st.plotly_chart(px.box(df_clean, x="potentiel", y="rdt", color="grp", notched=True))

        with tabs[1]:
            # Météo ultra-sécurisée (7 derniers jours seulement pour tester)
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=7&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            try:
                res = requests.get(url).json()['daily']
                w_df = pd.DataFrame(res)
                st.line_chart(w_df.set_index('time')['temperature_2m_max'])
                st.bar_chart(w_df.set_index('time')['precipitation_sum'])
            except:
                st.warning("Météo indisponible pour le moment.")

    except Exception as e:
        st.error(f"Erreur : {e}")
