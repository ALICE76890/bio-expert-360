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

st.set_page_config(page_title="Bio-Expert Live", layout="wide")
st.title("🌱 Bio-Expert 360 (Moteur Live)")

# --- 1. CONFIGURATION GAUCHE ---
with st.sidebar:
    st.header("📥 Données")
    uploaded_file = st.file_uploader("ZIP de la batteuse", type=["zip"])
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
    d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
    
    st.header("🧹 Paramètres")
    # On laisse l'utilisateur choisir s'il veut nettoyer les 449 points
    activer_iqr = st.checkbox("Nettoyer les données (IQR)", value=True)

# --- 2. TRAITEMENT ---
if uploaded_file:
    try:
        # Lecture du fichier
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        shp = [f for f in os.listdir("temp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp", shp)).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # Mapping Produit / Témoin
        val_p = st.selectbox("Quelle valeur de 'bande' = Produit ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # --- NETTOYAGE IQR (Tukey) ---
        df_final = df.copy()
        if activer_iqr:
            q1 = df['rdt'].quantile(0.25)
            q3 = df['rdt'].quantile(0.75)
            iqr = q3 - q1
            # On filtre pour garder le "coeur" des données
            df_final = df[(df['rdt'] >= q1 - 1.5*iqr) & (df['rdt'] <= q3 + 1.5*iqr)]
        
        # --- STATISTIQUES ---
        m_p = df_final[df_final['grp'] == 'Produit']['rdt'].mean()
        m_t = df_final[df_final['grp'] == 'Témoin']['rdt'].mean()
        gain = m_p - m_t
        _, p_val = stats.ttest_ind(df_final[df_final['grp']=='Produit']['rdt'], 
                                   df_final[df_final['grp']=='Témoin']['rdt'])

        # --- AFFICHAGE ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Gain Nettoyé", f"+{round(gain, 2)} qtx")
        c2.metric("P-Value", f"{p_val:.2e}")
        c3.metric("Points analysés", f"{len(df_final)} / {len(df)}")

        tab1, tab2 = st.tabs(["📊 Rendement", "🌦️ Météo Live"])

        with tab1:
            st.plotly_chart(px.box(df_final, x="potentiel", y="rdt", color="grp", notched=True))

        with tab2:
            # --- APPEL OPEN-METEO LIVE ---
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            # On demande les 60 derniers jours pour être sûr d'englober l'appli
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=60&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            try:
                res = requests.get(url).json()
                w_df = pd.DataFrame(res['daily'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                
                # On filtre depuis le semis pour le graphique
                mask = w_df['time'] >= pd.to_datetime(d_semis)
                plot_df = w_df[mask]

                # Graphique Température
                st.write(f"📍 Météo récupérée pour : {round(lat,3)}, {round(lon,3)}")
                fig_t = px.line(plot_df, x='time', y='temperature_2m_max', title="T° Max (60 derniers jours)")
                st.plotly_chart(fig_t, use_container_width=True)
                
                # Graphique Pluie
                fig_p = px.bar(plot_df, x='time', y='precipitation_sum', title="Pluie (mm)")
                st.plotly_chart(fig_p, use_container_width=True)
                
            except:
                st.error("L'API météo n'a pas pu répondre. Vérifie ta connexion.")

    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
