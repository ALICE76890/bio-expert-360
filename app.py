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

# --- 1. CONFIGURATION ET NETTOYAGE ---
st.set_page_config(page_title="Bio-Expert 360 Pro", layout="wide", page_icon="🌱")

def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

# --- 2. RÉFÉRENTIEL AGRONOMIQUE (ARVALIS) ---
PARAM_CULTURES = {
    "Blé Tendre": {"echaudage": 25, "critique": 30, "base_t": 0},
    "Maïs": {"echaudage": 35, "critique": 38, "base_t": 6},
    "Orge": {"echaudage": 25, "critique": 30, "base_t": 0}
}

# --- 3. INTERFACE LATERALE (SIDEBAR) ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2025, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2026, 3, 10))
        d_recolte = st.date_input("Date de Récolte", datetime(2026, 7, 15))
        clean_outliers = st.checkbox("Filtrer points aberrants (IQR)", value=True)

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- 4. TRAITEMENT DU FICHIER ---
if uploaded_file:
    try:
        clear_temp()
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        
        shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
        if not shp_files:
            st.error("❌ Aucun fichier .shp trouvé dans le ZIP.")
            st.stop()
            
        # Lecture et conversion GPS (EPSG:4326 pour la météo)
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        if 'bande' not in df.columns or 'rdt' not in df.columns:
            st.error(f"❌ Colonnes 'bande' ou 'rdt' manquantes. Trouvées : {list(df.columns)}")
            st.stop()

        # Selection de la bande testée
        val_p = st.sidebar.selectbox("Quelle bande est le 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Nettoyage IQR
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

        # Stats
        _, p_student = stats.ttest_ind(data_p, data_t)
        gain = data_p.mean() - data_t.mean()

        # KPIs
        st.markdown("### 📈 Résultats de l'Essai")
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_student:.4f}")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        # --- 5. ONGLETS D'ANALYSE ---
        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert"])

        with tab_rdt:
            fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all",
                           title="Dispersion des rendements par modalité",
                           color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            st.plotly_chart(fig_rdt, use_container_width=True)
with tab_climat:
            st.subheader(f"🌦️ Analyse Météo (Arvalis {culture})")
            
            # --- 1. SÉCURISATION DES COORDONNÉES ---
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            
            # Vérification : une latitude doit être entre -90 et 90
            if not (-90 <= lat <= 90):
                st.error(f"❌ Coordonnées invalides : Lat={lat}, Lon={lon}. Vérifie que ton fichier QGIS est bien en WGS84 (EPSG:4326).")
                st.stop()

            # --- 2. PRÉPARATION DES DATES ---
            # Sécurité J-5 pour l'archive
            date_max_dispo = (datetime.now() - timedelta(days=5)).date()
            safe_end_date = min(d_recolte, date_max_dispo)
            
            d_start_str = d_semis.strftime("%Y-%m-%d")
            d_end_str = safe_end_date.strftime("%Y-%m-%d")
            
            # --- 3. APPEL API AVEC GESTION D'ERREURS ---
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_start_str}&end_date={d_end_str}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            try:
                # verify=False permet d'ignorer les erreurs de certificat SSL sur certains réseaux restrictifs
                response = requests.get(url, timeout=15, verify=True) 
                
                if response.status_code == 200:
                    r = response.json()
                    if 'daily' in r:
                        w_df = pd.DataFrame(r['daily'])
                        w_df['time'] = pd.to_datetime(w_df['time'])
                        p_c = PARAM_CULTURES[culture]
                        
                        # Graphique Plotly
                        fig_w = go.Figure()
                        fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='rgba(0,0,255,0.3)'))
                        fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                        
                        # Zones de stress
                        fig_w.add_hrect(y0=p_c['echaudage'], y1=p_c['critique'], fillcolor="orange", opacity=0.2, annotation_text="Échaudage")
                        fig_w.add_hrect(y0=p_c['critique'], y1=max(w_df['temperature_2m_max'].max() + 2, 35), fillcolor="red", opacity=0.3, annotation_text="Stress Sévère")
                        
                        # Date d'application
                        fig_w.add_vline(x=pd.to_datetime(d_appli), line_dash="dash", line_color="green", annotation_text="Appli")
                        
                        st.plotly_chart(fig_w, use_container_width=True)
                        
                        # --- SYNTHÈSE MFE ---
                        st.markdown("---")
                        st.markdown("### 🔬 Note de Synthèse Climatique")
                        jours_stress = len(w_df[w_df['temperature_2m_max'] >= p_c['echaudage']])
                        pluie_totale = round(w_df['precipitation_sum'].sum(), 1)
                        
                        st.write(f"""
                        L'analyse entre le **{d_start_str}** et le **{d_end_str}** montre :
                        * **Stress thermique :** {jours_stress} jours au-dessus de {p_c['echaudage']}°C.
                        * **Hydrométrie :** Cumul de {pluie_totale} mm sur la période.
                        
                        **Interprétation :** Le gain de rendement de **{round(gain, 2)} qtx/ha** est à analyser au regard de ces contraintes. 
                        Un produit appliqué le {d_appli.strftime('%d/%m')} permet de limiter l'impact de l'échaudage sur le PMG.
                        """)
                    else:
                        st.warning("⚠️ L'API a répondu mais les données sont vides. Vérifie tes dates.")
                else:
                    st.error(f"❌ Erreur API (Code {response.status_code})")
                    with st.expander("Détails du lien généré (Debug)"):
                        st.write(url)

            except Exception as e:
                st.error(f"❌ La connexion a échoué.")
                st.info("Conseil : Vérifie que tu as accès à internet et que les dates ne sont pas dans le futur.")
                with st.expander("Erreur technique complète"):
                    st.write(str(e))
     
