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
            st.subheader(f"🌦️ Analyse Météo (Seuils Arvalis {culture})")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            p_c = PARAM_CULTURES[culture]
            
            # Dates sécurisées (Archive s'arrête à J-5)
            d_end_archive = min(d_recolte, (datetime.now() - timedelta(days=5)).date())
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_semis}&end_date={d_end_archive}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            try:
                r = requests.get(url).json()
                if 'daily' in r:
                    w_df = pd.DataFrame(r['daily'])
                    w_df['time'] = pd.to_datetime(w_df['time'])
                    
                    fig_w = go.Figure()
                    # Pluie
                    fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='blue', opacity=0.3))
                    # Temp Max
                    fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                    
                    # Seuils de stress Arvalis
                    fig_w.add_hrect(y0=p_c['echaudage'], y1=p_c['critique'], fillcolor="orange", opacity=0.2, annotation_text="Echaudage")
                    fig_w.add_hrect(y0=p_c['critique'], y1=max(w_df['temperature_2m_max'].max(), 35), fillcolor="red", opacity=0.3, annotation_text="Stress Sévère")
                    
                    # Appli
                    fig_w.add_vline(x=pd.to_datetime(d_appli), line_dash="dash", line_color="green", annotation_text="Application")
                    
                    st.plotly_chart(fig_w, use_container_width=True)
                    
                    # --- THÈSE AGRONOMIQUE ---
                    st.markdown("---")
                    st.markdown("### 🔬 Synthèse de l'aspect climatique de l'année")
                    jours_stress = len(w_df[w_df['temperature_2m_max'] >= p_c['echaudage']])
                    
                    st.write(f"""
                    L'analyse climatique entre le **{d_semis}** et le **{d_end_archive}** met en évidence plusieurs points déterminants pour ton MFE :
                    
                    * **Intensité de l'échaudage :** Nous avons comptabilisé **{jours_stress} jours** au-dessus du seuil de {p_c['echaudage']}°C. 
                        Sur blé, cela réduit la durée de remplissage du grain et peut impacter le PMG.
                    * **Fenêtre d'application :** L'application du produit le **{d_appli}** a eu lieu avant ou pendant les épisodes de chaleur. 
                        Le gain de rendement de **{round(gain,2)} qtx** suggère un effet protecteur du métabolisme face à ces pics de température.
                    * **Effet hydrique :** Un cumul de pluie de **{round(w_df['precipitation_sum'].sum(),1)} mm** a été enregistré. Si les pluies ont été faibles 
                        durant les pics de chaleur, le stress est multiplié, ce qui rend l'effet de BioExpert encore plus significatif.
                    """)
                else:
                    st.warning("⚠️ Aucune donnée météo trouvée pour ces dates.")
            except:
                st.error("❌ Erreur de connexion à l'API Météo.")

        with tab_stats:
            st.header("🔬 Analyse de Robustesse")
            st.plotly_chart(px.histogram(df_final, x="rdt", color="grp", barmode="overlay", title="Distribution des rendements"), use_container_width=True)
            st.write(f"Une P-value de **{p_student:.4f}** indique que la différence est {'statistiquement significative' if p_student < 0.05 else 'non significative'}.")

    except Exception as e:
        st.error(f"❌ Erreur système : {e}")
