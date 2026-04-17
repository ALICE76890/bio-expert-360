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

# --- 1. CONFIGURATION PAGE ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

# --- 2. RÉFÉRENTIEL ARVALIS ---
PARAM_CULTURES = {
    "Blé Tendre": {"echaudage": 25, "critique": 30, "base_t": 0},
    "Maïs": {"echaudage": 35, "critique": 38, "base_t": 6},
    "Orge": {"echaudage": 25, "critique": 30, "base_t": 0}
}

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        # Dates par défaut réalistes pour un test (campagne passée)
        d_semis = st.date_input("Date de Semis", datetime(2024, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2025, 3, 10))
        d_recolte = st.date_input("Date de Récolte", datetime(2025, 7, 15))
        clean_outliers = st.checkbox("Filtrer points aberrants (IQR)", value=True)

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)
        
    with st.expander("🔬 NIVEAU D'ANALYSE", expanded=True):
        mode_analyse = st.radio("Type d'affichage", ["Global par Bande", "Détaillé par Potentiel"])
        if mode_analyse == "Détaillé par Potentiel":
            if 'potentiel' in df.columns:
                liste_pot = ["Tous"] + list(df['potentiel'].unique())
                pot_cible = st.selectbox("Filtrer un potentiel ?", liste_pot)
            else:
                st.warning("⚠️ Colonne 'potentiel' non trouvée dans le fichier.")
                mode_analyse = "Global par Bande"

# --- 4. TRAITEMENT DU FICHIER ---
if uploaded_file:
    try:
        clear_temp()
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        
        shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
        if not shp_files:
            st.error("❌ Aucun fichier .shp trouvé.")
            st.stop()
            
        # Conversion GPS impérative pour l'API Météo
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        if 'bande' not in df.columns or 'rdt' not in df.columns:
            st.error(f"❌ Colonnes manquantes. Trouvées : {list(df.columns)}")
            st.stop()

        val_p = st.sidebar.selectbox("Bande 'Produit' ?", df['bande'].unique())
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
        st.markdown("### 📈 Analyse des Performances")
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("P-VALUE (Fiabilité)", f"{p_student:.4f}")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        # --- 5. ONGLETS ---
        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert"])

        with tab_rdt:
            fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all",
                           title="Comparaison des Rendements",
                           color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.subheader(f"🌦️ Étude de l'Échaudage (Seuils Arvalis)")
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            p_c = PARAM_CULTURES[culture]
            
            # Sécurité Date : Pas de données météo dans le futur
            # On prend la date la plus ancienne entre d_recolte et J-5
            limit_date = (datetime.now() - timedelta(days=5)).date()
            safe_end = min(d_recolte, limit_date)
            
            # Formatage strict pour l'URL
            s_str = d_semis.strftime("%Y-%m-%d")
            e_str = safe_end.strftime("%Y-%m-%d")
            
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={s_str}&end_date={e_str}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    w_data = resp.json()
                    if 'daily' in w_data:
                        w_df = pd.DataFrame(w_data['daily'])
                        w_df['time'] = pd.to_datetime(w_df['time'])
                        
                        fig_w = go.Figure()
                        fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Pluie (mm)", marker_color='blue', opacity=0.3))
                        fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], name="T° Max", line_color='red'))
                        
                        # Zones de Stress
                        fig_w.add_hrect(y0=p_c['echaudage'], y1=p_c['critique'], fillcolor="orange", opacity=0.2, annotation_text="Zone d'échaudage")
                        fig_w.add_hrect(y0=p_c['critique'], y1=max(w_df['temperature_2m_max'].max()+2, 35), fillcolor="red", opacity=0.3, annotation_text="Stress Sévère")
                        
                        # Appli
                        fig_w.add_vline(x=pd.to_datetime(d_appli), line_dash="dash", line_color="green", annotation_text="Appli")
                        st.plotly_chart(fig_w, use_container_width=True)
                        
                        # --- THÈSE AGRONOMIQUE DYNAMIQUE ---
                        st.markdown("---")
                        st.markdown("### 🔬 Synthèse Climatique pour le Mémoire")
                        jours_stress = len(w_df[w_df['temperature_2m_max'] >= p_c['echaudage']])
                        pluie_cumul = round(w_df['precipitation_sum'].sum(), 1)
                        
                        st.info(f"""
                        **Analyse du cycle (du {s_str} au {e_str}) :**
                        
                        1. **Fréquence de l'échaudage :** La culture a subi **{jours_stress} jours** au-dessus de {p_c['echaudage']}°C. 
                           Ces épisodes, survenant après l'application du produit le {d_appli.strftime('%d/%m')}, 
                           impactent normalement le PMG en écourtant le remplissage.
                        2. **Contexte hydrique :** Un cumul de **{pluie_cumul} mm** a été relevé. Si ce cumul est faible durant les pics thermiques, 
                           le gain observé de **{round(gain,2)} qtx** souligne l'efficacité de BioExpert 360 dans la gestion du stress abiotique.
                        3. **Conclusion :** Le produit semble avoir sécurisé le potentiel de rendement en protégeant le métabolisme foliaire.
                        """)
                    else:
                        st.error("L'API n'a pas renvoyé de données journalières.")
                else:
                    st.error(f"Erreur API (Code {resp.status_code})")
                    st.info(f"Vérifie l'URL : {url}")
            except Exception as e:
                st.error(f"Connexion impossible : {e}")

        with tab_stats:
            st.header("🔬 Fiabilité Statistique")
            st.plotly_chart(px.histogram(df_final, x="rdt", color="grp", barmode="overlay"), use_container_width=True)
            st.write(f"Significativité (alpha=0.05) : **{'OUI' if p_student < 0.05 else 'NON'}**")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
