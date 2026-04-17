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
        d_semis = st.date_input("Date de Semis", datetime(2024, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2025, 3, 10))
        d_recolte = st.date_input("Date de Récolte", datetime(2025, 7, 15))
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
            st.error("❌ Aucun fichier .shp trouvé.")
            st.stop()
            
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        if 'bande' not in df.columns or 'rdt' not in df.columns:
            st.error(f"❌ Colonnes manquantes. Trouvées : {list(df.columns)}")
            st.stop()

        # --- LOGIQUE D'ANALYSE (Sidebar Dynamique) ---
        with st.sidebar:
            with st.expander("🔬 NIVEAU D'ANALYSE", expanded=True):
                mode_analyse = st.radio("Type d'affichage", ["Global par Bande", "Détaillé par Potentiel"])
                pot_cible = "Tous"
                if mode_analyse == "Détaillé par Potentiel":
                    if 'potentiel' in df.columns:
                        liste_pot = ["Tous"] + list(df['potentiel'].unique())
                        pot_cible = st.selectbox("Sélectionner le Potentiel", liste_pot)
                    else:
                        st.warning("⚠️ Colonne 'potentiel' manquante.")
                        mode_analyse = "Global par Bande"

        # Définition du groupe
        val_p = st.sidebar.selectbox("Bande 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Filtrage par potentiel
        df_travail = df.copy()
        if mode_analyse == "Détaillé par Potentiel" and pot_cible != "Tous":
            df_travail = df[df['potentiel'] == pot_cible]

        # Nettoyage IQR
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df_travail[df_travail['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
        else:
            df_final = df_travail.copy()

        # Données pour calculs
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean() if not data_p.empty else 0

        # KPIs principaux
        st.markdown(f"### 📈 Performance : {mode_analyse} {'(' + str(pot_cible) + ')' if mode_analyse != 'Global par Bande' else ''}")
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        # --- 5. ONGLETS ---
        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert"])

        with tab_rdt:
            st.subheader("📊 Visualisation des rendements")
            if mode_analyse == "Global par Bande":
                fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
                               title="Comparaison globale")
            else:
                fig_rdt = px.box(df_final, x="potentiel", y="rdt", color="grp", points="all",
                               title=f"Performance dans la zone : {pot_cible}")
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.info("Visualisation climatique désactivée (Mode Focus Données).")

        with tab_stats:
            st.header(f"🔬 Expertise Statistique Fine : {pot_cible}")
            
            if len(data_p) > 2 and len(data_t) > 2:
                # Calculs
                n_p, n_t = len(data_p), len(data_t)
                cv_p, cv_t = (data_p.std()/data_p.mean())*100, (data_t.std()/data_t.mean())*100
                
                # Tests
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                _, p_stud = stats.ttest_ind(data_p, data_t)
                _, p_welch = stats.ttest_ind(data_p, data_t, equal_var=False)
                _, p_mann = stats.mannwhitneyu(data_p, data_t)
                ks_stat, p_ks = stats.ks_2samp(data_p, data_t)

                # Cohen's D
                pooled_std = np.sqrt(((n_p - 1) * data_p.std()**2 + (n_t - 1) * data_t.std()**2) / (n_p + n_t - 2))
                d_cohen = (data_p.mean() - data_t.mean()) / pooled_std

                # Sélection du test
                if p_shapiro > 0.05 and p_levene > 0.05:
                    best_test, p_final = "Student", p_stud
                elif p_shapiro > 0.05:
                    best_test, p_final = "Welch", p_welch
                else:
                    best_test, p_final = "Mann-Whitney", p_mann

                # Affichage des KPIs Stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CV Moyen", f"{round((cv_p+cv_t)/2, 1)}%")
                c2.metric("Effet (Cohen)", f"{round(d_cohen, 2)}")
                c3.metric("Fiabilité (1-p)", f"{round((1-p_final)*100, 1)}%")
                c4.metric("Distinction (K-S)", f"{'Forte' if p_ks < 0.05 else 'Faible'}")

                st.markdown("---")
                
                col_txt, col_graph = st.columns([1, 2])
                with col_txt:
                    st.write("### 📖 Analyse de la Différence")
                    st.write(f"**Zone analysée :** {pot_cible}")
                    st.write(f"**Test retenu :** {best_test}")
                    
                    if p_ks < 0.05:
                        st.success("✅ Distributions structurellement différentes.")
                    else:
                        st.warning("⚠️ Distributions similaires (chevauchement).")
                        
                    st.write(f"**Interprétation MFE :** Le D de Cohen de `{round(d_cohen,2)}` montre que le produit déplace la performance de la zone de manière **{'majeure' if abs(d_cohen)>0.8 else 'notable'}**.")

                with col_graph:
                    fig_ks = px.ecdf(df_final, x="rdt", color="grp", 
                                   title=f"Courbe de probabilité cumulée (Zone {pot_cible})",
                                   labels={'rdt': 'Rendement (qtx/ha)', 'probability': 'Probabilité'})
                    st.plotly_chart(fig_ks, use_container_width=True)

                with st.expander("🔬 Détails techniques"):
                    st.table(pd.DataFrame({
                        "Indicateur": ["Normalité", "Homogénéité", "P-Value Test", "K-S Stat", "Cohen D"],
                        "Valeur": [p_shapiro, p_levene, p_final, p_ks, d_cohen]
                    }))
            else:
                st.error("Données insuffisantes pour cette zone spécifique.")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
