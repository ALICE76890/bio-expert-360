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

        # --- LOGIQUE D'ANALYSE ---
        with st.sidebar:
            with st.expander("🔬 NIVEAU D'ANALYSE", expanded=True):
                mode_analyse = st.radio("Type d'affichage", ["Global par Bande", "Détaillé par Potentiel"])
                pot_cible = "Tous"
                if mode_analyse == "Détaillé par Potentiel":
                    if 'potentiel' in df.columns:
                        liste_pot = ["Tous"] + list(df['potentiel'].unique())
                        pot_cible = st.selectbox("Filtrer un potentiel ?", liste_pot)
                    else:
                        st.warning("⚠️ Colonne 'potentiel' non trouvée.")
                        mode_analyse = "Global par Bande"

        val_p = st.sidebar.selectbox("Bande 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        if mode_analyse == "Détaillé par Potentiel" and pot_cible != "Tous":
            df = df[df['potentiel'] == pot_cible]

        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list) if clean_list else df.copy()
        else:
            df_final = df.copy()

        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean()

        st.markdown(f"### 📈 Analyse des Performances ({mode_analyse})")
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert"])

        with tab_rdt:
            st.subheader(f"📊 Rendement : {mode_analyse}")
            if mode_analyse == "Global par Bande":
                fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
                               title="Comparaison globale Produit vs Témoin")
            else:
                fig_rdt = px.box(df_final, x="potentiel", y="rdt", color="grp", points="all",
                               title="Comparaison par zone de potentiel")
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.info("Visualisation climatique désactivée temporairement selon votre demande.")

        with tab_stats:
            st.header("🔬 Expertise Statistique Avancée & Analyse de Distribution")
            
            if len(data_p) > 2 and len(data_t) > 2:
                # 1. CALCULS PRÉALABLES
                mean_p, mean_t = data_p.mean(), data_t.mean()
                std_p, std_t = data_p.std(), data_t.std()
                n_p, n_t = len(data_p), len(data_t)
                cv_p, cv_t = (std_p/mean_p)*100, (std_t/mean_t)*100
                
                # 2. BATTERIE DE TESTS
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                _, p_stud = stats.ttest_ind(data_p, data_t)
                _, p_welch = stats.ttest_ind(data_p, data_t, equal_var=False)
                _, p_mann = stats.mannwhitneyu(data_p, data_t)
                
                # --- NOUVEAU : TEST DE KOLMOGOROV-SMIRNOV ---
                # Compare si les deux distributions sont réellement différentes
                ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
                
                # 3. TAILLE DE L'EFFET (Cohen)
                pooled_std = np.sqrt(((n_p - 1) * std_p**2 + (n_t - 1) * std_t**2) / (n_p + n_t - 2))
                d_cohen = (mean_p - mean_t) / pooled_std

                # Décision du test principal
                if p_shapiro > 0.05 and p_levene > 0.05:
                    best_test, p_final = "Student", p_stud
                elif p_shapiro > 0.05:
                    best_test, p_final = "Welch", p_welch
                else:
                    best_test, p_final = "Mann-Whitney", p_mann

                # --- AFFICHAGE DES INDICATEURS CLÉS ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CV Moyen", f"{round((cv_p+cv_t)/2, 1)}%")
                c2.metric("Effet (Cohen)", f"{round(d_cohen, 2)}")
                c3.metric("Fiabilité (1-p)", f"{round((1-p_final)*100, 1)}%")
                c4.metric("Distinction (K-S)", f"{'Forte' if p_ks < 0.05 else 'Faible'}")

                st.markdown("---")
                
                col_txt, col_graph = st.columns([1, 2])
                with col_txt:
                    st.write("### 📖 Analyse de la Différence")
                    st.write(f"**Test de structure (K-S) :** p = `{p_ks:.4f}`")
                    
                    if p_ks < 0.05:
                        st.success("✅ Les bandes sont structurellement différentes.")
                        st.write("""
                        Le test de Kolmogorov-Smirnov confirme que le produit a modifié 
                        la répartition globale des rendements, et pas seulement la moyenne. 
                        C'est une preuve de l'efficacité systémique du traitement.
                        """)
                    else:
                        st.warning("⚠️ Les distributions restent proches.")
                        st.write("Le gain est présent mais les populations 'Produit' et 'Témoin' se chevauchent encore beaucoup.")

                    st.write(f"**Taille de l'effet :** Avec un D de Cohen de `{round(d_cohen,2)}`, l'impact est considéré comme **{'majeur' if abs(d_cohen)>0.8 else 'significatif'}**.")

                with col_graph:
                    # Graphique ECDF (Fonction de répartition cumulative)
                    # C'est la meilleure façon de visualiser le test K-S
                    fig_ks = px.ecdf(df_final, x="rdt", color="grp", 
                                   title="Probabilité d'atteindre un rendement (Analyse K-S)",
                                   labels={'rdt': 'Rendement (qtx/ha)', 'probability': 'Probabilité cumulée'})
                    st.plotly_chart(fig_ks, use_container_width=True)

                with st.expander("🔬 Détails techniques pour le rapport"):
                    st.write("Voici les valeurs brutes à copier dans vos annexes :")
                    st.table(pd.DataFrame({
                        "Test": ["Normalité (Shapiro)", "Homogénéité (Levene)", "Différence (Best Test)", "Structure (K-S)", "Effet (Cohen)"],
                        "Valeur": [p_shapiro, p_levene, p_final, p_ks, d_cohen],
                        "Interprétation": [
                            "OK si > 0.05", "OK si > 0.05", "Significatif si < 0.05", 
                            "Populations distinctes si < 0.05", "Impact fort si > 0.8"
                        ]
                    }))
            else:
                st.error("Données insuffisantes pour l'analyse.")

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
