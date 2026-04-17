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
            st.header(f"🔬 Rapport d'Expertise Statistique : {pot_cible}")
            
            if len(data_p) > 3 and len(data_t) > 3:
                # --- 1. CALCULS PRÉALABLES ---
                n_p, n_t = len(data_p), len(data_t)
                mean_p, mean_t = data_p.mean(), data_t.mean()
                std_p, std_t = data_p.std(), data_t.std()
                cv_p, cv_t = (std_p/mean_p)*100, (std_t/mean_t)*100
                
                # --- 2. TESTS DE DIAGNOSTIC ---
                # Test de Normalité (Shapiro-Wilk)
                stat_sha, p_shapiro = stats.shapiro(data_p)
                # Test d'Homogénéité (Levene)
                stat_lev, p_levene = stats.levene(data_p, data_t)
                
                # --- 3. SÉLECTION DU TEST DE COMPARAISON ---
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom, p_val = "Student (T-test)", stats.ttest_ind(data_p, data_t)[1]
                    test_desc = "Vos données suivent une loi normale et ont des variances égales. C'est le scénario idéal pour la puissance statistique."
                elif p_shapiro > 0.05:
                    test_nom, p_val = "Welch (T-test)", stats.ttest_ind(data_p, data_t, equal_var=False)[1]
                    test_desc = "Les données sont normales mais les dispersions diffèrent. Welch ajuste les degrés de liberté pour rester précis."
                else:
                    test_nom, p_val = "Mann-Whitney (U-test)", stats.mannwhitneyu(data_p, data_t)[1]
                    test_desc = "La distribution n'est pas normale (asymétrie). On utilise un test de rangs, plus robuste aux valeurs extrêmes."

                # --- 4. TESTS DE STRUCTURE ET D'EFFET ---
                ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
                pooled_std = np.sqrt(((n_p - 1) * std_p**2 + (n_t - 1) * std_t**2) / (n_p + n_t - 2))
                d_cohen = (mean_p - mean_t) / pooled_std

                # --- AFFICHAGE DES RÉSULTATS (KPIs) ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Fiabilité Scientifique", f"{round((1-p_val)*100, 2)}%", help="Plus ce chiffre est proche de 100%, moins le gain est dû au hasard.")
                c2.metric("Taille de l'effet (Cohen)", round(d_cohen, 2), help="Indique la force brute du produit au-delà de la moyenne.")
                c3.metric("Stabilité (CV)", f"{round((cv_p+cv_t)/2, 1)}%", help="Indique si le rendement est régulier ou très hétérogène.")

                st.markdown("---")

                # --- SECTION EXPLICATIVE DÉTAILLÉE ---
                col_expli, col_viz = st.columns([1, 1.5])
                
                with col_expli:
                    st.subheader("📖 Interprétation des Tests")
                    
                    # Explication Shapiro
                    with st.expander("1. Test de Normalité (Shapiro-Wilk)", expanded=(p_shapiro < 0.05)):
                        st.write(f"**P-Value :** `{round(p_shapiro, 4)}`")
                        if p_shapiro > 0.05:
                            st.success("Données Normales : La répartition des rendements est équilibrée (courbe en cloche).")
                        else:
                            st.warning("Données Atypiques : La répartition est décalée ou présente des 'queues' de distribution.")
                    
                    # Explication Levene
                    with st.expander("2. Test d'Homogénéité (Levene)"):
                        st.write(f"**P-Value :** `{round(p_levene, 4)}`")
                        if p_levene > 0.05:
                            st.success("Variances égales : Les deux bandes ont la même régularité.")
                        else:
                            st.info("Variances inégales : Une bande est plus hétérogène que l'autre.")

                    # Explication K-S
                    with st.expander("3. Test de Structure (Kolmogorov-Smirnov)"):
                        st.write(f"**P-Value :** `{round(p_ks, 4)}`")
                        if p_ks < 0.05:
                            st.success("Changement de structure : Le produit a modifié la forme de la performance parcellaire.")
                        else:
                            st.write("Structure identique : Les courbes de probabilité se chevauchent trop.")

                    st.info(f"💡 **Synthèse pour le MFE :** Le test **{test_nom}** a été retenu. {test_desc}")

                with col_viz:
                    st.subheader("📊 Visualisation de la Probabilité")
                    fig_ks = px.ecdf(df_final, x="rdt", color="grp", 
                                   title="Comparaison des fonctions de répartition (K-S)",
                                   labels={'rdt': 'Rendement (qtx/ha)', 'probability': 'Cumul de probabilité'})
                    st.plotly_chart(fig_ks, use_container_width=True)
                    st.caption("Plus l'écart horizontal entre les courbes est grand, plus le test K-S est significatif.")

                # --- TABLEAU RÉCAPITULATIF POUR ANNEXES ---
                st.markdown("### 📋 Tableau récapitulatif pour les annexes")
                df_annexe = pd.DataFrame({
                    "Analyse": ["Normalité", "Homogénéité", "Comparaison Moyennes", "Changement Structure", "Force de l'Impact"],
                    "Outil utilisé": ["Shapiro-Wilk", "Levene", test_nom, "Kolmogorov-Smirnov", "D de Cohen"],
                    "Valeur (p)": [f"{p_shapiro:.4f}", f"{p_levene:.4f}", f"{p_val:.4f}", f"{p_ks:.4f}", f"D = {d_cohen:.2f}"],
                    "Significatif ?": ["Oui" if p_shapiro < 0.05 else "Non", "Oui" if p_levene < 0.05 else "Non", 
                                       "OUI ✅" if p_val < 0.05 else "NON", "OUI ✅" if p_ks < 0.05 else "NON",
                                       "Fort Impact" if abs(d_cohen) > 0.8 else "Impact Modéré"]
                })
                st.table(df_annexe)
            else:
                st.error("❌ Données insuffisantes pour générer l'expertise statistique.")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
