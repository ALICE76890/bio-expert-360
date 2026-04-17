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

        # --- LOGIQUE D'ANALYSE (DYNAMIQUE) ---
        with st.sidebar:
            with st.expander("🔬 NIVEAU D'ANALYSE", expanded=True):
                mode_analyse = st.radio("Type d'affichage", ["Global par Bande", "Détaillé par Potentiel"])
                pot_cible = "Tous"
                if mode_analyse == "Détaillé par Potentiel":
                    if 'potentiel' in df.columns:
                        liste_pot = ["Tous"] + sorted(list(df['potentiel'].unique()))
                        pot_cible = st.selectbox("Sélectionner le Potentiel", liste_pot)
                    else:
                        st.warning("⚠️ Colonne 'potentiel' manquante.")
                        mode_analyse = "Global par Bande"

        # Définition des groupes
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

        # Données cibles pour les calculs
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean() if not data_p.empty and not data_t.empty else 0

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
                               title="Comparaison globale Produit vs Témoin",
                               color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            else:
                fig_rdt = px.box(df_final, x="potentiel", y="rdt", color="grp", points="all",
                               title=f"Performance dans la zone : {pot_cible}",
                               color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.info("Visualisation climatique désactivée (Focus Statistiques).")

        with tab_stats:
            st.header(f"🔬 Rapport d'Expertise Statistique : {pot_cible if mode_analyse != 'Global par Bande' else 'Global'}")
            
            if len(data_p) > 3 and len(data_t) > 3:
                # --- CALCULS PRÉALABLES ---
                n_p, n_t = len(data_p), len(data_t)
                mean_p, mean_t = data_p.mean(), data_t.mean()
                std_p, std_t = data_p.std(), data_t.std()
                cv_p, cv_t = (std_p/mean_p)*100, (std_t/mean_t)*100
                
                # --- TESTS DE DIAGNOSTIC ---
                stat_sha, p_shapiro = stats.shapiro(data_p)
                stat_lev, p_levene = stats.levene(data_p, data_t)
                
                # --- SÉLECTION DU TEST ---
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom, p_val = "Student (T-test)", stats.ttest_ind(data_p, data_t)[1]
                    test_desc = "Données normales et variances égales. Student est optimal."
                elif p_shapiro > 0.05:
                    test_nom, p_val = "Welch (T-test)", stats.ttest_ind(data_p, data_t, equal_var=False)[1]
                    test_desc = "Données normales mais variances inégales. Welch est plus robuste."
                else:
                    test_nom, p_val = "Mann-Whitney (U-test)", stats.mannwhitneyu(data_p, data_t)[1]
                    test_desc = "Distribution non-normale. Utilisation d'un test de rangs."

                # --- STRUCTURE ET EFFET ---
                ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
                pooled_std = np.sqrt(((n_p - 1) * std_p**2 + (n_t - 1) * std_t**2) / (n_p + n_t - 2))
                d_cohen = (mean_p - mean_t) / pooled_std

                # KPIs Stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Fiabilité Scientifique", f"{round((1-p_val)*100, 2)}%")
                c2.metric("Taille de l'effet (Cohen)", round(d_cohen, 2))
                c3.metric("Stabilité (CV)", f"{round((cv_p+cv_t)/2, 1)}%")

                st.markdown("---")

                col_expli, col_viz = st.columns([1, 1.5])
                
                with col_expli:
                    st.subheader("📖 Interprétation pour le Jury")
                    
                    with st.expander("1. Test de Normalité (Shapiro-Wilk)", expanded=True):
                        st.write(f"**P-Value :** `{p_shapiro:.4f}`")
                        st.write("Vérifie si la répartition suit une courbe de Gauss.")
                    
                    with st.expander("2. Test d'Homogénéité (Levene)"):
                        st.write(f"**P-Value :** `{p_levene:.4f}`")
                        st.write("Vérifie si les deux bandes ont la même régularité.")

                    with st.expander("3. Test de Structure (K-S)"):
                        st.write(f"**P-Value :** `{p_ks:.4f}`")
                        st.write("Vérifie si le produit a modifié la forme de la performance.")

                    st.info(f"💡 **Test retenu : {test_nom}**. {test_desc}")

                with col_viz:
                    fig_ks = px.ecdf(df_final, x="rdt", color="grp", 
                                   title=f"ECDF - Comparaison des Probabilités ({pot_cible})",
                                   color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
                    st.plotly_chart(fig_ks, use_container_width=True)

                st.markdown("### 📋 Annexe Statistique pour le Mémoire")
                df_annexe = pd.DataFrame({
                    "Analyse": ["Normalité", "Homogénéité", "Comparaison Moyennes", "Changement Structure", "Force de l'Impact"],
                    "Outil utilisé": ["Shapiro-Wilk", "Levene", test_nom, "Kolmogorov-Smirnov", "D de Cohen"],
                    "Valeur (p)": [f"{p_shapiro:.4f}", f"{p_levene:.4f}", f"{p_val:.4f}", f"{p_ks:.4f}", f"D = {d_cohen:.2f}"],
                    "Significatif ?": ["Oui" if p_shapiro < 0.05 else "Non", "Oui" if p_levene < 0.05 else "Non", 
                                       "OUI ✅" if p_val < 0.05 else "NON", "OUI ✅" if p_ks < 0.05 else "NON",
                                       "Fort" if abs(d_cohen) > 0.8 else "Modéré"]
                })
                st.table(df_annexe)
            else:
                st.error("❌ Données insuffisantes pour cette zone (minimum 4 points requis).")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
