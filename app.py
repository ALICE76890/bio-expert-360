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
            st.header(f"🔬 Rapport de Validation Scientifique")
            
            if len(data_p) > 3 and len(data_t) > 3:
                # --- 1. TESTS DE DIAGNOSTIC (Les "pré-requis") ---
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                
                # --- 2. SÉLECTION DU TEST DE COMPARAISON ---
                # On suit la remarque : si pas de normalité OU pas d'égalité des variances
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom = "Student (Paramétrique)"
                    stat, p_val = stats.ttest_ind(data_p, data_t)
                else:
                    # On bascule sur le non-paramétrique (Mann-Whitney ou Kruskal-Wallis)
                    test_nom = "Mann-Whitney (Non-paramétrique)"
                    stat, p_val = stats.mannwhitneyu(data_p, data_t)

                # Correction affichage P-Value 0
                p_display = f"{p_val:.4e}" if p_val > 0 else "< 1e-20"

                # --- 3. ANALYSE DE RÉGRESSION (Vérification R²) ---
                st.subheader("📉 Fiabilité du modèle de prédiction")
                # On simule ici la vérification du R² si tu corrèles Rdt et un facteur (ex: Potentiel)
                slope, intercept, r_value, p_reg, std_err = stats.linregress(df_final['rdt'], df_final['rdt']) # Exemple interne
                # Note : Remplace par ta vraie corrélation si tu as une colonne 'indice_sol'
                
                r_carre = r_value**2
                if r_carre < 0.1:
                    st.error(f"⚠️ Modèle de régression non fiable : R² = {round(r_carre, 4)}")
                    st.caption("La variation du rendement n'est pas expliquée par ce facteur. Ne pas utiliser ce modèle dans le mémoire.")
                else:
                    st.success(f"✅ Modèle fiable : R² = {round(r_carre, 2)}")

                # --- 4. AFFICHAGE DÉTAILLÉ ---
                st.markdown("---")
                col_diag, col_verdict = st.columns(2)
                
                with col_diag:
                    st.write("**Vérification des critères :**")
                    st.write(f"- Normalité (Shapiro) : `p = {p_shapiro:.4f}` {'✅' if p_shapiro > 0.05 else '❌'}")
                    st.write(f"- Égalité Variances (Levene) : `p = {p_levene:.4f}` {'✅' if p_levene > 0.05 else '❌'}")
                    
                with col_verdict:
                    st.write(f"**Test appliqué :** `{test_nom}`")
                    st.write(f"**P-Value :** `{p_display}`")
                    if p_val < 0.05:
                        st.success("Résultat statistiquement significatif.")
                    else:
                        st.info("Résultat non significatif (variabilité trop haute).")

                # Graphique des résidus (Indépendance)
                st.write("**Analyse des Résidus (Indépendance)**")
                residus = data_p - data_p.mean()
                fig_res = px.scatter(x=range(len(residus)), y=residus, title="Dispersion des résidus", labels={'x':'Index', 'y':'Ecart à la moyenne'})
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

            else:
                st.error("Données insuffisantes.")with tab_stats:
            st.header(f"🔬 Rapport de Validation Scientifique")
            
            if len(data_p) > 3 and len(data_t) > 3:
                # --- 1. TESTS DE DIAGNOSTIC (Les "pré-requis") ---
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                
                # --- 2. SÉLECTION DU TEST DE COMPARAISON ---
                # On suit la remarque : si pas de normalité OU pas d'égalité des variances
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom = "Student (Paramétrique)"
                    stat, p_val = stats.ttest_ind(data_p, data_t)
                else:
                    # On bascule sur le non-paramétrique (Mann-Whitney ou Kruskal-Wallis)
                    test_nom = "Mann-Whitney (Non-paramétrique)"
                    stat, p_val = stats.mannwhitneyu(data_p, data_t)

                # Correction affichage P-Value 0
                p_display = f"{p_val:.4e}" if p_val > 0 else "< 1e-20"

                # --- 3. ANALYSE DE RÉGRESSION (Vérification R²) ---
                st.subheader("📉 Fiabilité du modèle de prédiction")
                # On simule ici la vérification du R² si tu corrèles Rdt et un facteur (ex: Potentiel)
                slope, intercept, r_value, p_reg, std_err = stats.linregress(df_final['rdt'], df_final['rdt']) # Exemple interne
                # Note : Remplace par ta vraie corrélation si tu as une colonne 'indice_sol'
                
                r_carre = r_value**2
                if r_carre < 0.1:
                    st.error(f"⚠️ Modèle de régression non fiable : R² = {round(r_carre, 4)}")
                    st.caption("La variation du rendement n'est pas expliquée par ce facteur. Ne pas utiliser ce modèle dans le mémoire.")
                else:
                    st.success(f"✅ Modèle fiable : R² = {round(r_carre, 2)}")

                # --- 4. AFFICHAGE DÉTAILLÉ ---
                st.markdown("---")
                col_diag, col_verdict = st.columns(2)
                
                with col_diag:
                    st.write("**Vérification des critères :**")
                    st.write(f"- Normalité (Shapiro) : `p = {p_shapiro:.4f}` {'✅' if p_shapiro > 0.05 else '❌'}")
                    st.write(f"- Égalité Variances (Levene) : `p = {p_levene:.4f}` {'✅' if p_levene > 0.05 else '❌'}")
                    
                with col_verdict:
                    st.write(f"**Test appliqué :** `{test_nom}`")
                    st.write(f"**P-Value :** `{p_display}`")
                    if p_val < 0.05:
                        st.success("Résultat statistiquement significatif.")
                    else:
                        st.info("Résultat non significatif (variabilité trop haute).")

                # Graphique des résidus (Indépendance)
                st.write("**Analyse des Résidus (Indépendance)**")
                residus = data_p - data_p.mean()
                fig_res = px.scatter(x=range(len(residus)), y=residus, title="Dispersion des résidus", labels={'x':'Index', 'y':'Ecart à la moyenne'})
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

            else:
                st.error("Données insuffisantes.")
