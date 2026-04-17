import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime

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
        clean_outliers = st.checkbox("Nettoyage strict des données (IQR 1.2)", value=True)

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
        
        # --- LOGIQUE D'ANALYSE DYNAMIQUE ---
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

        # Groupement
        val_p = st.sidebar.selectbox("Bande 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Filtrage Potentiel
        df_travail = df.copy()
        if mode_analyse == "Détaillé par Potentiel" and pot_cible != "Tous":
            df_travail = df[df['potentiel'] == pot_cible]

        n_initial = len(df_travail)

        # --- NETTOYAGE IQR SÉVÈRE ---
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df_travail[df_travail['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    # Filtre strict à 1.2
                    sub = sub[(sub['rdt'] >= q1 - 1.2*iqr) & (sub['rdt'] <= q3 + 1.2*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
        else:
            df_final = df_travail.copy()

        n_final = len(df_final)
        pts_ecartes = n_initial - n_final

        # Séparation pour stats
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean() if not data_p.empty and not data_t.empty else 0

        # --- AFFICHAGE KPIs ---
        st.markdown(f"### 📈 Synthèse de Performance ({mode_analyse})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Points analysés (N)", f"{n_final} pts")
        c2.metric("Points écartés", pts_ecartes, delta=f"-{round((pts_ecartes/n_initial)*100, 1)}%", delta_color="inverse")
        c3.metric("Gain Moyen", f"+{round(gain, 2)} qtx")
        c4.metric("Marge Nette", f"{round(((gain/10)*prix_vente)-cout_prod, 1)} €/ha")

        # --- ONGLETS ---
        tab_rdt, tab_stats = st.tabs(["📊 Rendement", "🔬 Stat Expert"])

        with tab_rdt:
            st.subheader("📊 Distribution des rendements")
            fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
                           color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_stats:
            st.header(f"🔬 Rapport d'Expertise Statistique Approfondi : {pot_cible}")
            
            # Recalcul des effectifs précis
            n_p = len(data_p)
            n_t = len(data_t)
            
            if n_p > 3 and n_t > 3:
                # --- 1. CALCULS ET TESTS ---
                mean_p, mean_t = data_p.mean(), data_t.mean()
                std_p, std_t = data_p.std(), data_t.std()
                cv_p, cv_t = (std_p/mean_p)*100, (std_t/mean_t)*100
                
                # Tests de diagnostic
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                
                # Test de structure (K-S)
                ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
                
                # Taille de l'effet (Cohen)
                pooled_std = np.sqrt(((n_p - 1) * std_p**2 + (n_t - 1) * std_t**2) / (n_p + n_t - 2))
                d_cohen = (mean_p - mean_t) / pooled_std

                # Sélection du test de comparaison
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom, p_val = "Student", stats.ttest_ind(data_p, data_t)[1]
                    justif = "Données normales et homogènes."
                elif p_shapiro > 0.05:
                    test_nom, p_val = "Welch", stats.ttest_ind(data_p, data_t, equal_var=False)[1]
                    justif = "Variances hétérogènes détectées (Welch appliqué)."
                else:
                    test_nom, p_val = "Mann-Whitney", stats.mannwhitneyu(data_p, data_t)[1]
                    justif = "Distribution non-normale (Test de rangs appliqué)."

                # --- AFFICHAGE DES EFFECTIFS (N) ---
                st.subheader("1️⃣ Échantillonnage et Stabilité")
                c_n1, c_n2, c_n3, c_n4 = st.columns(4)
                c_n1.metric("N Produit", f"{n_p} pts")
                c_n2.metric("N Témoin", f"{n_t} pts")
                c_n3.metric("CV Produit", f"{round(cv_p, 1)}%")
                c_n4.metric("CV Témoin", f"{round(cv_t, 1)}%")
                
                st.markdown("---")

                # --- RÉSULTATS EXPERTS ---
                st.subheader("2️⃣ Puissance et Fiabilité du Gain")
                col_res1, col_res2, col_res3 = st.columns(3)
                
                p_format = f"{p_val:.4e}" if p_val > 0 else "< 1e-20"
                col_res1.metric("Fiabilité (1-p)", f"{round((1-p_val)*100, 2)}%")
                
                # Interprétation Cohen
                if abs(d_cohen) < 0.2: d_desc = "Négligeable"
                elif abs(d_cohen) < 0.5: d_desc = "Faible"
                elif abs(d_cohen) < 0.8: d_desc = "Modérée"
                else: d_desc = "Forte"
                
                col_res2.metric("Effet (Cohen's D)", round(d_cohen, 2), help=f"Impact {d_desc}")
                col_res3.metric("Distinction (K-S)", "Oui ✅" if p_ks < 0.05 else "Non ❌")

                st.markdown("---")

                # --- SYNTHÈSE RÉDACTIONNELLE ---
                col_txt, col_graph = st.columns([1, 1.2])
                with col_txt:
                    st.write("### 📖 Analyse pour le mémoire")
                    st.info(f"**Test retenu : {test_nom}**\n\n{justif}")
                    
                    if p_val < 0.05:
                        st.success(f"L'impact de la modalité **{val_p}** est statistiquement prouvé sur cet échantillon.")
                    else:
                        st.warning("La variabilité est trop forte pour valider le gain avec certitude.")
                    
                    st.write(f"""
                    **Points clés à retenir :**
                    * La taille de l'effet (**D={round(d_cohen,2)}**) est qualifiée de **{d_desc}**.
                    * Le test de Kolmogorov-Smirnov (**p={round(p_ks, 4)}**) confirme que le produit a 
                    **{'modifié' if p_ks < 0.05 else 'pas modifié'}** la structure globale de rendement.
                    """)

                with col_graph:
                    fig_ks = px.ecdf(df_final, x="rdt", color="grp", 
                                   title="Comparaison des Probabilités Cumulées (K-S)",
                                   color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
                    st.plotly_chart(fig_ks, use_container_width=True)

                # --- TABLEAU ANNEXE ---
                with st.expander("📊 Tableau des tests détaillés (Annexe MFE)"):
                    st.table(pd.DataFrame({
                        "Indicateur": ["Normalité (Shapiro)", "Homogénéité (Levene)", "Comparaison (P-value)", "Structure (K-S)", "Taille d'effet (D)"],
                        "Valeur brute": [p_shapiro, p_levene, p_val, p_ks, d_cohen],
                        "Seuil / Statut": ["> 0.05", "> 0.05", "< 0.05", "< 0.05", d_desc]
                    }))
            else:
                st.error("❌ Données insuffisantes pour l'expertise détaillée.")
    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
