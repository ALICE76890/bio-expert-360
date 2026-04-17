import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from scipy import stats
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🌾 CONFIGURATION", expanded=True):
        culture = st.selectbox("Culture", ["Blé Tendre", "Maïs", "Orge"])
        clean_outliers = st.checkbox("Nettoyage IQR Strict (1.2)", value=True)
        prix_vente = st.number_input("Prix (€/T)", value=210)
        cout_prod = st.number_input("Coût (€/ha)", value=45)

# --- 3. LOGIQUE PRINCIPALE ---
if uploaded_file:
    try:
        clear_temp()
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        
        shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
        if not shp_files:
            st.error("Fichier .shp manquant.")
            st.stop()
            
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()

        # Choix du niveau d'analyse
        with st.sidebar:
            mode_analyse = st.radio("Niveau d'analyse", ["Global", "Par Potentiel"])
            pot_cible = "Tous"
            if mode_analyse == "Par Potentiel" and 'potentiel' in df.columns:
                liste_pot = sorted(list(df['potentiel'].unique()))
                pot_cible = st.selectbox("Choisir Potentiel", liste_pot)
            
            val_p = st.selectbox("Bande 'Produit' ?", df['bande'].unique())

        # Création des groupes
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
        
        # Filtrage
        df_travail = df.copy()
        if mode_analyse == "Par Potentiel" and pot_cible != "Tous":
            df_travail = df[df['potentiel'] == pot_cible]

        # Nettoyage IQR 1.2
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df_travail[df_travail['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    sub = sub[(sub['rdt'] >= q1 - 1.2*iqr) & (sub['rdt'] <= q3 + 1.2*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list)
        else:
            df_final = df_travail.copy()

        # Effectifs (N)
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        n_p, n_t = len(data_p), len(data_t)

        # Affichage
        st.subheader(f"Analyse : {mode_analyse} ({pot_cible})")
        
        tab_rdt, tab_stats = st.tabs(["📊 Rendement", "🔬 Stat Expert"])

        with tab_rdt:
            st.plotly_chart(px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True), use_container_width=True)

        with tab_stats:
            if n_p > 3 and n_t > 3:
                # 1. DIAGNOSTICS RÉELS
                _, p_sha = stats.shapiro(data_p)
                _, p_lev = stats.levene(data_p, data_t)
                
                # 2. ARBRE DE DÉCISION
                if p_sha > 0.05 and p_lev > 0.05:
                    test_nom, p_val = "Student (T-test)", stats.ttest_ind(data_p, data_t)[1]
                    raison = "Données normales et stables : Student est le plus précis."
                else:
                    test_nom = "Mann-Whitney (U-test)"
                    p_val = stats.mannwhitneyu(data_p, data_t)[1]
                    raison = "Données atypiques ou instables : Mann-Whitney est plus fiable."

                # 3. RÉGRESSION ET R²
                slope, intercept, r_val, p_reg, std_err = stats.linregress(range(len(data_p)), data_p)
                r2 = r_val**2

                # 4. AFFICHAGE DÉTAILLÉ
                st.write(f"### 🛡️ Audit Statistique ({test_nom})")
                st.info(f"**Pourquoi ce test ?** {raison}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("N (Prd / Tém)", f"{n_p} / {n_t}")
                c2.metric("Fiabilité (1-p)", f"{round((1-p_val)*100, 2)}%")
                c3.metric("Qualité Modèle (R²)", round(r2, 3))

                st.markdown("---")
                st.subheader("📝 Synthèse Agronomique")
                if p_val < 0.05:
                    if r2 < 0.2:
                        st.warning("⚠️ **Impact réel mais masqué** : Le gain est fiable, mais le sol (faible R²) domine l'essai.")
                    else:
                        st.success("✅ **Impact réel et solide** : Le produit explique bien la performance.")
                else:
                    st.error("❌ **Aucun impact prouvé** : La variabilité naturelle explique les différences.")

                with st.expander("🔍 Glossaire des tests effectués"):
                    st.write("**Shapiro-Wilk** : Test de normalité. On cherche p > 0.05.")
                    st.write("**Levene** : Test d'homogénéité (stabilité). On cherche p > 0.05.")
                    st.write("**R²** : Part de variation expliquée par le modèle (0 à 1).")
            else:
                st.error("Pas assez de points pour les stats.")

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Veuillez charger un fichier ZIP pour démarrer l'analyse.")
