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
        clean_outliers = st.checkbox("Nettoyage strict (IQR 1.2)", value=True)

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
                    sub = sub[(sub['rdt'] >= q1 - 1.2*iqr) & (sub['rdt'] <= q3 + 1.2*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
        else:
            df_final = df_travail.copy()

        # Effectifs par bande
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        n_p, n_t = len(data_p), len(data_t)
        gain = data_p.mean() - data_t.mean() if n_p > 0 and n_t > 0 else 0

        # --- CALCULS STATS DÉTAILLÉS ---
        if n_p > 3 and n_t > 3:
            _, p_shapiro = stats.shapiro(data_p)
            _, p_levene = stats.levene(data_p, data_t)
            ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
            
            # Sélection du test
            if p_shapiro > 0.05 and p_levene > 0.05:
                test_nom, p_val = "Student", stats.ttest_ind(data_p, data_t)[1]
                test_id = "PARAM"
            else:
                test_nom, p_val = "Mann-Whitney", stats.mannwhitneyu(data_p, data_t)[1]
                test_id = "NON_PARAM"

            # --- SIDEBAR : GLOSSAIRE DYNAMIQUE ---
            with st.sidebar:
                with st.expander("📖 COMPRENDRE LES TESTS", expanded=False):
                    st.write(f"**Test actuel : {test_nom}**")
                    if test_id == "PARAM":
                        st.info("On utilise **Student** car vos données sont 'normales' (en cloche) et stables. C'est le test le plus puissant.")
                    else:
                        st.warning("On utilise **Mann-Whitney** car vos données présentent des anomalies ou une forte asymétrie. Ce test est plus sûr ici.")
                    
                    st.write("---")
                    st.write("**Shapiro-Wilk** : Vérifie si vos rendements suivent une distribution naturelle.")
                    st.write("**Levene** : Vérifie si la variabilité est la même dans les deux bandes.")
                    st.write("**K-S** : Regarde si toute la 'forme' de votre rendement a changé avec le produit.")

        # --- AFFICHAGE KPIs ---
        st.markdown(f"### 📈 Synthèse de Performance ({mode_analyse})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N Produit", f"{n_p} pts")
        c2.metric("N Témoin", f"{n_t} pts")
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
            st.header("🔬 Expertise Statistique Avancée")
            if n_p > 3 and n_t > 3:
                # Calcul D de Cohen
                std_p, std_t = data_p.std(), data_t.std()
                pooled_std = np.sqrt(((n_p - 1) * std_p**2 + (n_t - 1) * std_t**2) / (n_p + n_t - 2))
                d_cohen = (data_p.mean() - data_t.mean()) / pooled_std
                
                # Interface Expert
                st.subheader("1️⃣ Diagnostics de validité")
                d1, d2, d3 = st.columns(3)
                d1.write(f"**Normalité** : {'✅' if p_shapiro > 0.05 else '❌'} (p={round(p_shapiro,4)})")
                d2.write(f"**Homogénéité** : {'✅' if p_levene > 0.05 else '❌'} (p={round(p_levene,4)})")
                d3.write(f"**Taille de l'effet** : {round(d_cohen, 2)} (Cohen's D)")

                st.markdown("---")
                
                st.subheader("2️⃣ Verdict Scientifique")
                p_format = f"{p_val:.4e}" if p_val > 0 else "< 1e-20"
                
                if p_val < 0.05:
                    st.success(f"✅ **Impact Significatif** (p={p_format})")
                    st.write(f"Le test de **{test_nom}** confirme que la différence n'est pas due au hasard.")
                else:
                    st.error(f"❌ **Impact Non Démontré** (p={round(p_val,4)})")
                    st.write("La variabilité de la parcelle masque l'effet potentiel du produit.")

                # Courbe ECDF
                st.plotly_chart(px.ecdf(df_final, x="rdt", color="grp", title="Analyse de structure (K-S)"), use_container_width=True)
                
                # Tableau Annexe
                with st.expander("📝 Détails pour le mémoire"):
                    st.table(pd.DataFrame({
                        "Test": ["Shapiro", "Levene", "Comparaison", "Structure (K-S)"],
                        "P-Value": [p_shapiro, p_levene, p_val, p_ks],
                        "Rôle": ["Normalité", "Stabilité", "Performance", "Répartition"]
                    }))
            else:
                st.error("Données insuffisantes.")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
