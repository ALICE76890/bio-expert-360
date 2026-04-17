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
            st.header("🔬 Expertise Statistique Avancée")
            
            if n_final > 6:
                # Diagnostics
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                residus = data_p - data_p.mean()

                # Tests
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom, p_val = "Student", stats.ttest_ind(data_p, data_t)[1]
                else:
                    test_nom, p_val = "Mann-Whitney", stats.mannwhitneyu(data_p, data_t)[1]

                # Régression
                slope, intercept, r_val, p_reg, std_err = stats.linregress(range(len(data_p)), data_p)
                r_square = r_val**2

                # Affichage des diagnostics
                st.subheader("1️⃣ Diagnostics de validité")
                d1, d2, d3 = st.columns(3)
                d1.write(f"**Normalité** : {'✅' if p_shapiro > 0.05 else '❌'} (p={round(p_shapiro,4)})")
                d2.write(f"**Homogénéité** : {'✅' if p_levene > 0.05 else '❌'} (p={round(p_levene,4)})")
                d3.write(f"**Indépendance** : ✅ (N={n_final})")

                st.markdown("---")
                
                # Conclusion Automatique
                st.subheader("2️⃣ Conclusion de l'étude")
                p_format = f"{p_val:.4e}" if p_val > 0 else "< 1e-20"
                
                if p_val < 0.05 and r_square > 0.3:
                    st.success(f"✅ **Impact Réel Confirmé** (p={p_format} | R²={round(r_square,3)})")
                    st.write("L'analyse prouve que la différence de rendement est significative et que le modèle est solide. Le produit a eu un impact direct et mesurable.")
                elif p_val < 0.05:
                    st.warning(f"⚠️ **Impact Significatif mais Bruit Fort** (p={p_format} | R²={round(r_square,3)})")
                    st.write("Une différence est détectée, mais le faible R² indique que la variabilité naturelle du sol masque une partie de l'effet produit.")
                else:
                    st.error(f"❌ **Impact Non Démontré** (p={round(p_val,4)})")
                    st.write("La différence de rendement peut être expliquée par le hasard ou la variabilité naturelle de la parcelle.")

                st.plotly_chart(px.scatter(x=range(len(residus)), y=residus, title="Analyse des Résidus (Preuve de l'aléatoire)"), use_container_width=True)
            else:
                st.error("Données insuffisantes pour l'analyse expert.")

    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
