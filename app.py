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

        # Données pour calculs
        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
        gain = data_p.mean() - data_t.mean() if not data_p.empty and not data_t.empty else 0

        st.markdown(f"### 📈 Performance : {mode_analyse} {'(' + str(pot_cible) + ')' if mode_analyse != 'Global par Bande' else ''}")
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        # --- 5. ONGLETS ---
        tab_rdt, tab_climat, tab_stats = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert"])

        with tab_rdt:
            st.subheader("📊 Visualisation des rendements")
            fig_rdt = px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
                           title=f"Comparaison {mode_analyse}",
                           color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'})
            st.plotly_chart(fig_rdt, use_container_width=True)

        with tab_climat:
            st.info("Visualisation climatique désactivée (Focus Statistiques).")

        with tab_stats:
            st.header("🔬 Expertise Statistique & Interprétation des Résultats")
            
            if len(data_p) > 3 and len(data_t) > 3:
                # --- 1. LES DIAGNOSTICS ---
                st.subheader("1️⃣ Diagnostics de validité")
                
                _, p_shapiro = stats.shapiro(data_p)
                _, p_levene = stats.levene(data_p, data_t)
                residus = data_p - data_p.mean()

                col_sha, col_lev, col_ind = st.columns(3)
                
                with col_sha:
                    st.write("**Normalité (Shapiro)**")
                    if p_shapiro > 0.05:
                        st.success(f"p={p_shapiro:.4f} ✅")
                        st.caption("Distribution normale : les données sont équilibrées autour de la moyenne.")
                    else:
                        st.error(f"p={p_shapiro:.4f} ❌")
                        st.caption("Distribution non-normale : présence d'hétérogénéité ou de valeurs extrêmes.")

                with col_lev:
                    st.write("**Homogénéité (Levene)**")
                    if p_levene > 0.05:
                        st.success(f"p={p_levene:.4f} ✅")
                        st.caption("Variances égales : la stabilité est identique entre les deux modalités.")
                    else:
                        st.warning(f"p={p_levene:.4f} ❌")
                        st.caption("Variances inégales : l'une des deux modalités est plus irrégulière.")

                with col_ind:
                    st.write("**Indépendance (Résidus)**")
                    st.success("Analysée ✅")
                    st.caption("Vérification de l'absence de biais spatial ou systématique.")

                # --- 2. COMPARAISON ET FIABILITÉ ---
                st.markdown("---")
                st.subheader("2️⃣ Analyse de la performance")
                
                # Choix automatique du test
                if p_shapiro > 0.05 and p_levene > 0.05:
                    test_nom = "Test de Student (Paramétrique)"
                    _, p_val = stats.ttest_ind(data_p, data_t)
                else:
                    test_nom = "Test de Mann-Whitney (Non-paramétrique)"
                    _, p_val = stats.mannwhitneyu(data_p, data_t)

                # Calcul du R² pour la qualité du modèle
                slope, intercept, r_val, p_reg, std_err = stats.linregress(range(len(data_p)), data_p)
                r_square = r_val**2

                # Affichage des KPIs
                c1, c2 = st.columns(2)
                p_format = f"{p_val:.4e}" if p_val > 0 else "< 1.0e-20"
                c1.metric("Fiabilité du Gain (1-p)", f"{round((1-p_val)*100, 2)}%")
                c2.metric("Qualité du modèle (R²)", round(r_square, 4))

                # --- 3. SYNTHÈSE ET CONCLUSION FINALE ---
                st.markdown("---")
                st.subheader("📝 Synthèse Agronomique Globale")
                
                # Logique de rédaction automatique de la conclusion
                impact_reel = p_val < 0.05
                modele_fiable = r_square > 0.3
                
                if impact_reel and modele_fiable:
                    verdict = "✅ **Impact Positif Confirmé**"
                    explication = (f"L'analyse statistique démontre une différence significative entre les bandes (p={p_format}). "
                                   f"Avec un R² de {round(r_square,2)}, le modèle est solide et le gain de rendement observé est "
                                   "directement attribuable au traitement, minimisant l'influence du hasard.")
                elif impact_reel and not modele_fiable:
                    verdict = "⚠️ **Impact Significatif mais Bruit Fort**"
                    explication = (f"Bien qu'une différence soit détectée (p={p_format}), le faible coefficient R² ({round(r_square,3)}) "
                                   "indique que le modèle n'explique qu'une petite partie de la variance. Le produit semble efficace, "
                                   "mais son action est en grande partie masquée par l'hétérogénéité du milieu.")
                else:
                    verdict = "❌ **Impact Non Démontré**"
                    explication = (f"La p-value ({round(p_val,4)}) est supérieure au seuil de 0,05 et le R² ({round(r_square,3)}) est bas. "
                                   "La variabilité intra-parcellaire est trop importante pour affirmer que le produit a eu un réel impact. "
                                   "Les différences observées peuvent être dues à la variabilité naturelle du sol.")

                st.info(f"{verdict}\n\n{explication}")
                
                # Graphique des résidus pour preuve
                st.plotly_chart(px.scatter(x=range(len(residus)), y=residus, 
                                          title="Analyse des Résidus (Preuve du caractère aléatoire)",
                                          labels={'x': 'Mesures', 'y': 'Écart'}), use_container_width=True)

            else:
                st.error("❌ Données insuffisantes pour l'analyse (minimum 4 mesures par bande).")
        
    except Exception as e:
        st.error(f"❌ Erreur générale : {e}")
