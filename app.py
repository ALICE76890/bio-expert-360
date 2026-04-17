import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from scipy import stats
import numpy as np

# ... (garder les fonctions d'import et de nettoyage IQR 1.2 précédentes) ...

with tab_stats:
    st.header("🔬 Audit de Décision Statistique")
    
    if n_p > 3 and n_t > 3:
        # --- ETAPE 1 : DIAGNOSTIC RÉEL ---
        stat_sha, p_shapiro = stats.shapiro(data_p)
        stat_lev, p_levene = stats.levene(data_p, data_t)
        
        # --- ETAPE 2 : ARBRE DE DÉCISION ---
        if p_shapiro > 0.05 and p_levene > 0.05:
            # Cas idéal
            test_nom = "Student (T-test)"
            res_test = stats.ttest_ind(data_p, data_t)
            p_val = res_test[1]
            raison = "Données conformes aux lois normales : calcul sur les moyennes."
        elif p_shapiro > 0.05:
            # Variances diff
            test_nom = "Welch (T-test)"
            res_test = stats.ttest_ind(data_p, data_t, equal_var=False)
            p_val = res_test[1]
            raison = "Variabilité inégale entre bandes : correction de Welch appliquée."
        else:
            # Pas normal
            test_nom = "Mann-Whitney (U-test)"
            res_test = stats.mannwhitneyu(data_p, data_t)
            p_val = res_test[1]
            raison = "Distribution atypique (non-normale) : calcul sur les rangs (plus robuste)."

        # --- ETAPE 3 : RÉSULTATS ---
        st.subheader(f"Statut : {test_nom}")
        st.info(f"**Pourquoi ce test ?** {raison}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fiabilité réelle", f"{round((1-p_val)*100, 2)}%")
        
        # Taille de l'effet (D de Cohen)
        pooled_std = np.sqrt(((n_p - 1) * data_p.std()**2 + (n_t - 1) * data_t.std()**2) / (n_p + n_t - 2))
        d_cohen = (data_p.mean() - data_t.mean()) / pooled_std
        c2.metric("Force de l'impact", round(d_cohen, 2))
        
        # R² de la régression
        slope, intercept, r_val, p_reg, std_err = stats.linregress(range(len(data_p)), data_p)
        r_2 = r_val**2
        c3.metric("Solidité du modèle (R²)", round(r_2, 3))

        st.markdown("---")
        st.subheader("📝 Interprétation pour le Jury")
        
        if p_val < 0.05:
            if r_2 < 0.2:
                st.warning(f"**Verdict :** Le gain est **réel** (p < 0.05) mais le produit est 'subi' par la parcelle. Le faible R² ({round(r_2,2)}) prouve que l'hétérogénéité du sol domine l'essai, même si le produit tire son épingle du jeu.")
            else:
                st.success(f"**Verdict :** Le gain est **réel et solide**. Le produit explique une part importante de la performance.")
        else:
            st.error("**Verdict :** Aucun impact prouvé. La variabilité naturelle explique tout le résultat.")

        # Rappel des seuils pour prouver que c'est pas du "pipo"
        with st.expander("🔍 Voir les seuils mathématiques appliqués"):
            st.write(f"- Seuil de Normalité (Shapiro) : cible > 0.05 | Obtenu : {round(p_shapiro, 4)}")
            st.write(f"- Seuil d'Homogénéité (Levene) : cible > 0.05 | Obtenu : {round(p_levene, 4)}")
            st.write(f"- Seuil de Significativité : cible < 0.05 | Obtenu : {p_val:.4e}")

    else:
        st.error("Données insuffisantes.")
