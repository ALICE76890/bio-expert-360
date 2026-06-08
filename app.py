import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
 
# ── Imports optionnels avancés ─────────────────────────────────────────────
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
 
try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTITEST = True
except ImportError:
    HAS_MULTITEST = False
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bio-Expert 360",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.verdict-sig   { background:#d4edda; border-left:4px solid #28a745; padding:12px 16px; border-radius:6px; color:#155724; }
.verdict-nosig { background:#f8d7da; border-left:4px solid #dc3545; padding:12px 16px; border-radius:6px; color:#721c24; }
.method-box    { background:#e8f4fd; border-left:4px solid #0077b6; padding:10px 14px; border-radius:6px; font-size:.9rem; }
.vulgarisation { background:#f9f9f9; border-left:4px solid #6c757d; padding:12px; margin-bottom:10px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)
 
 
def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. RÉFÉRENTIEL ARVALIS
# ══════════════════════════════════════════════════════════════════════════════
PARAM_CULTURES = {
    "Blé Tendre": {"echaudage": 25, "critique": 30, "base_t": 0},
    "Maïs":       {"echaudage": 35, "critique": 38, "base_t": 6},
    "Orge":       {"echaudage": 25, "critique": 30, "base_t": 0},
}
 
ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}
CORRECTION_METHODS = {
    "Aucune": None,
    "Holm-Šídák (recommandé)": "holm-sidak",
    "Benjamini-Hochberg (FDR)": "fdr_bh",
    "Bonferroni (conservatif)": "bonferroni",
}
N_BOOTSTRAP = 5000
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    st.caption("Protocole essais en grand bandes — v2.0")
 
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
        st.caption("Colonnes attendues : `bande`, `rdt`, `potentiel` (optionnel), `bloc` (optionnel)")
 
    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture  = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis  = st.date_input("Date de Semis",       datetime(2024, 10, 20))
        d_appli  = st.date_input("Date d'Application",  datetime(2025,  3, 10))
        d_recolt = st.date_input("Date de Récolte",     datetime(2025,  7, 15))
        alpha    = st.selectbox("Seuil de significativité α", list(ALPHA_LEVELS.keys()))
        alpha_v  = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage strict (IQR 1.2)", value=True)
 
    with st.expander("📊 OPTIONS STATISTIQUES", expanded=True):
        corr_method  = st.selectbox("Correction tests multiples", list(CORRECTION_METHODS.keys()))
        n_boot       = st.select_slider("Itérations Bootstrap CI", [1000, 2000, 5000, 10000], value=N_BOOTSTRAP)
        run_anova    = st.checkbox("ANOVA complète (si potentiel disponible)", value=True)
        run_mixed    = st.checkbox("Modèle mixte (si bloc disponible)", value=True)
 
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod  = st.number_input("Coût Produit (€/ha)", value=45)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════════
 
def bootstrap_ci(data, stat_fn=np.mean, n=5000, ci=0.95):
    """Intervalle de confiance bootstrap BCa."""
    rng = np.random.default_rng(42)
    boot_stats = np.array([stat_fn(rng.choice(data, len(data), replace=True)) for _ in range(n)])
    lo = (1 - ci) / 2
    return np.percentile(boot_stats, [lo * 100, (1 - lo) * 100]), boot_stats
 
 
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
 
 
def interpret_d(d):
    d = abs(d)
    if d < 0.2:  return "négligeable"
    if d < 0.5:  return "faible"
    if d < 0.8:  return "moyen"
    if d < 1.2:  return "fort"
    return "très fort"
 
 
def power_estimate(d, n1, n2, alpha=0.05):
    """Puissance approximative (test bilatéral, Welch)."""
    try:
        from scipy.stats import norm
        se  = np.sqrt(1/n1 + 1/n2)
        ncp = abs(d) / se
        z_a = norm.ppf(1 - alpha/2)
        return 1 - norm.cdf(z_a - ncp) + norm.cdf(-z_a - ncp)
    except Exception:
        return None
 
 
def run_statistical_tests(data_p, data_t, alpha_v=0.05, n_boot=5000):
    """Batterie de tests complète. Retourne un dict structuré."""
    result = {}
    n_p, n_t = len(data_p), len(data_t)
 
    # ── Normalité & homogénéité ────────────────────────────────────────────
    _, p_shap_p = stats.shapiro(data_p) if n_p >= 3 else (None, None)
    _, p_shap_t = stats.shapiro(data_t) if n_t >= 3 else (None, None)
    _, p_lev    = stats.levene(data_p, data_t)
    ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
 
    result['diagnostics'] = {
        'shapiro_p': (p_shap_p, p_shap_t),
        'levene_p': p_lev,
        'ks_stat': ks_stat, 'ks_p': p_ks,
    }
 
    # ── Choix du test principal ─────────────────────────────────────────────
    normal = (p_shap_p or 0) > alpha_v and (p_shap_t or 0) > alpha_v
    homog  = p_lev > alpha_v
    if normal and homog:
        t_stat, p_main = stats.ttest_ind(data_p, data_t)
        test_nom, test_id = "Student (paramétrique)", "PARAM"
    elif normal:
        t_stat, p_main = stats.ttest_ind(data_p, data_t, equal_var=False)
        test_nom, test_id = "Welch (variances inégales)", "WELCH"
    else:
        t_stat, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
        test_nom, test_id = "Mann-Whitney U (non-paramétrique)", "NP"
 
    result['main_test'] = {
        'name': test_nom, 'id': test_id,
        'stat': t_stat, 'p': p_main,
    }
 
    # ── Bootstrap CI ───────────────────────────────────────────────────────
    ci_p, boot_p = bootstrap_ci(data_p.values, n=n_boot)
    ci_t, boot_t = bootstrap_ci(data_t.values, n=n_boot)
    boot_diff    = boot_p - boot_t
    ci_diff      = np.percentile(boot_diff, [2.5, 97.5])
 
    result['bootstrap'] = {
        'ci_p': ci_p, 'ci_t': ci_t,
        'ci_diff': ci_diff, 'boot_diff': boot_diff,
        'mean_p': np.mean(boot_p), 'mean_t': np.mean(boot_t),
    }
 
    # ── Taille d'effet ─────────────────────────────────────────────────────
    d   = cohen_d(data_p.values, data_t.values)
    pwr = power_estimate(d, n_p, n_t, alpha_v)
 
    result['effect'] = {
        'd': d, 'label': interpret_d(d), 'power': pwr,
    }
 
    return result
 
 
def run_anova_analysis(df_final, alpha_v=0.05):
    """ANOVA à deux facteurs si 'potentiel' comporte plusieurs niveaux. Bloque si unique."""
    if not HAS_STATSMODELS:
        return None, "statsmodels non installé", None, False
 
    if 'potentiel' not in df_final.columns:
        return None, "Colonne 'potentiel' absente du fichier.", None, False
 
    nb_zones = df_final['potentiel'].dropna().nunique()
 
    if nb_zones == 1:
        seule_zone = df_final['potentiel'].dropna().iloc[0]
        msg_erreur = f"Impossible de réaliser l'ANOVA : une seule zone de potentiel détectée ('{seule_zone}'). Pour croiser le traitement avec le milieu, il faut au moins 2 zones différentes."
        return None, msg_erreur, None, False
        
    if nb_zones == 0:
        return None, "Aucune donnée de potentiel valide trouvée.", None, False
 
    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    title   = "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel"
 
    try:
        model   = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, title, model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {str(e)}", None, False 
 
 
def run_mixed_model(df_final):
    """Modèle mixte avec 'bloc' comme effet aléatoire."""
    if not HAS_STATSMODELS:
        return None, "statsmodels non installé ou indisponible."
        
    if 'bloc' not in df_final.columns:
        return None, "Colonne 'bloc' absente du fichier QGIS. Impossible de construire le modèle mixte."
        
    if df_final['bloc'].dropna().nunique() < 2:
        return None, "Données de blocs insuffisantes (< 2 blocs uniques trouvés). Le modèle mixte nécessite au moins 2 répétitions distinctes."
 
    try:
        model = smf.mixedlm("rdt ~ C(grp)", df_final, groups=df_final["bloc"]).fit(reml=True)
        return model, None
    except Exception as e:
        return None, f"Erreur de convergence ou de calcul du modèle mixte : {str(e)}"
 
 
def apply_correction(p_values_dict, method_key, alpha_v):
    """Correction Bonferroni / Holm / BH sur un dict de p-values."""
    if not HAS_MULTITEST or not method_key:
        return {k: {'p_raw': v, 'p_adj': v, 'reject': v < alpha_v} for k, v in p_values_dict.items()}
    
    keys   = list(p_values_dict.keys())
    pvals  = [p_values_dict[k] for k in keys]
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha_v, method=method_key)
    return {k: {'p_raw': pvals[i], 'p_adj': p_adj[i], 'reject': reject[i]} for i, k in enumerate(keys)}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. LECTURE FICHIER & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
    st.markdown("""
    **Colonnes attendues dans le fichier .shp / attributs QGIS :**
 
    | Colonne | Obligatoire | Description |
    |---------|-------------|-------------|
    | `bande` | ✅ | Identifiant de bande (ex: A, B, Produit, Témoin) |
    | `rdt` | ✅ | Rendement (qtx/ha ou t/ha) |
    | `potentiel` | ⚙️ recommandé | Zone de potentiel sol (ex: Faible, Moyen, Fort) |
    | `bloc` | ⚙️ optionnel | Identifiant de bloc expérimental |
    """)
    st.stop()
 
# ── Chargement ────────────────────────────────────────────────────────────────
try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")
 
    shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé dans le zip.")
        st.stop()
 
    # Lecture brute et forçage de la projection géographique en degrés GPS
    gdf_raw = gpd.read_file(os.path.join("temp", shp_files[0]))
    if gdf_raw.crs is None:
        gdf_raw.crs = "EPSG:2154" # Par défaut si absent (Lambert-93 France)
    gdf = gdf_raw.to_crs(epsg=4326)
    
    df  = pd.DataFrame(gdf.drop(columns='geometry'))
    df.columns = df.columns.str.lower().str.strip()
 
    # Validation colonnes obligatoires
    missing = [c for c in ['bande', 'rdt'] if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}")
        st.stop()
 
    df['rdt'] = pd.to_numeric(df['rdt'], errors='coerce')
    df = df.dropna(subset=['rdt'])
 
except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()
 
# ── Options dynamiques sidebar ────────────────────────────────────────────────
with st.sidebar:
    with st.expander("🔬 NIVEAU D'ANALYSE", expanded=True):
        mode_analyse = st.radio("Affichage", ["Global par Bande", "Détaillé par Potentiel"])
        pot_cible = "Tous"
        if mode_analyse == "Détaillé par Potentiel":
            if 'potentiel' in df.columns:
                liste_pot  = ["Tous"] + sorted(list(df['potentiel'].dropna().unique()))
                pot_cible  = st.selectbox("Zone de potentiel", liste_pot)
            else:
                st.warning("⚠️ Colonne 'potentiel' absente.")
                mode_analyse = "Global par Bande"
 
    bandes_dispo = sorted(df['bande'].unique().tolist())
    val_p = st.selectbox("Bande = 'Produit' ?", bandes_dispo)
 
# ── Groupement & filtrage ─────────────────────────────────────────────────────
df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
df_travail = df.copy()
if mode_analyse == "Détaillé par Potentiel" and pot_cible != "Tous":
    df_travail = df[df['potentiel'] == pot_cible].copy()
 
# ── Nettoyage IQR ─────────────────────────────────────────────────────────────
n_initial = len(df_travail)
if clean_iqr:
    clean_list = []
    for g in ['Produit', 'Témoin']:
        sub = df_travail[df_travail['grp'] == g]
        if not sub.empty:
            q1, q3 = sub['rdt'].quantile([0.25, 0.75])
            iqr     = q3 - q1
            sub     = sub[(sub['rdt'] >= q1 - 1.2 * iqr) & (sub['rdt'] <= q3 + 1.2 * iqr)]
            clean_list.append(sub)
    df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
else:
    df_final = df_travail.copy()
 
n_removed = n_initial - len(df_final)
data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
n_p, n_t = len(data_p), len(data_t)
 
has_enough = n_p > 3 and n_t > 3
gain = data_p.mean() - data_t.mean() if has_enough else 0.0
marge = ((gain / 10) * prix_vente) - cout_prod
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. KPIs PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 Synthèse · {mode_analyse}" +
            (f" · Zone *{pot_cible}*" if pot_cible != "Tous" else ""))
 
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Obs. Produit",  f"{n_p}")
c2.metric("Obs. Témoin",   f"{n_t}")
c3.metric("Moy. Produit",  f"{data_p.mean():.1f} qtx" if has_enough else "—")
c4.metric("Moy. Témoin",   f"{data_t.mean():.1f} qtx" if has_enough else "—")
c5.metric("Gain Moyen",    f"+{gain:.2f} qtx" if has_enough else "—")
c6.metric("Marge Nette",   f"{marge:.0f} €/ha" if has_enough else "—")
if n_removed > 0:
    st.caption(f"⚠️ {n_removed} points supprimés par nettoyage IQR 1.2 sur {n_initial} observations.")
 
 
if not has_enough:
    st.error("Données insuffisantes (< 4 obs. par groupe). Vérifiez votre fichier ou les filtres.")
    st.stop()
 
# ── Calculs statistiques ──────────────────────────────────────────────────────
with st.spinner("Calculs statistiques en cours…"):
    stat_res = run_statistical_tests(data_p, data_t, alpha_v=alpha_v, n_boot=n_boot)
 
p_main = stat_res['main_test']['p']
d_val  = stat_res['effect']['d']
pwr    = stat_res['effect']['power']
ci_diff = stat_res['bootstrap']['ci_diff']
 
# ── Verdict principal ────────────────────────────────
sig  = p_main < alpha_v
html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
<strong>{'✅ Impact Significatif' if sig else '❌ Impact Non Démontré'}</strong> 
— {stat_res['main_test']['name']} · p = {p_main:.4f} · Cohen's d = {d_val:.2f} ({stat_res['effect']['label']})
{'<br>L\'effet n\'est pas dû au hasard avec une confiance ≥ '+str(int((1-alpha_v)*100))+'%.' if sig else '<br>La variabilité de la parcelle masque l\'effet potentiel du produit.'}
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7. ONGLETS ENRICHIS D'EXPLICATIONS PEDAGOGIQUES
# ══════════════════════════════════════════════════════════════════════════════
tab_rdt, tab_boot, tab_anova, tab_mix, tab_corr, tab_map = st.tabs([
    "📊 Distribution & Tests",
    "🎲 Bootstrap & IC",
    "📐 ANOVA Spatial",
    "🔀 Modèle Mixte",
    "🔁 Tests Multiples",
    "🗺️ Carte parcelle",
])
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution & Validation
# ─────────────────────────────────────────────────────────────────────────────
with tab_rdt:
    st.markdown("""
    <div class="vulgarisation">
    💡 <b>Comprendre cette page :</b> Avant de regarder si le produit fonctionne, on vérifie la "tête" de vos données. Les graphiques montrent comment se répartissent vos points. Les tests mathématiques du bas vérifient si vos données sont stables ou trop chaotiques, ce qui permet à l'application de choisir le test de comparaison le plus juste.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(
            df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Distribution des rendements avec encoche de confiance"
        )
        fig_box.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig_box, use_container_width=True)
 
    with col2:
        fig_viol = px.violin(
            df_final, x="grp", y="rdt", color="grp", box=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Densité de probabilité (Forme de la répartition)"
        )
        st.plotly_chart(fig_viol, use_container_width=True)
 
    desc = df_final.groupby('grp')['rdt'].describe().round(2)
    desc.columns = ['N (Nb points)', 'Moyenne', 'Écart-type (Variabilité)', 'Min', 'Q25', 'Médiane', 'Q75', 'Max']
    st.dataframe(desc, use_container_width=True)
 
    st.subheader("📋 Les verrous de sécurité mathématiques")
    p_sp, p_st = stat_res['diagnostics']['shapiro_p']
    p_lev = stat_res['diagnostics']['levene_p']
 
    d1, d2, d3 = st.columns(3)
    d1.metric("Shapiro-Wilk (Produit)", f"p = {p_sp:.4f}", "✅ En cloche" if p_sp > alpha_v else "⚠️ Asymétrique")
    d2.metric("Shapiro-Wilk (Témoin)",  f"p = {p_st:.4f}", "✅ En cloche" if p_st > alpha_v else "⚠️ Asymétrique")
    d3.metric("Levene (Variances)", f"p = {p_lev:.4f}", "✅ Risque Homogène" if p_lev > alpha_v else "⚠️ Hétérogène")
 
    with st.expander("🔍 Guide de vulgarisation : C'est quoi la p-value, Shapiro et Levene ?"):
        st.markdown(f"""
        * **La p-value (p) :** C'est la jauge du hasard. Plus elle est petite (inférieure à votre seuil de {alpha_v}), plus on est sûr que le hasard n'y est pour rien. Si $p = 0.001$, il n'y avait qu'une chance sur 1000 que ce résultat arrive par chance.
        * **Test de Shapiro-Wilk (Normalité) :** Il vérifie si vos rendements suivent une courbe naturelle "en cloche". 
            * *Si $p > {alpha_v}$ (Normal) :* Vos données sont bien réparties autour de la moyenne.
            * *Si $p < {alpha_v}$ (Non-normal) :* Il y a des paquets de points bizarres ou trop étalés.
        * **Test de Levene (Homogénéité) :** Il mesure si le "bruit de fond" ou l'instabilité est la même dans la bande Produit et dans la bande Témoin. 
        * **Le choix automatique de l'algorithme :** * Si tout est au vert (Normal et Homogène), l'application utilise le test de **Student**, le plus puissant.
            * Si les données sont instables ou asymétriques, elle bascule automatiquement sur **Welch** ou **Mann-Whitney**, qui sont des barrières de sécurité pour éviter de valider un faux résultat.
        """)
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bootstrap & IC
# ─────────────────────────────────────────────────────────────────────────────
with tab_boot:
    st.markdown("""
    <div class="vulgarisation">
    🎲 <b>Comprendre cette page :</b> Imaginez qu'on puisse cloner votre parcelle 5 000 fois et refaire la récolte à chaque fois pour voir ce qu'il se passe. C'est exactement ce que fait le <b>Bootstrap</b> par ordinateur. Il mélange vos points réels pour mesurer la solidité du gain.
    </div>
    """, unsafe_allow_html=True)

    st.subheader(f"Intervalles de confiance Bootstrap ({n_boot:,} simulations)")
 
    boot_res = stat_res['bootstrap']
    ci_p, ci_t = boot_res['ci_p'], boot_res['ci_t']
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Rendement attendu Produit", f"[{ci_p[0]:.2f} à {ci_p[1]:.2f}] qtx", f"Moyenne = {data_p.mean():.2f}")
    with col_b:
        st.metric("Rendement attendu Témoin", f"[{ci_t[0]:.2f} à {ci_t[1]:.2f}] qtx", f"Moyenne = {data_t.mean():.2f}")
 
    ci_diff = boot_res['ci_diff']
    contains_zero = ci_diff[0] <= 0 <= ci_diff[1]
    st.metric(
        "VRAI GAIN NET (Intervalle de confiance de la différence)",
        f"[{ci_diff[0]:.2f} à {ci_diff[1]:.2f}] qtx/ha",
        "❌ Le gain peut être nul ou négatif (contient 0)" if contains_zero else "✅ Le gain est mathématiquement garanti (exclut 0)",
        delta_color="inverse" if contains_zero else "normal"
    )
 
    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(
        x=boot_res['boot_diff'], nbinsx=80, name="Gains simulés", marker_color='#3498db', opacity=0.7,
    ))
    fig_boot.add_vline(x=0, line_dash="dash", line_color="red",   annotation_text="Zone de Danger (Gain nul)")
    fig_boot.add_vline(x=gain, line_dash="dot",  line_color="green", annotation_text=f"Gain réel observé = {gain:.2f}")
    fig_boot.add_vrect(x0=ci_diff[0], x1=ci_diff[1], fillcolor="#3498db", opacity=0.15, annotation_text="95% des parcelles virtuelles")
    fig_boot.update_layout(title="Où se situe le gain après 5 000 récoltes virtuelles ?", xaxis_title="Gain de rendement (qtx/ha)", yaxis_title="Nombre de simulations", height=400)
    st.plotly_chart(fig_boot, use_container_width=True)
 
    st.subheader("⚡ Puissance de votre essai (Rigueur du protocole)")
    if pwr is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Intensité de l'effet (Cohen's d)", f"{d_val:.2f}", stat_res['effect']['label'])
        p2.metric("Fiabilité du test (Puissance)", f"{pwr:.1%}", "Essai Robuste (≥ 80%)" if pwr >= 0.8 else "Essai trop petit/brouillon (< 80%)")
        p3.metric("Verdict sur la taille de l'effet", stat_res['effect']['label'].upper())
 
    with st.expander("🔍 Guide de vulgarisation : Pourquoi faire 5 000 simulations (Bootstrap) ?"):
        st.markdown("""
        * **Le piège de la simple moyenne :** Si votre bande Produit fait en moyenne $+2 \text{ qtx}$ de plus que le Témoin, est-ce grâce au produit, ou est-ce juste parce que la bande était placée sur un meilleur morceau de terre ?
        * **La réponse du Vrai Gain :** L'intervalle de confiance de la différence vous donne la fourchette de sécurité. 
            * Si l'intervalle est de `[-0.5 à +4.5]`, la moyenne est positive, mais l'ordinateur vous dit : *"Attention, au vu du désordre dans vos points, il est possible que le vrai gain soit de 0 ou négatif"*. C'est **non significatif**.
            * Si l'intervalle est de `[+1.1 à +3.9]`, le zéro est exclu. Vous êtes sûr à 95% que le produit apporte *au moins* $+1.1 \text{ qtx}$. C'est **gagné**.
        * **La Puissance ($1-\beta$) :** C'est la capacité de votre essai à détecter le bonus du produit. Si votre essai est trop encombré par les mauvaises herbes ou les cailloux (bruit), la puissance s'effondre sous les 80%, signifiant que l'essai n'est pas assez propre pour conclure.
        """)
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOVA Spatial
# ─────────────────────────────────────────────────────────────────────────────
with tab_anova:
    st.markdown("""
    <div class="vulgarisation">
    📐 <b>Comprendre cette page :</b> Une parcelle n'est jamais uniforme (zones de cailloux, fonds de vallons fertiles). L'<b>ANOVA</b> est un découpeur de variabilité : elle sépare ce qui revient à la qualité naturelle de votre sol de ce qui revient à l'effet réel de votre produit.
    </div>
    """, unsafe_allow_html=True)

    if not run_anova:
        st.info("ANOVA désactivée. Activez-la dans les options statistiques (sidebar).")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : ajoutez-le à votre fichier requirements.txt")
    else:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
 
        if anova_table is None:
            st.error(f"❌ {anova_title}")
            st.info("💡 Note : Les autres tests restent disponibles. L'ANOVA a été coupée pour préserver la cohérence de vos calculs.")
        else:
            st.subheader(anova_title)
            
            at = anova_table.copy()
            at.columns = [c.replace('PR(>F)', 'p-value').replace('sum_sq', 'SCE (Part de responsabilité)').replace('mean_sq', 'CME') for c in at.columns]
            at = at.round(4)
            
            def style_pval(val):
                try:
                    v = float(val)
                    if v < 0.001: return 'background-color:#d4edda; font-weight:bold; color:#155724;'
                    if v < alpha_v:  return 'background-color:#fff3cd; color:#856404;'
                    return ''
                except: 
                    return ''
 
            target_cols = [c for c in at.columns if 'p-value' in c.lower() or 'PR' in c]
            styled_at = at.style.map(style_pval, subset=target_cols) if target_cols else at.style
            st.dataframe(styled_at, use_container_width=True)
 
            if anova_model is not None:
                r2 = anova_model.rsquared
                r2_adj = anova_model.rsquared_adj
                col1, col2, col3 = st.columns(3)
                col1.metric("Explication du champ (R²)", f"{r2:.1%}")
                col2.metric("R² ajusté", f"{r2_adj:.3f}")
                col3.metric("F-Value globale", f"{anova_model.fvalue:.2f}")
 
            if has_pot and 'potentiel' in df_final.columns:
                st.subheader("📊 Croisement : Rendement par Zone × Traitement")
                pivot = df_final.groupby(['potentiel', 'grp'])['rdt'].agg(['mean', 'std', 'count']).round(2)
                pivot.columns = ['Rendement Moyen', 'Écart-Type', 'Nombre de points']
                st.dataframe(pivot, use_container_width=True)
 
                fig_inter = px.box(
                    df_final, x="potentiel", y="rdt", color="grp",
                    color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
                    title="Le produit fonctionne-t-il mieux sur certaines zones de sol ?"
                )
                st.plotly_chart(fig_inter, use_container_width=True)
 
            with st.expander("🔍 Guide de vulgarisation : Comment lire le tableau d'ANOVA ?"):
                st.markdown(f"""
                L'ANOVA analyse trois sources de variations pour voir qui est "responsable" du rendement final :
                1.  **C(grp) [Le Traitement] :** Si sa p-value est inférieure à {alpha_v}, le produit fonctionne de manière générale sur l'ensemble du champ.
                2.  **C(potentiel) [Le Sol] :** Si sa p-value est très petite, cela prouve que vos zones cartographiées (ex: Faible, Moyen, Fort) décrivent bien la réalité du terrain.
                3.  **C(grp):C(potentiel) [L'Interaction] :** C'est le graal de l'agronomie de précision. 
                    * *Si cette p-value est sous le seuil ({alpha_v}) :* Le produit **change de comportement selon le sol**. Par exemple, il donne $+8 \text{ qtx}$ en potentiel faible mais $0 \text{ qtx}$ en potentiel fort. Vous devez adapter vos préconisations !
                * **Le R² (R-deux) :** Si votre $R^2 = {r2:.1%}$, cela signifie que votre carte de sol et votre traitement suffisent à expliquer {r2:.1%} de tout ce qui s'est passé dans la parcelle. Le reste ({10-r2*100:.1f}%) provient d'imprévus (ravageurs, ombres de haies, etc.).
                """)
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Modèle Mixte
# ─────────────────────────────────────────────────────────────────────────────
with tab_mix:
    st.markdown("""
    <div class="vulgarisation">
    🔀 <b>Comprendre cette page :</b> Si vous avez répété vos bandes à plusieurs endroits du champ (Bloc 1, Bloc 2, Répétition A, B...), le sol change entre ces blocs. Le <b>Modèle Mixte</b> agit comme un filtre : il nettoie la signature de fertilité propre à chaque bloc pour mesurer le pur effet biologique de votre intrant.
    </div>
    """, unsafe_allow_html=True)

    if not run_mixed:
        st.info("Modèle mixte désactivé. Activez-le dans les options statistiques (sidebar).")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : `pip install statsmodels`")
    else:
        mix_model, mix_err = run_mixed_model(df_final)
 
        if mix_model is not None:
            st.subheader("Résultats épurés des effets Blocs")
 
            fe = mix_model.fe_params.round(4)
            pv = mix_model.pvalues.round(4)
            ci_mix = mix_model.conf_int().round(4)
            res_fe = pd.DataFrame({
                'Variable': fe.index,
                'Efficacité pure calculée (qtx)': fe.values,
                'Borne basse IC': ci_mix.iloc[:, 0].values,
                'Borne haute IC': ci_mix.iloc[:, 1].values,
                'p-value': pv.values,
            })
            res_fe['Significatif ?'] = res_fe['p-value'].apply(lambda p: '✅ Oui, impact prouvé' if p < alpha_v else '❌ Non prouvé')
            st.dataframe(res_fe, use_container_width=True)
 
            re_var = mix_model.cov_re.values[0][0] if mix_model.cov_re is not None else None
            res_var = mix_model.scale
            col1, col2, col3 = st.columns(3)
            if re_var is not None:
                icc = re_var / (re_var + res_var)
                col1.metric("Bruit de fertilité (Inter-blocs)",  f"{re_var:.2f}")
                col2.metric("Erreur résiduelle",   f"{res_var:.2f}")
                col3.metric("Poids des blocs (ICC)", f"{icc:.1%}",
                            "Répétitions indispensables" if icc > 0.2 else "Blocs homogènes, peu d'effet")
 
            try:
                re_vals = mix_model.random_effects
                re_df   = pd.DataFrame({'Bloc': list(re_vals.keys()),
                                        'Hétérogénéité naturelle du bloc (qtx)': [v.values[0] for v in re_vals.values()]})
                fig_re = px.bar(re_df, x='Bloc', y='Hétérogénéité naturelle du bloc (qtx)',
                                title="Fertilité naturelle corrigée pour chaque bloc",
                                color='Hétérogénéité naturelle du bloc (qtx)', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_re, use_container_width=True)
            except Exception:
                pass
 
            with st.expander("🔍 Guide de vulgarisation : Effet Fixe vs Effet Aléatoire ?"):
                st.markdown("""
                * **L'Effet Fixe (Ce que l'on veut mesurer) :** C'est l'action biologique pure du Produit par rapport au Témoin. Le modèle extrait toutes les perturbations pour vous donner la valeur exacte du gain biologique.
                * **L'Effet Aléatoire (Le bloc) :** On considère que chaque zone ou répétition possède sa propre déviation naturelle (un bloc en bas de pente produira toujours plus qu'un bloc sur une crête séchante). Le modèle calcule ce "poids du milieu" (BLUP) et le soustrait de l'analyse pour que la comparaison Produit/Témoin soit équitable.
                * **L'ICC (Indicateur de structure) :** Si l'ICC est élevé (ex: $30\%$), cela signifie que $30\%$ des variations de rendement de votre champ étaient simplement dues à l'emplacement de vos blocs. Cela prouve que vous avez eu raison de mettre en place des répétitions !
                """)
        else:
            st.error(f"❌ {mix_err}")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Tests Multiples
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("Correction pour tests multiples")
    corr_key = CORRECTION_METHODS[corr_method]
    has_pot_col = 'potentiel' in df_final.columns and df_final['potentiel'].nunique() > 1
 
    if not has_pot_col:
        st.info("La correction pour tests multiples s'applique lorsque vous comparez le produit sur **plusieurs zones de potentiel** simultanément.")
    else:
        zones = sorted(df_final['potentiel'].dropna().unique())
        p_raw_dict = {}
        gain_dict  = {}
 
        for z in zones:
            sub = df_final[df_final['potentiel'] == z]
            dp  = sub[sub['grp'] == 'Produit']['rdt'].dropna()
            dt  = sub[sub['grp'] == 'Témoin']['rdt'].dropna()
            if len(dp) > 2 and len(dt) > 2:
                if corr_key:
                    _, p = stats.mannwhitneyu(dp, dt, alternative='two-sided')
                else:
                    _, p = stats.ttest_ind(dp, dt)
                p_raw_dict[z] = p
                gain_dict[z]  = dp.mean() - dt.mean()
 
        if p_raw_dict:
            corr_res = apply_correction(p_raw_dict, corr_key, alpha_v)
 
            rows = []
            for z, r in corr_res.items():
                rows.append({
                    'Zone potentiel': z,
                    'Gain (qtx)': round(gain_dict[z], 2),
                    'p brut (sans filtre)': round(r['p_raw'], 4),
                    'p corrigé (sécurisé)': round(r['p_adj'], 4),
                    'Verdict Scientifique': '✅ Vrai Gain Géo-localisé' if r['reject'] else '❌ Différence due au hasard',
                })
            df_corr = pd.DataFrame(rows)
            st.dataframe(df_corr, use_container_width=True)
 
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=df_corr['Zone potentiel'], y=df_corr['p brut (sans filtre)'], name='p-value non filtrée', marker_color='#3498db', opacity=0.6
            ))
            fig_corr.add_trace(go.Bar(
                x=df_corr['Zone potentiel'], y=df_corr['p corrigé (sécurisé)'], name='p-value sécurisée (Filtre anti-hasard)', marker_color='#e67e22', opacity=0.8
            ))
            fig_corr.add_hline(y=alpha_v, line_dash="dash", line_color="red", annotation_text=f"Seuil critique α = {alpha_v}")
            fig_corr.update_layout(barmode='group', title="Comparaison des p-values avant/après filtrage de sécurité", yaxis_title="Niveau de risque d'erreur", height=380)
            st.plotly_chart(fig_corr, use_container_width=True)
 
            fig_forest = go.Figure()
            for i, row in df_corr.iterrows():
                color = '#2ecc71' if 'Vrai' in row['Verdict Scientifique'] else '#e74c3c'
                fig_forest.add_trace(go.Scatter(
                    x=[row['Gain (qtx)']], y=[row['Zone potentiel']], mode='markers', marker=dict(size=14, color=color), name=row['Zone potentiel']
                ))
            fig_forest.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_forest.update_layout(title="Forest plot — Synthèse visuelle des gains nets par environnement", xaxis_title="Gain net (qtx/ha)", showlegend=False, height=max(250, len(zones) * 60))
            st.plotly_chart(fig_forest, use_container_width=True)
 
    with st.expander("🔍 Guide de vulgarisation : C'est quoi l'erreur des tests multiples ?"):
        st.markdown("""
        * **Le paradoxe du loto :** Si vous jouez une fois au loto, vous avez très peu de chances de gagner. Si vous achetez 100 tickets, vos chances augmentent. En statistiques, c'est pareil : si vous cherchez un effet du produit sur 5 ou 6 zones de potentiel différentes en même temps, vous allez finir par trouver une zone "significative" **par pur hasard**, simplement parce que vous multipliez les tentatives.
        * **À quoi servent les corrections (Holm / Bonferroni) ?** Elles font office de douane. Elles recalculent et durcissent le niveau d'exigence des p-values. Si un gain dans une zone était un "coup de chance", la p-value corrigée va remonter au-dessus du seuil d'erreur et le verdict passera en *Non significatif*. Cela évite de conseiller un produit à un agriculteur sur la base d'un faux positif.
        """)
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Carte parcelle
# ─────────────────────────────────────────────────────────────────────────────
with tab_map:
    if 'geometry' not in gdf.columns or gdf.empty:
        st.info("Géométrie non disponible ou fichier vide.")
    else:
        try:
            gdf_plot = gdf.copy()
            gdf_plot['grp'] = gdf_plot['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
            
            gdf_plot = gdf_plot.merge(
                df_final[['rdt']].assign(idx=df_final.index),
                left_index=True, right_on='idx', how='left',
                suffixes=('_brut', '_nettoye')
            )
            gdf_plot['rdt_carte'] = gdf_plot['rdt_nettoye'].fillna(gdf_plot['rdt_brut'])
 
            # Extraction des coordonnées géographiques stables calculées au chargement
            gdf_plot['lat'] = gdf_plot.geometry.centroid.y
            gdf_plot['lon'] = gdf_plot.geometry.centroid.x
 
            st.markdown("### 🗺️ Carte de Rendement Géoréférencée")
            mode_carte = st.radio(
                "Sélectionnez le niveau de détail de la carte :",
                ["Vue par point individuel", "Synthèse par Zone de Potentiel"], horizontal=True
            )
 
            center_lat = float(gdf_plot['lat'].median())
            center_lon = float(gdf_plot['lon'].mean())
 
            if mode_carte == "Synthèse par Zone de Potentiel" and 'potentiel' in gdf_plot.columns:
                df_pot_carte = gdf_plot.groupby('potentiel').agg({
                    'rdt_carte': 'mean', 'lat': 'mean', 'lon': 'mean'
                }).reset_index()
                df_pot_carte['rdt_moyen'] = df_pot_carte['rdt_carte'].round(1)
 
                fig_map = px.scatter_mapbox(
                    df_pot_carte, lat="lat", lon="lon", color="potentiel", size="rdt_moyen", size_max=22,
                    color_discrete_sequence=px.colors.qualitative.Bold, mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, hover_data={'potentiel': True, 'rdt_moyen': True},
                    labels={'rdt_moyen': 'Rdt Moyen (qtx/ha)', 'potentiel': 'Zone'}, title="Rendement Moyen par Zone de Potentiel"
                )
            else:
                if mode_carte == "Synthèse par Zone de Potentiel":
                    st.warning("⚠️ Colonne 'potentiel' absente : affichage individuel uniquement.")
 
                fig_map = px.scatter_mapbox(
                    gdf_plot, lat="lat", lon="lon", color="rdt_carte", size="rdt_carte", size_max=12,
                    color_continuous_scale="RdYlGn", mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, opacity=0.8,
                    hover_data={'bande': True, 'rdt_carte': ':.1f'}, labels={'rdt_carte': 'Rendement (qtx/ha)'},
                    title="Carte de rendement spatiale par point"
                )
            
            fig_map.update_layout(height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Carte indisponible : {e}")
 
# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPORT RAPPORT
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Export des résultats")
 
report_lines = [
    f"# Rapport Bio-Expert 360 — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    f"**Culture** : {culture}  |  **Date récolte** : {d_recolt}",
    f"**Mode** : {mode_analyse}" + (f" — Zone : {pot_cible}" if pot_cible != "Tous" else ""),
    "",
    "## Résultats principaux",
    f"- N Produit : {n_p}  |  N Témoin : {n_t}",
    f"- Gain moyen : +{gain:.2f} qtx/ha",
    f"- Marge nette : {marge:.0f} €/ha",
    f"- IC 95 % Bootstrap (différence) : [{ci_diff[0]:.2f} ; {ci_diff[1]:.2f}] qtx",
    "",
    "## Statistiques",
    f"- Test : {stat_res['main_test']['name']}  |  p = {p_main:.4f}  |  {'Significatif' if sig else 'Non significatif'} (α = {alpha_v})",
    f"- Cohen's d : {d_val:.3f} ({stat_res['effect']['label']})",
    f"- Puissance estimée : {pwr:.1%}" if pwr else "",
    "",
    "## Diagnostic",
    f"- Shapiro-Wilk Produit : p = {stat_res['diagnostics']['shapiro_p'][0]:.4f}",
    f"- Shapiro-Wilk Témoin  : p = {stat_res['diagnostics']['shapiro_p'][1]:.4f}",
    f"- Levene              : p = {stat_res['diagnostics']['levene_p']:.4f}",
    f"- K-S                 : p = {stat_res['diagnostics']['ks_p']:.4f}",
]
 
report_text = "\n".join(report_lines)
st.download_button(
    "⬇️ Télécharger le rapport (.md)",
    report_text.encode("utf-8"),
    file_name=f"bio_expert_360_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
    mime="text/markdown"
)
 
with st.expander("📋 Données filtrées (après nettoyage IQR)"):
    st.dataframe(df_final.reset_index(drop=True), use_container_width=True)
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Exporter en CSV", csv, file_name="bio_expert_donnees_filtrees.csv", mime="text/csv")
