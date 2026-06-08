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
 
    gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
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
 
# ── Verdict principal (visible dès l'ouverture) ────────────────────────────────
sig  = p_main < alpha_v
html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
<strong>{'✅ Impact Significatif' if sig else '❌ Impact Non Démontré'}</strong> 
— {stat_res['main_test']['name']} · p = {p_main:.4f} · Cohen's d = {d_val:.2f} ({stat_res['effect']['label']})
{'<br>L\'effet n\'est pas dû au hasard avec une confiance ≥ '+str(int((1-alpha_v)*100))+'%.' if sig else '<br>La variabilité de la parcelle masque l\'effet potentiel du produit.'}
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7. ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
tab_rdt, tab_boot, tab_anova, tab_mix, tab_corr, tab_map = st.tabs([
    "📊 Distribution",
    "🎲 Bootstrap & IC",
    "📐 ANOVA",
    "🔀 Modèle Mixte",
    "🔁 Tests Multiples",
    "🗺️ Carte parcelle",
])
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution
# ─────────────────────────────────────────────────────────────────────────────
with tab_rdt:
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(
            df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Distribution des rendements avec encoche IC 95 %"
        )
        fig_box.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig_box, use_container_width=True)
 
    with col2:
        fig_viol = px.violin(
            df_final, x="grp", y="rdt", color="grp", box=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Densité de probabilité (violin)"
        )
        st.plotly_chart(fig_viol, use_container_width=True)
 
    # Stats descriptives
    desc = df_final.groupby('grp')['rdt'].describe().round(2)
    desc.columns = ['N', 'Moy.', 'Éc.-type', 'Min', 'Q25', 'Méd.', 'Q75', 'Max']
    st.dataframe(desc, use_container_width=True)
 
    # Diagnostics normalité
    st.subheader("Diagnostics de validité")
    p_sp, p_st = stat_res['diagnostics']['shapiro_p']
    p_lev = stat_res['diagnostics']['levene_p']
 
    d1, d2, d3 = st.columns(3)
    d1.metric("Shapiro-Wilk Produit", f"p = {p_sp:.4f}", "✅ Normal" if p_sp > alpha_v else "⚠️ Non-normal")
    d2.metric("Shapiro-Wilk Témoin",  f"p = {p_st:.4f}", "✅ Normal" if p_st > alpha_v else "⚠️ Non-normal")
    d3.metric("Levene (homogénéité)", f"p = {p_lev:.4f}", "✅ Homogène" if p_lev > alpha_v else "⚠️ Hétérogène")
 
    with st.expander("ℹ️ Méthode sélectionnée"):
        m = stat_res['main_test']
        st.markdown(f"""<div class="method-box">
Test utilisé : <b>{m['name']}</b> · Statistique : {m['stat']:.4f} · p = {m['p']:.4f}<br>
Seuil α = {alpha_v} — {"La normalité ET l'homogénéité sont vérifiées → test paramétrique." if m['id']=='PARAM' else "Conditions paramétriques non remplies → test robuste."}
</div>""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bootstrap & IC
# ─────────────────────────────────────────────────────────────────────────────
with tab_boot:
    st.subheader(f"Intervalles de confiance Bootstrap ({n_boot:,} itérations · méthode percentile)")
 
    boot_res = stat_res['bootstrap']
    ci_p, ci_t = boot_res['ci_p'], boot_res['ci_t']
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("IC 95 % Produit",
                  f"[{ci_p[0]:.2f} ; {ci_p[1]:.2f}] qtx",
                  f"Moy. = {data_p.mean():.2f}")
    with col_b:
        st.metric("IC 95 % Témoin",
                  f"[{ci_t[0]:.2f} ; {ci_t[1]:.2f}] qtx",
                  f"Moy. = {data_t.mean():.2f}")
 
    ci_diff = boot_res['ci_diff']
    contains_zero = ci_diff[0] <= 0 <= ci_diff[1]
    st.metric(
        "IC 95 % de la différence (Produit − Témoin)",
        f"[{ci_diff[0]:.2f} ; {ci_diff[1]:.2f}] qtx",
        "⚠️ Contient zéro → non significatif" if contains_zero else "✅ Ne contient pas zéro → significatif",
        delta_color="inverse" if contains_zero else "normal"
    )
 
    # Distribution bootstrap de la différence
    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(
        x=boot_res['boot_diff'],
        nbinsx=80,
        name="Δ bootstrap",
        marker_color='#3498db',
        opacity=0.7,
    ))
    fig_boot.add_vline(x=0, line_dash="dash", line_color="red",   annotation_text="Zéro (H₀)")
    fig_boot.add_vline(x=gain, line_dash="dot",  line_color="green", annotation_text=f"Δ observé = {gain:.2f}")
    fig_boot.add_vrect(x0=ci_diff[0], x1=ci_diff[1], fillcolor="#3498db", opacity=0.15,
                       annotation_text="IC 95%", annotation_position="top left")
    fig_boot.update_layout(
        title="Distribution bootstrap de la différence de rendement (Produit − Témoin)",
        xaxis_title="Δ rendement (qtx/ha)", yaxis_title="Fréquence",
        height=400
    )
    st.plotly_chart(fig_boot, use_container_width=True)
 
    # Puissance statistique
    st.subheader("Puissance statistique a posteriori")
    if pwr is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Cohen's d", f"{d_val:.3f}", stat_res['effect']['label'])
        p2.metric("Puissance (1-β)", f"{pwr:.1%}", "Suffisant (≥ 80%)" if pwr >= 0.8 else "Insuffisant (< 80%)")
        p3.metric("Taille d'effet", stat_res['effect']['label'])
 
        # Courbe puissance vs N
        ns   = np.arange(5, 200, 5)
        pwrs = [power_estimate(d_val, n, n, alpha_v) or 0 for n in ns]
        fig_pwr = px.line(
            x=ns, y=pwrs,
            labels={"x": "N par groupe", "y": "Puissance statistique"},
            title=f"Puissance théorique vs N (d = {d_val:.2f}, α = {alpha_v})"
        )
        fig_pwr.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="80 % (seuil usuel)")
        fig_pwr.add_hline(y=0.9, line_dash="dash", line_color="green",  annotation_text="90 %")
        fig_pwr.add_vline(x=n_p,  line_color="blue",  annotation_text=f"N actuel Produit={n_p}")
        fig_pwr.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_pwr, use_container_width=True)
 
    with st.expander("📝 Pourquoi le bootstrap ?"):
        st.markdown("""
Le bootstrap est **non-paramétrique** : il ne suppose aucune distribution théorique.  
Il rééchantillonne vos données réelles 5 000 fois et mesure la variabilité empirique.  
L'IC sur la **différence** est le test le plus direct : si l'intervalle exclut 0, l'effet est réel.  
Avantage clé en agro : robuste aux **distributions asymétriques** fréquentes sur les rendements.
""")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOVA
# ─────────────────────────────────────────────────────────────────────────────
with tab_anova:
    if not run_anova:
        st.info("ANOVA désactivée. Activez-la dans les options statistiques (sidebar).")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : ajoutez-le à votre fichier requirements.txt")
    else:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
 
        if anova_table is None:
            st.error(f"❌ {anova_title}")
            st.info("💡 Note : Les autres tests (Distribution, Bootstrap, Modèle mixte) restent valides et calculés sur l'ensemble de la bande.")
        else:
            st.subheader(anova_title)
            
            at = anova_table.copy()
            at.columns = [c.replace('PR(>F)', 'p-value').replace('sum_sq', 'SCE').replace('mean_sq', 'CME') for c in at.columns]
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
                col1.metric("R² du modèle", f"{r2:.3f}")
                col2.metric("R² ajusté", f"{r2_adj:.3f}")
                col3.metric("F global (p)", f"{anova_model.fvalue:.2f} ({anova_model.f_pvalue:.4f})")
 
            if has_pot and 'potentiel' in df_final.columns:
                st.subheader("Moyennes de cellule (Traitement × Zone de potentiel)")
                pivot = df_final.groupby(['potentiel', 'grp'])['rdt'].agg(['mean', 'std', 'count']).round(2)
                pivot.columns = ['Moy. (qtx)', 'Éc.-type', 'N']
                st.dataframe(pivot, use_container_width=True)
 
                fig_inter = px.box(
                    df_final, x="potentiel", y="rdt", color="grp",
                    color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
                    title="Rendement par zone de potentiel et traitement"
                )
                st.plotly_chart(fig_inter, use_container_width=True)
 
            if anova_model is not None:
                residuals = anova_model.resid
                fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Distribution des résidus", "Q-Q plot résidus"])
                fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Résidus", marker_color='#9b59b6', opacity=0.7), row=1, col=1)
                qq = stats.probplot(residuals)
                fig_res.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Obs.', marker_color='#9b59b6'), row=1, col=2)
                fig_res.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * np.array(qq[0][0]), mode='lines', name='Théorique', line=dict(color='red')), row=1, col=2)
                fig_res.update_layout(height=380, title_text="Analyse des résidus du modèle ANOVA")
                st.plotly_chart(fig_res, use_container_width=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Modèle Mixte
# ─────────────────────────────────────────────────────────────────────────────
with tab_mix:
    if not run_mixed:
        st.info("Modèle mixte désactivé. Activez-le dans les options statistiques (sidebar).")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : `pip install statsmodels`")
    else:
        mix_model, mix_err = run_mixed_model(df_final)
 
        if mix_model is not None:
            st.subheader("Modèle Mixte — rdt ~ C(grp) + (1|bloc)")
            st.caption("Traitement en effet fixe · Bloc expérimental en effet aléatoire (REML)")
 
            fe = mix_model.fe_params.round(4)
            pv = mix_model.pvalues.round(4)
            ci_mix = mix_model.conf_int().round(4)
            res_fe = pd.DataFrame({
                'Effet fixe': fe.index,
                'Estimé (qtx)': fe.values,
                'IC 2.5%': ci_mix.iloc[:, 0].values,
                'IC 97.5%': ci_mix.iloc[:, 1].values,
                'p-value': pv.values,
            })
            res_fe['Significatif'] = res_fe['p-value'].apply(lambda p: '✅' if p < alpha_v else '❌')
            st.dataframe(res_fe, use_container_width=True)
 
            re_var = mix_model.cov_re.values[0][0] if mix_model.cov_re is not None else None
            res_var = mix_model.scale
            col1, col2, col3 = st.columns(3)
            if re_var is not None:
                icc = re_var / (re_var + res_var)
                col1.metric("Variance inter-blocs",  f"{re_var:.3f}")
                col2.metric("Variance résiduelle",   f"{res_var:.3f}")
                col3.metric("ICC (Intraclass Corr.)", f"{icc:.2%}",
                            "Blocs très structurants" if icc > 0.3 else "Blocs peu structurants")
 
            try:
                re_vals = mix_model.random_effects
                re_df   = pd.DataFrame({'Bloc': list(re_vals.keys()),
                                        'Effet aléatoire (qtx)': [v.values[0] for v in re_vals.values()]})
                fig_re = px.bar(re_df, x='Bloc', y='Effet aléatoire (qtx)',
                                title="Effets aléatoires par bloc (BLUP)",
                                color='Effet aléatoire (qtx)', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_re, use_container_width=True)
            except Exception:
                pass
 
        else:
            st.error(f"❌ {mix_err}")
            st.markdown("""
**Quand utiliser le modèle mixte ?** Lorsque vos points de rendement sont groupés en **blocs expérimentaux** (ex: répétitions, passages de batteuse).  
Le modèle sépare la variabilité due aux blocs de l'effet réel du traitement,  
ce qui augmente la **puissance** de détection d'un effet même sur de petits échantillons.
""")
 
        with st.expander("📝 Modèle mixte vs ANOVA classique"):
            st.markdown("""
| Critère | ANOVA classique | Modèle Mixte |
|---------|----------------|--------------|
| Prise en compte des blocs | Effet fixe (blocs doivent être équilibrés) | Effet aléatoire (robuste aux données manquantes) |
| Inférence | Sur les niveaux observés | Peut généraliser à d'autres blocs |
| Recommandé si | Blocs équilibrés, petit N blocs | Blocs déséquilibrés, N blocs élevé |
| Estimateur | MCO | REML (Restricted Maximum Likelihood) |
""")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Tests Multiples
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("Correction pour tests multiples")
    corr_key = CORRECTION_METHODS[corr_method]
 
    has_pot_col = 'potentiel' in df_final.columns and df_final['potentiel'].nunique() > 1
 
    if not has_pot_col:
        st.info("La correction pour tests multiples s'applique lorsque vous comparez le produit sur **plusieurs zones de potentiel** simultanément.")
        st.markdown("""
Ajoutez la colonne `potentiel` dans votre fichier QGIS pour activer cette analyse.  
Sans correction, le risque d'erreur de type I (faux positif) augmente avec chaque test :  
- 3 tests à α=5 % → risque réel ≈ 14 %  
- 5 tests à α=5 % → risque réel ≈ 23 %
""")
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
                    'p brut': round(r['p_raw'], 4),
                    'p corrigé': round(r['p_adj'], 4),
                    'Significatif': '✅' if r['reject'] else '❌',
                })
            df_corr = pd.DataFrame(rows)
            st.dataframe(df_corr, use_container_width=True)
 
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=df_corr['Zone potentiel'], y=df_corr['p brut'],
                name='p brut', marker_color='#3498db', opacity=0.6
            ))
            fig_corr.add_trace(go.Bar(
                x=df_corr['Zone potentiel'], y=df_corr['p corrigé'],
                name='p corrigé', marker_color='#e67e22', opacity=0.8
            ))
            fig_corr.add_hline(y=alpha_v, line_dash="dash", line_color="red",
                               annotation_text=f"α = {alpha_v}")
            fig_corr.update_layout(
                barmode='group',
                title=f"p-values brutes vs corrigées ({corr_method})",
                yaxis_title="p-value",
                height=380
            )
            st.plotly_chart(fig_corr, use_container_width=True)
 
            fig_forest = go.Figure()
            for i, row in df_corr.iterrows():
                color = '#2ecc71' if row['Significatif'] == '✅' else '#e74c3c'
                fig_forest.add_trace(go.Scatter(
                    x=[row['Gain (qtx)']], y=[row['Zone potentiel']],
                    mode='markers', marker=dict(size=14, color=color),
                    name=row['Zone potentiel']
                ))
            fig_forest.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_forest.update_layout(
                title="Forest plot — Gain par zone de potentiel",
                xaxis_title="Gain (qtx/ha)",
                showlegend=False, height=max(250, len(zones) * 60)
            )
            st.plotly_chart(fig_forest, use_container_width=True)
 
    with st.expander("📝 Quelle correction choisir ?"):
        st.markdown("""
| Méthode | Contrôle | Puissance | Recommandée quand |
|---------|----------|-----------|-------------------|
| **Holm-Šídák** | FWER (taux d'erreur familiale) | Bonne | Comparaisons planifiées (≤ 5 zones) |
| **Benjamini-Hochberg** | FDR (faux positifs parmi les positifs) | Meilleure | Exploration de nombreuses zones |
| **Bonferroni** | FWER (très conservatif) | Faible | Quand chaque faux positif est coûteux |
 
En essais grandes bandes avec **2-4 zones de potentiel**, **Holm-Šídák est recommandé**.
""")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Carte parcelle
# ─────────────────────────────────────────────────────────────────────────────
with tab_map:
    if 'geometry' not in gdf.columns:
        st.info("Géométrie non disponible.")
    else:
        try:
            gdf_plot = gdf.copy()
            gdf_plot.columns = gdf_plot.columns.str.lower().str.strip()
            gdf_plot['grp'] = gdf_plot['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
            
            gdf_plot = gdf_plot.merge(
                df_final[['rdt']].assign(idx=df_final.index),
                left_index=True, right_on='idx', how='left',
                suffixes=('_brut', '_nettoye')
            )
            gdf_plot['rdt_carte'] = gdf_plot['rdt_nettoye'].fillna(gdf_plot['rdt_brut'])

            # Recentrage ultra-sécurisé via les limites globales de la couche géospatiale
            bounds = gdf_plot.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
 
            fig_map = px.choropleth_mapbox(
                gdf_plot, 
                geojson=gdf_plot.__geo_interface__,
                locations=gdf_plot.index, 
                color='rdt_carte',
                color_continuous_scale='RdYlGn',
                mapbox_style="open-street-map",
                zoom=14,
                center={"lat": center_lat, "lon": center_lon},
                opacity=0.75,
                hover_data={
                    'bande': True, 
                    'rdt_carte': ':.1f', 
                    'potentiel': True
                } if 'potentiel' in gdf_plot.columns else {'bande': True, 'rdt_carte': ':.1f'},
                labels={'rdt_carte': 'Rendement (qtx/ha)'},
                title="Carte de rendement géoréférencée"
            )
            fig_map.update_layout(height=550, margin={"r": 0, "t": 40, "l": 0, "b": 0})
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
    st.download_button("⬇️ Exporter en CSV", csv,
                       file_name="bio_expert_donnees_filtrees.csv", mime="text/csv")
