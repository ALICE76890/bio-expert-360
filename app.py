import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱",
                    initial_sidebar_state="expanded")

# ── Imports critiques protégés : on ne veut JAMAIS un écran "Oh no" vide ──
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
    GEOPANDAS_ERROR = None
except Exception as e:
    HAS_GEOPANDAS = False
    GEOPANDAS_ERROR = str(e)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

if not HAS_GEOPANDAS:
    st.error(
        "❌ Le module **geopandas** n'a pas pu être chargé, l'application ne peut pas démarrer.\n\n"
        f"Détail technique : `{GEOPANDAS_ERROR}`\n\n"
        "**Comment corriger (Streamlit Cloud) :**\n"
        "1. Dans `requirements.txt`, utilisez `geopandas` + `pyogrio` (pas `fiona`).\n"
        "2. Cliquez sur **Manage app → Reboot app** pour forcer une réinstallation complète.\n"
        "3. Si l'erreur persiste, ajoutez un fichier `packages.txt` à la racine du repo avec :\n"
        "```\nlibgdal-dev\ngdal-bin\nlibgeos-dev\n```"
    )
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #fafbf9; }

.hero {
    background: linear-gradient(120deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
    border-radius: 18px; padding: 28px 32px; margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(27,67,50,0.25);
}
.hero h1 { color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800; }
.hero p  { color: #d8f3dc; margin: 4px 0 0 0; font-size: 1rem; }

[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 800; color:#1b4332; }
[data-testid="stMetricLabel"] { font-weight: 600; color:#52796f; }
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 16px; padding: 16px 18px;
    box-shadow: 0 3px 14px rgba(27,67,50,0.07); border: 1px solid #eef2ef;
    transition: transform .15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }

h1, h2, h3 { color:#1b4332; font-weight: 800; }
h2 { border-bottom: 3px solid #d8f3dc; padding-bottom: 6px; }

.verdict-sig {
    background: linear-gradient(135deg,#d8f3dc,#b7e4c7); border-left:6px solid #2d6a4f;
    padding:18px 22px; border-radius:14px; color:#1b4332; font-size:1.05rem;
    box-shadow: 0 4px 14px rgba(45,106,79,0.12);
}
.verdict-nosig {
    background: linear-gradient(135deg,#fde2e2,#f8c6c6); border-left:6px solid #c0392b;
    padding:18px 22px; border-radius:14px; color:#7b241c; font-size:1.05rem;
    box-shadow: 0 4px 14px rgba(192,57,43,0.10);
}

.vulgarisation {
    background:#f1f5f2; border-left:4px solid #74a892; padding:14px 18px;
    margin-bottom:14px; border-radius:10px; font-size:0.92rem; color:#33433a;
}

.badge {
    display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.78rem;
    font-weight:700; margin-right:6px;
}
.badge-orange { background:#ffe8cc; color:#a65c00; }
.badge-blue { background:#d6e9f8; color:#1b4f72; }

hr { border-top: 1px solid #e0e4e1; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background:#f1f5f2; border-radius:10px 10px 0 0; padding:10px 18px; font-weight:600;
}
.stTabs [aria-selected="true"] { background:#2d6a4f !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)


def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")


ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}

# ══════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌱 Bio-Expert 360")
    st.caption("Analyse statistique d'essais en bandes — v4.0")
    st.divider()

    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
        st.caption("Colonnes attendues : `bande`, `rdt`, `potentiel` (optionnel)")

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", ["Blé Tendre", "Maïs", "Orge", "Colza", "Tournesol", "Autre"])
        alpha = st.selectbox("Seuil de significativité α", list(ALPHA_LEVELS.keys()))
        alpha_v = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage strict des outliers (IQR 1.2)", value=True)

    with st.expander("📊 OPTIONS STATISTIQUES", expanded=True):
        run_anova = st.checkbox("Activer l'ANOVA spatiale (si ≥ 2 zones de potentiel)", value=True)
        st.caption("Le test de comparaison principal est toujours choisi automatiquement.")

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)


# ══════════════════════════════════════════════════════════════════════════
# 3. MOTEUR STATISTIQUE — sélection automatique et adaptative du test
# ══════════════════════════════════════════════════════════════════════════
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "négligeable"
    if d < 0.5: return "faible"
    if d < 0.8: return "moyen"
    if d < 1.2: return "fort"
    return "très fort"


def run_main_test(data_p, data_t, alpha_v=0.05):
    """
    Sélectionne automatiquement le test le plus adapté à l'échantillon :
    - n trop petit (< 8 par groupe) : Shapiro non fiable, on bascule directement
      vers Mann-Whitney (non-paramétrique), plus prudent sur petits échantillons.
    - n suffisant : on teste normalité (Shapiro) + homogénéité des variances (Levene)
      puis on choisit Student / Welch / Mann-Whitney en conséquence.
    """
    n_p, n_t = len(data_p), len(data_t)
    small_sample = n_p < 8 or n_t < 8

    p_shap_p = p_shap_t = p_lev = None

    if small_sample:
        t_stat, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
        test_nom = "Mann-Whitney U (échantillon réduit → test robuste)"
    else:
        _, p_shap_p = stats.shapiro(data_p)
        _, p_shap_t = stats.shapiro(data_t)
        _, p_lev = stats.levene(data_p, data_t)
        normal = p_shap_p > alpha_v and p_shap_t > alpha_v
        homog = p_lev > alpha_v

        if normal and homog:
            t_stat, p_main = stats.ttest_ind(data_p, data_t)
            test_nom = "Test de Student (paramétrique)"
        elif normal:
            t_stat, p_main = stats.ttest_ind(data_p, data_t, equal_var=False)
            test_nom = "Test de Welch (variances inégales)"
        else:
            t_stat, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
            test_nom = "Mann-Whitney U (non-paramétrique)"

    d = cohen_d(data_p.values, data_t.values)
    return {
        'name': test_nom, 'p': p_main, 'd': d, 'label': interpret_d(d),
        'shapiro_p': p_shap_p, 'shapiro_t': p_shap_t, 'levene_p': p_lev,
        'small_sample': small_sample,
    }


def run_anova_analysis(df_final, alpha_v=0.05):
    if not HAS_STATSMODELS:
        return None, "statsmodels non installé.", None, False
    if 'potentiel' not in df_final.columns:
        return None, "Colonne 'potentiel' absente du fichier : l'ANOVA spatiale nécessite des zones de sol.", None, False
    nb_zones = df_final['potentiel'].dropna().nunique()
    if nb_zones <= 1:
        return None, ("Une seule zone de potentiel détectée. L'ANOVA à 2 facteurs n'a de sens que pour "
                       "croiser le traitement avec le sol — sans variation de sol, elle se réduirait au "
                       "test principal déjà calculé. Ajoutez au moins 2 zones de potentiel pour l'activer."), None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {e}", None, False


# ══════════════════════════════════════════════════════════════════════════
# 4. LECTURE FICHIER
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 Bio-Expert 360</h1>
        <p>Analysez vos essais terrain en quelques secondes — comparaison statistique, ANOVA spatiale et carte interactive.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
    st.markdown("""
    **Colonnes attendues dans le fichier .shp / attributs QGIS :**

    | Colonne | Obligatoire | Description |
    |---------|-------------|-------------|
    | `bande` | ✅ | Identifiant de bande (ex: A, B, Produit, Témoin) |
    | `rdt` | ✅ | Rendement (qtx/ha ou t/ha) |
    | `potentiel` | ⚙️ recommandé | Zone de potentiel sol (ex: Faible, Moyen, Fort) |

    **Ce que fait l'application :**
    - 📊 Comparaison Produit vs Témoin avec **test choisi automatiquement** selon vos données
    - 📐 ANOVA spatiale Traitement × Zone de potentiel (si au moins 2 zones disponibles)
    - 🗺️ Carte interactive de la parcelle, par bande ou par rendement
    """)
    st.stop()

try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")

    shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
    if shp_files:
        shp_files = [os.path.join("temp", f) for f in shp_files]
    else:
        # cherche aussi dans des sous-dossiers
        for root, _, files in os.walk("temp"):
            shp_files += [os.path.join(root, f) for f in files if f.endswith('.shp')]

    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé dans le zip (même en sous-dossier).")
        st.stop()

    try:
        gdf_raw = gpd.read_file(shp_files[0], engine="pyogrio")
    except Exception:
        gdf_raw = gpd.read_file(shp_files[0])
    if gdf_raw.crs is None:
        gdf_raw.crs = "EPSG:2154"
    gdf = gdf_raw.to_crs(epsg=4326)

    df = pd.DataFrame(gdf.drop(columns='geometry'))
    df.columns = df.columns.str.lower().str.strip()

    missing = [c for c in ['bande', 'rdt'] if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}")
        st.stop()

    df['rdt'] = pd.to_numeric(df['rdt'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['rdt'])

    if df.empty:
        st.error("❌ Aucune valeur de rendement exploitable après nettoyage de la colonne 'rdt'.")
        st.stop()

except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()

# ── Options dynamiques ───────────────────────────────────────────────────
with st.sidebar:
    bandes_dispo = sorted(df['bande'].unique().tolist())
    val_p = st.selectbox("Bande = 'Produit' ?", bandes_dispo)

df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
df_travail = df.copy()

n_initial = len(df_travail)
if clean_iqr:
    clean_list = []
    for g in ['Produit', 'Témoin']:
        sub = df_travail[df_travail['grp'] == g]
        if not sub.empty:
            q1, q3 = sub['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            sub = sub[(sub['rdt'] >= q1 - 1.2 * iqr) & (sub['rdt'] <= q3 + 1.2 * iqr)]
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

# ══════════════════════════════════════════════════════════════════════════
# 5. EN-TÊTE & KPIs
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🌱 Bio-Expert 360</h1>
    <p>Culture analysée : <b>{culture}</b> &nbsp;·&nbsp; Bande Produit : <b>{val_p}</b></p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Obs. Produit", f"{n_p}")
c2.metric("Obs. Témoin", f"{n_t}")
c3.metric("Moy. Produit", f"{data_p.mean():.1f} qtx" if has_enough else "—")
c4.metric("Moy. Témoin", f"{data_t.mean():.1f} qtx" if has_enough else "—")
c5.metric("Gain Moyen", f"+{gain:.2f} qtx" if has_enough else "—")
c6.metric("Marge Nette", f"{marge:.0f} €/ha" if has_enough else "—")
if n_removed > 0:
    st.caption(f"⚠️ {n_removed} points supprimés par nettoyage IQR sur {n_initial} observations.")

if not has_enough:
    st.error("Données insuffisantes (< 4 obs. par groupe). Vérifiez votre fichier.")
    st.stop()

stat_res = run_main_test(data_p, data_t, alpha_v=alpha_v)
sig = stat_res['p'] < alpha_v

badge_n = '<span class="badge badge-orange">Échantillon réduit</span>' if stat_res['small_sample'] else '<span class="badge badge-blue">Test auto-adapté</span>'
html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
{badge_n}
<br><br>
<strong>{'✅ Impact Significatif' if sig else '❌ Impact Non Démontré'}</strong>
— {stat_res['name']} · p = {stat_res['p']:.4f} · Cohen's d = {stat_res['d']:.2f} ({stat_res['label']})
{'<br>L\'effet n\'est probablement pas dû au hasard (confiance ≥ '+str(int((1-alpha_v)*100))+'%).' if sig else '<br>La variabilité de la parcelle empêche de conclure à un effet du produit.'}
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")

# ══════════════════════════════════════════════════════════════════════════
# 6. ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_map = st.tabs([
    "📊 Résultats & Distribution",
    "📐 ANOVA Spatiale",
    "🗺️ Carte parcelle",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution
# ─────────────────────────────────────────────────────────────────────────
with tab_rdt:
    explication = (
        "💡 Avec moins de 8 observations par groupe, le test de normalité (Shapiro) n'est pas fiable : "
        "l'application bascule automatiquement vers le test de Mann-Whitney, plus robuste sur petits échantillons."
        if stat_res['small_sample'] else
        "💡 Le test est choisi automatiquement selon la normalité (Shapiro) et l'homogénéité des variances (Levene) "
        "de vos données : Student si tout est conforme, Welch si les variances diffèrent, Mann-Whitney si la "
        "distribution n'est pas normale."
    )
    st.markdown(f'<div class="vulgarisation">{explication}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(
            df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
            color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Distribution des rendements"
        )
        fig_box.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(
            df_final, x="grp", y="rdt", color="grp", box=True,
            color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Densité de probabilité"
        )
        fig_viol.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
        st.plotly_chart(fig_viol, use_container_width=True)

    desc = df_final.groupby('grp')['rdt'].describe().round(2)
    desc.columns = ['N', 'Moyenne', 'Écart-type', 'Min', 'Q25', 'Médiane', 'Q75', 'Max']
    st.dataframe(desc, use_container_width=True)

    if not stat_res['small_sample']:
        d1, d2, d3 = st.columns(3)
        sp_p, sp_t, lev = stat_res['shapiro_p'], stat_res['shapiro_t'], stat_res['levene_p']
        d1.metric("Shapiro (Produit)", f"p = {sp_p:.4f}", "✅ Normal" if sp_p > alpha_v else "⚠️ Asymétrique")
        d2.metric("Shapiro (Témoin)", f"p = {sp_t:.4f}", "✅ Normal" if sp_t > alpha_v else "⚠️ Asymétrique")
        d3.metric("Levene (Variances)", f"p = {lev:.4f}", "✅ Homogène" if lev > alpha_v else "⚠️ Hétérogène")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — ANOVA
# ─────────────────────────────────────────────────────────────────────────
with tab_anova:
    st.markdown("""
    <div class="vulgarisation">
    📐 L'ANOVA sépare ce qui revient à la qualité du sol de ce qui revient à l'effet réel du produit,
    et révèle si le produit fonctionne différemment selon la zone de potentiel. Elle nécessite
    <b>au moins 2 zones de potentiel</b> distinctes pour être informative.
    </div>
    """, unsafe_allow_html=True)

    if not run_anova:
        st.info("ANOVA désactivée dans la barre latérale.")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : ajoutez-le à votre requirements.txt")
    else:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is None:
            st.warning(f"ℹ️ {anova_title}")
        else:
            st.subheader(anova_title)
            at = anova_table.copy()
            at.columns = [c.replace('PR(>F)', 'p-value').replace('sum_sq', 'SCE').replace('mean_sq', 'CME') for c in at.columns]
            at = at.round(4)

            def style_pval(val):
                try:
                    v = float(val)
                    if v < 0.001: return 'background-color:#d8f3dc; font-weight:bold; color:#1b4332;'
                    if v < alpha_v: return 'background-color:#ffe8cc; color:#a65c00;'
                    return ''
                except Exception:
                    return ''

            target_cols = [c for c in at.columns if 'p-value' in c.lower()]
            try:
                styled_at = at.style.applymap(style_pval, subset=target_cols) if target_cols else at.style
            except Exception:
                styled_at = at
            st.dataframe(styled_at, use_container_width=True)

            if anova_model is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (variance expliquée)", f"{anova_model.rsquared:.1%}")
                col2.metric("R² ajusté", f"{anova_model.rsquared_adj:.3f}")
                col3.metric("F-Value globale", f"{anova_model.fvalue:.2f}")

            st.subheader("📊 Rendement par Zone × Traitement")
            pivot = df_final.groupby(['potentiel', 'grp'])['rdt'].agg(['mean', 'std', 'count']).round(2)
            pivot.columns = ['Rendement Moyen', 'Écart-Type', 'Nombre de points']
            st.dataframe(pivot, use_container_width=True)

            fig_inter = px.box(
                df_final, x="potentiel", y="rdt", color="grp",
                color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'},
                title="Le produit fonctionne-t-il mieux sur certaines zones de sol ?"
            )
            fig_inter.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
            st.plotly_chart(fig_inter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Carte parcelle
# ─────────────────────────────────────────────────────────────────────────
with tab_map:
    if 'geometry' not in gdf.columns or gdf.empty:
        st.info("Géométrie non disponible ou fichier vide.")
    else:
        try:
            gdf_plot = gdf.copy()
            gdf_plot['grp'] = gdf_plot['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
            gdf_plot = gdf_plot.merge(
                df_final[['rdt']].assign(idx=df_final.index),
                left_index=True, right_on='idx', how='left'
            )
            gdf_plot['rdt_carte'] = gdf_plot['rdt']
            gdf_plot['lat'] = gdf_plot.geometry.centroid.y
            gdf_plot['lon'] = gdf_plot.geometry.centroid.x

            st.markdown("### 🗺️ Carte interactive de votre parcelle")
            vue = st.radio("Colorer la carte par :", ["Bande (Produit / Témoin)", "Rendement (qtx/ha)"], horizontal=True)

            center_lat = float(gdf_plot['lat'].median())
            center_lon = float(gdf_plot['lon'].mean())

            if vue.startswith("Bande"):
                fig_map = px.scatter_mapbox(
                    gdf_plot, lat="lat", lon="lon", color="grp", size="rdt_carte", size_max=16,
                    color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'},
                    mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, opacity=0.85,
                    hover_data={'bande': True, 'rdt_carte': ':.1f'},
                    labels={'rdt_carte': 'Rendement (qtx/ha)', 'grp': 'Groupe'},
                    title="Carte par bande — Produit vs Témoin"
                )
            else:
                fig_map = px.scatter_mapbox(
                    gdf_plot, lat="lat", lon="lon", color="rdt_carte", size="rdt_carte", size_max=16,
                    color_continuous_scale="RdYlGn", mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, opacity=0.9,
                    hover_data={'bande': True, 'rdt_carte': ':.1f'},
                    labels={'rdt_carte': 'Rendement (qtx/ha)'},
                    title="Carte de rendement — zones chaudes/froides"
                )

            fig_map.update_layout(height=650, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

            if 'potentiel' in gdf_plot.columns:
                with st.expander("🌍 Vue par zone de potentiel"):
                    df_pot_carte = gdf_plot.groupby('potentiel').agg(
                        rdt_moyen=('rdt_carte', 'mean'), lat=('lat', 'mean'), lon=('lon', 'mean')
                    ).reset_index()
                    df_pot_carte['rdt_moyen'] = df_pot_carte['rdt_moyen'].round(1)
                    fig_pot = px.scatter_mapbox(
                        df_pot_carte, lat="lat", lon="lon", color="potentiel", size="rdt_moyen", size_max=24,
                        color_discrete_sequence=px.colors.qualitative.Bold, mapbox_style="open-street-map",
                        zoom=16, center={"lat": center_lat, "lon": center_lon},
                        title="Rendement moyen par zone de potentiel"
                    )
                    fig_pot.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
                    st.plotly_chart(fig_pot, use_container_width=True)
        except Exception as e:
            st.warning(f"Carte indisponible : {e}")

# ══════════════════════════════════════════════════════════════════════════
# 7. EXPORT RAPPORT
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Export des résultats")

report_lines = [
    f"# Rapport Bio-Expert 360 — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    f"**Culture** : {culture}  |  **Bande Produit** : {val_p}",
    "",
    "## Résultats principaux",
    f"- N Produit : {n_p}  |  N Témoin : {n_t}",
    f"- Gain moyen : +{gain:.2f} qtx/ha",
    f"- Marge nette : {marge:.0f} €/ha",
    "",
    "## Statistiques",
    f"- Test utilisé : {stat_res['name']}  |  p = {stat_res['p']:.4f}  |  {'Significatif' if sig else 'Non significatif'} (α = {alpha_v})",
    f"- Cohen's d : {stat_res['d']:.3f} ({stat_res['label']})",
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
