import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="NATUP Analyse Grande Bande", layout="wide", page_icon="🌱",
                    initial_sidebar_state="expanded")

# ── Imports critiques protégés : jamais un écran "Oh no" vide ─────────────
try:
    import shapefile as pyshp  # pyshp — lecture pure Python, sans GDAL
    HAS_PYSHP = True
    PYSHP_ERROR = None
except Exception as e:
    HAS_PYSHP = False
    PYSHP_ERROR = str(e)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

if not HAS_PYSHP:
    st.error(
        "❌ Le module **pyshp** n'a pas pu être chargé.\n\n"
        f"Détail technique : `{PYSHP_ERROR}`\n\n"
        "Ajoutez `pyshp` à votre `requirements.txt`, puis **Manage app → Reboot app**."
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
.stress-high { background:#fdecea; border-left:6px solid #e74c3c; padding:14px 18px; border-radius:10px; color:#7b241c; }
.stress-low  { background:#eafaf1; border-left:6px solid #27ae60; padding:14px 18px; border-radius:10px; color:#145a32; }

.vulgarisation {
    background:#f1f5f2; border-left:4px solid #74a892; padding:14px 18px;
    margin-bottom:14px; border-radius:10px; font-size:0.92rem; color:#33433a;
}
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:700; margin-right:6px; }
.badge-orange { background:#ffe8cc; color:#a65c00; }
.badge-blue { background:#d6e9f8; color:#1b4f72; }

hr { border-top: 1px solid #e0e4e1; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#f1f5f2; border-radius:10px 10px 0 0; padding:10px 18px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#2d6a4f !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)


def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")


ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}
PARAM_CULTURES = {
    "Blé Tendre": {"t_echaudage": 25, "t_critique": 30, "t_gel": -2, "precip_min_jour": 0.5},
    "Maïs":       {"t_echaudage": 32, "t_critique": 36, "t_gel": 0,  "precip_min_jour": 0.5},
    "Orge":       {"t_echaudage": 25, "t_critique": 30, "t_gel": -3, "precip_min_jour": 0.5},
    "Colza":      {"t_echaudage": 27, "t_critique": 32, "t_gel": -5, "precip_min_jour": 0.5},
    "Tournesol":  {"t_echaudage": 30, "t_critique": 35, "t_gel": -2, "precip_min_jour": 0.5},
}

# ══════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌱 NATUP Analyse Grande Bande")
    st.caption("Analyse statistique d'essais en bandes — v5.0")
    st.divider()

    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
        st.caption("Colonnes attendues : `bande`, `rdt`, `potentiel` (optionnel)")

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", date(2024, 10, 20))
        d_appli = st.date_input("Date d'Application produit", date(2025, 3, 10))
        d_recolt = st.date_input("Date de Récolte", date(2025, 7, 15))
        alpha = st.selectbox("Seuil de significativité α", list(ALPHA_LEVELS.keys()))
        alpha_v = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage strict des outliers (IQR 1.2)", value=True)

    with st.expander("📊 OPTIONS STATISTIQUES", expanded=True):
        run_anova = st.checkbox("Activer l'ANOVA spatiale (si ≥ 2 zones de potentiel)", value=True)
        st.caption("Le test de comparaison principal est toujours choisi automatiquement.")

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    with st.expander("🌦️ MÉTÉO (position de la parcelle)", expanded=True):
        st.caption("Indiquez les coordonnées GPS approximatives de votre parcelle (clic droit sur Google Maps → copier les coordonnées).")
        lat_input = st.number_input("Latitude", value=48.8566, format="%.4f")
        lon_input = st.number_input("Longitude", value=2.3522, format="%.4f")

    with st.expander("🌡️ SEUILS DE STRESS (ajustables)", expanded=True):
        st.caption("Réglez vous-même les seuils selon votre culture, votre région ou votre variété.")
        t_echaudage = st.slider("Seuil chaleur — échaudage (°C)", 15, 45, 25)
        t_critique = st.slider("Seuil chaleur — critique (°C)", t_echaudage, 50, max(t_echaudage + 5, 30))
        t_gel = st.slider("Seuil de gel (°C)", -15, 5, -2)
        precip_min_jour = st.slider("Pluie minimale considérée comme utile (mm/jour)", 0.0, 5.0, 0.5, step=0.1)
        jours_secheresse = st.slider("Nb de jours secs consécutifs = séquence de sécheresse", 3, 21, 7)


# ══════════════════════════════════════════════════════════════════════════
# 3. MOTEUR STATISTIQUE — sélection automatique et adaptative du test
# ══════════════════════════════════════════════════════════════════════════
def format_pval(p):
    """Évite l'affichage trompeur '0.0000' : une p-value n'est jamais exactement 0,
    elle peut juste être extrêmement petite (gros échantillons, effet très net)."""
    if p is None:
        return "—"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


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
    - n trop petit (< 8 par groupe) : Shapiro non fiable -> Mann-Whitney directement.
    - n suffisant : Shapiro (normalité) + Levene (homogénéité) -> Student / Welch / Mann-Whitney.
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
                       "croiser le traitement avec le sol — sans variation de sol elle se réduirait au "
                       "test principal déjà calculé. Ajoutez au moins 2 zones de potentiel pour l'activer."), None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {e}", None, False


# ══════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS MÉTÉO (Open-Meteo — gratuit, sans clé API)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_weather(lat, lon, start, end):
    today = date.today()
    url_parts = []

    if start < today:
        archive_end = min(end, today - timedelta(days=1))
        if start <= archive_end:
            url_parts.append(
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={archive_end}"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
            )
    if end >= today:
        forecast_start = max(start, today)
        if forecast_start <= end:
            url_parts.append(
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}&start_date={forecast_start}&end_date={end}"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
            )

    frames = []
    for url in url_parts:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            d = r.json().get("daily", {})
            if d:
                frames.append(pd.DataFrame(d))
        except Exception:
            continue

    if not frames:
        return None
    df_w = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    df_w["time"] = pd.to_datetime(df_w["time"])
    return df_w.sort_values("time").reset_index(drop=True)


def compute_stress(df_w, params, jours_secheresse=7):
    df_w = df_w.copy()
    df_w["stress_chaleur"] = df_w["temperature_2m_max"] >= params["t_echaudage"]
    df_w["stress_critique"] = df_w["temperature_2m_max"] >= params["t_critique"]
    df_w["stress_gel"] = df_w["temperature_2m_min"] <= params["t_gel"]
    df_w["jour_sec"] = df_w["precipitation_sum"] < params["precip_min_jour"]
    df_w["run_id"] = (df_w["jour_sec"] != df_w["jour_sec"].shift()).cumsum()
    run_len = df_w.groupby("run_id")["jour_sec"].transform("size")
    df_w["stress_secheresse"] = df_w["jour_sec"] & (run_len >= jours_secheresse)
    return df_w


# ══════════════════════════════════════════════════════════════════════════
# 5. LECTURE FICHIER (pyshp — sans GDAL)
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 NATUP Grande Bande</h1>
        <p>Analysez vos essais terrain en quelques secondes — comparaison statistique, ANOVA spatiale et météo de la parcelle.</p>
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
    - 🌦️ Analyse météo semis → récolte avec détection de stress thermique/hydrique
    """)
    st.stop()

try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")

    shp_files = []
    for root, _, files in os.walk("temp"):
        shp_files += [os.path.join(root, f) for f in files if f.lower().endswith('.shp')]

    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé dans le zip.")
        st.stop()

    sf = pyshp.Reader(shp_files[0])
    field_names = [f[0] for f in sf.fields[1:]]  # on ignore le champ DeletionFlag
    records = [list(r) for r in sf.records()]
    df = pd.DataFrame(records, columns=field_names)
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
# 6. EN-TÊTE & KPIs
# ══════════════════════════════════════════════════════════════════════════
# 1. On crée deux colonnes pour aligner le logo et le titre sur la même ligne
col_logo, col_titre = st.columns([1, 5])

with col_logo:
    # REMPLACEZ l'URL ci-dessous par le lien direct de votre logo officiel
    url_logo = "https://votre-site-hebergeur.com/logo-natup.png"
    st.image(url_logo, width=120)

with col_titre:
    # 2. On affiche le reste de l'en-tête à droite du logo
    st.markdown(f"""
    <div class="hero" style="padding: 18px 32px; margin-bottom: 0px;">
        <h1>🌱 NATUP Grande Bande</h1>
        <p>Culture analysée : <b>{culture}</b> &nbsp;·&nbsp; Bande Produit : <b>{val_p}</b></p>
    </div>
    """, unsafe_allow_html=True)

# Un petit espace propre avant d'attaquer les chiffres
st.markdown("<br>", unsafe_allow_html=True)

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
— {stat_res['name']} · p = {format_pval(stat_res['p'])} · Cohen's d = {stat_res['d']:.2f} ({stat_res['label']})
{'<br>L\'effet n\'est probablement pas dû au hasard (confiance ≥ '+str(int((1-alpha_v)*100))+'%).' if sig else '<br>La variabilité de la parcelle empêche de conclure à un effet du produit.'}
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")

# ══════════════════════════════════════════════════════════════════════════
# 7. ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_meteo = st.tabs([
    "📊 Résultats & Distribution",
    "📐 ANOVA Spatiale",
    "🌦️ Météo & Stress",
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
        d1.metric("Shapiro (Produit)", f"p = {format_pval(sp_p)}", "✅ Normal" if sp_p > alpha_v else "⚠️ Asymétrique")
        d2.metric("Shapiro (Témoin)", f"p = {format_pval(sp_t)}", "✅ Normal" if sp_t > alpha_v else "⚠️ Asymétrique")
        d3.metric("Levene (Variances)", f"p = {format_pval(lev)}", "✅ Homogène" if lev > alpha_v else "⚠️ Hétérogène")

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
            st.caption("ℹ️ Une valeur affichée à 0.0000 signifie p < 0.0001 (extrêmement significatif), jamais exactement zéro.")

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
# TAB 3 — Météo & Stress
# ─────────────────────────────────────────────────────────────────────────
with tab_meteo:
    params = {
        "t_echaudage": t_echaudage,
        "t_critique": t_critique,
        "t_gel": t_gel,
        "precip_min_jour": precip_min_jour,
    }
    st.markdown(f"""
    <div class="vulgarisation">
    🌦️ Analyse météo de la période <b>semis → récolte</b> au point GPS indiqué dans la barre latérale
    (lat {lat_input:.4f}, lon {lon_input:.4f}). Seuils actuellement réglés :
    chaleur ≥ {t_echaudage}°C, critique ≥ {t_critique}°C, gel ≤ {t_gel}°C,
    sécheresse = {jours_secheresse} jours consécutifs sans pluie utile (&lt; {precip_min_jour} mm/j).
    Ajustez-les dans la barre latérale (🌡️ Seuils de stress).
    </div>
    """, unsafe_allow_html=True)

    if d_recolt < d_semis:
        st.error("La date de récolte doit être postérieure à la date de semis.")
    else:
        with st.spinner("Récupération des données météo…"):
            df_w = fetch_weather(lat_input, lon_input, d_semis, d_recolt)

        if df_w is None or df_w.empty:
            st.warning("Données météo indisponibles pour cette période/localisation. Vérifiez vos coordonnées GPS.")
        else:
            df_w = compute_stress(df_w, params, jours_secheresse)

            nb_chaleur = int(df_w['stress_chaleur'].sum())
            nb_critique = int(df_w['stress_critique'].sum())
            nb_gel = int(df_w['stress_gel'].sum())
            nb_secheresse = int(df_w['stress_secheresse'].sum())
            total_jours = len(df_w)

            stress_total = nb_critique > 0 or nb_secheresse > 0
            html_s = f"""<div class="{'stress-high' if stress_total else 'stress-low'}">
            <strong>{'⚠️ Stress détecté pendant le cycle' if stress_total else '✅ Aucun stress majeur détecté'}</strong>
            — {nb_chaleur} jour(s) ≥ seuil d'échaudage, {nb_critique} jour(s) de chaleur critique,
            {nb_gel} jour(s) de gel, {nb_secheresse} jour(s) en séquence de sécheresse (≥ {jours_secheresse}j), sur {total_jours} jours analysés.
            </div>"""
            st.markdown(html_s, unsafe_allow_html=True)
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Jours chaleur (échaudage)", nb_chaleur)
            m2.metric("🌡️ Jours chaleur critique", nb_critique)
            m3.metric("❄️ Jours de gel", nb_gel)
            m4.metric("🏜️ Jours en séquence sèche", nb_secheresse)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_w['time'], y=df_w['precipitation_sum'], name="Précipitations (mm)",
                marker_color='#3498db', opacity=0.5, yaxis='y2'
            ))
            fig.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_max'], name="T° Max",
                line=dict(color='#e74c3c', width=2), mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_min'], name="T° Min",
                line=dict(color='#5dade2', width=2), mode='lines', fill='tonexty', fillcolor='rgba(93,173,226,0.08)'
            ))
            fig.add_hline(y=params['t_echaudage'], line_dash="dash", line_color="orange",
                          annotation_text=f"Seuil échaudage ({params['t_echaudage']}°C)")
            fig.add_hline(y=params['t_critique'], line_dash="dash", line_color="red",
                          annotation_text=f"Seuil critique ({params['t_critique']}°C)")
            fig.add_hline(y=params['t_gel'], line_dash="dash", line_color="#2980b9",
                          annotation_text=f"Seuil gel ({params['t_gel']}°C)")

            for _, row in df_w[df_w['stress_critique']].iterrows():
                fig.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="red", opacity=0.08, line_width=0)
            for _, row in df_w[df_w['stress_secheresse']].iterrows():
                fig.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="#d35400", opacity=0.06, line_width=0)

            appli_ts = pd.Timestamp(d_appli)
            if df_w['time'].min() <= appli_ts <= df_w['time'].max():
                fig.add_vline(x=appli_ts, line_dash="dot", line_color="green",
                              annotation_text="Application produit", annotation_position="top")

            fig.update_layout(
                title="Évolution météo et zones de stress pendant le cycle cultural",
                xaxis_title="Date", yaxis_title="Température (°C)",
                yaxis2=dict(title="Précipitations (mm)", overlaying='y', side='right', showgrid=False),
                height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Données météo journalières détaillées"):
                show_cols = ['time', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                             'stress_chaleur', 'stress_critique', 'stress_gel', 'stress_secheresse']
                st.dataframe(df_w[show_cols].rename(columns={
                    'time': 'Date', 'temperature_2m_max': 'T° Max', 'temperature_2m_min': 'T° Min',
                    'precipitation_sum': 'Précip. (mm)', 'stress_chaleur': 'Stress chaleur',
                    'stress_critique': 'Stress critique', 'stress_gel': 'Gel', 'stress_secheresse': 'Sécheresse'
                }), use_container_width=True)

            with st.expander("🔍 Comment interpréter ce graphique ?"):
                st.markdown(f"""
                - Courbe **rouge** = température max du jour ; courbe **bleue** = température min.
                - Zones **rouges légères** = jours où la chaleur a dépassé votre seuil critique ({t_critique}°C).
                - Zones **orange** = séquence de sécheresse (≥ {jours_secheresse} jours sans pluie utile).
                - Ligne **verte pointillée** = date d'application produit : regardez si elle tombe juste avant
                  ou pendant une période de stress, ce qui peut influencer l'efficacité du traitement.
                """)

# ══════════════════════════════════════════════════════════════════════════
# 8. EXPORT RAPPORT
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Export des résultats")

report_lines = [
    f"# Rapport NATUP Grande Bande — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    f"**Culture** : {culture}  |  **Bande Produit** : {val_p}",
    f"**Semis** : {d_semis}  |  **Application** : {d_appli}  |  **Récolte** : {d_recolt}",
    "",
    "## Résultats principaux",
    f"- N Produit : {n_p}  |  N Témoin : {n_t}",
    f"- Gain moyen : +{gain:.2f} qtx/ha",
    f"- Marge nette : {marge:.0f} €/ha",
    "",
    "## Statistiques",
    f"- Test utilisé : {stat_res['name']}  |  p = {format_pval(stat_res['p'])}  |  {'Significatif' if sig else 'Non significatif'} (α = {alpha_v})",
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
