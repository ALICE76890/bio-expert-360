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
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱",
                    initial_sidebar_state="expanded")

# ── Imports critiques protégés ────────────────────────────────────────────
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

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

if not HAS_PYSHP:
    st.error(
        "❌ Le module **pyshp** n'a pas pu être chargé.\n\n"
        f"Détail technique : `{PYSHP_ERROR}`\n\n"
        "Ajoutez `pyshp` à votre `requirements.txt`, puis **Manage app → Reboot app**."
    )
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700;800&display=swap');
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
    st.markdown("## 🌱 Bio-Expert 360")
    st.caption("Analyse statistique d'essais — v6.0")
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
        st.caption("Indiquez les coordonnées GPS approximatives de votre parcelle.")
        lat_input = st.number_input("Latitude", value=48.8566, format="%.4f")
        lon_input = st.number_input("Longitude", value=2.3522, format="%.4f")

    with st.expander("🌡️ SEUILS DE STRESS (ajustables)", expanded=True):
        st.caption("Réglez vous-même les seuils selon votre culture.")
        t_echaudage = st.slider("Seuil chaleur — échaudage (°C)", 15, 45, 25)
        t_critique = st.slider("Seuil chaleur — critique (°C)", t_echaudage, 50, max(t_echaudage + 5, 30))
        t_gel = st.slider("Seuil de gel (°C)", -15, 5, -2)
        precip_min_jour = st.slider("Pluie minimale considérée comme utile (mm/jour)", 0.0, 5.0, 0.5, step=0.1)
        jours_secheresse = st.slider("Nb de jours secs consécutifs = séquence de sécheresse", 3, 21, 7)


# ══════════════════════════════════════════════════════════════════════════
# 3. MOTEUR STATISTIQUE & MODELISATION
# ══════════════════════════════════════════════════════════════════════════
def format_pval(p):
    if p is None: return "—"
    if p < 0.0001: return f"{p:.2e}"
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
        return None, "Colonne 'potentiel' absente du fichier.", None, False
    nb_zones = df_final['potentiel'].dropna().nunique()
    if nb_zones <= 1:
        return None, "Une seule zone de potentiel détectée. Ajoutez au moins 2 zones.", None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {e}", None, False


# ══════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS MÉTÉO (Open-Meteo & Source info)
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
            if d: frames.append(pd.DataFrame(d))
        except Exception:
            continue

    if not frames: return None
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
# 5. GENERATEUR RAPPORT PDF (Protected & Clean)
# ══════════════════════════════════════════════════════════════════════════
def create_pdf_report(culture, val_p, d_semis, d_appli, d_recolt, n_p, n_t, gain, marge, stat_res, sig, alpha_v, weather_summary=None):
    if not HAS_FPDF:
        return None
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(190, 10, "Rapport d'Analyse - Bio-Expert 360", ln=True, align='C')
    pdf.ln(5)
    
    # Métadonnées Essai
    pdf.set_font("Helvetica", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 8, "1. Configuration de l'Essai", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(95, 6, f"Culture : {culture}", ln=False)
    pdf.cell(95, 6, f"Bande Produit sélectionnée : {val_p}", ln=True)
    pdf.cell(63, 6, f"Semis : {d_semis}", ln=False)
    pdf.cell(63, 6, f"Application : {d_appli}", ln=False)
    pdf.cell(64, 6, f"Récolte : {d_recolt}", ln=True)
    pdf.ln(5)
    
    # Indicateurs de rendement
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(190, 8, "2. Indicateurs Clés de Rendement", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(95, 6, f"Observations Produit : {n_p}", ln=False)
    pdf.cell(95, 6, f"Observations Témoin : {n_t}", ln=True)
    pdf.cell(95, 6, f"Gain Moyen : +{gain:.2f} qtx/ha", ln=False)
    pdf.cell(95, 6, f"Marge Net Estimée : {marge:.0f} EUR/ha", ln=True)
    pdf.ln(5)
    
    # Verdict Statistique
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(190, 8, "3. Conclusions Statistiques", ln=True)
    pdf.set_font("Helvetica", '', 10)
    verdict_str = "IMPACT SIGNIFICATIF DEMONTRE" if sig else "IMPACT NON DEMONTRE"
    pdf.cell(190, 6, f"Verdict : {verdict_str} (alpha = {alpha_v})", ln=True)
    pdf.cell(190, 6, f"Test appliqué : {stat_res['name']}", ln=True)
    pdf.cell(190, 6, f"p-value obtenues : {format_pval(stat_res['p'])}  |  Cohen's d : {stat_res['d']:.2f} ({stat_res['label']})", ln=True)
    pdf.ln(5)
    
    # Volet Environnemental / Météo
    if weather_summary:
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(190, 8, "4. Synthèse Climatique & Stress Parcelle", ln=True)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(190, 6, f"Total Jours Cycles Analysés : {weather_summary['total_jours']} jours", ln=True)
        pdf.cell(95, 6, f"Jours Chaleur (Échaudage) : {weather_summary['nb_chaleur']}", ln=False)
        pdf.cell(95, 6, f"Jours Chaleur Critique : {weather_summary['nb_critique']}", ln=True)
        pdf.cell(95, 6, f"Jours Gel : {weather_summary['nb_gel']}", ln=False)
        pdf.cell(95, 6, f"Jours Séquence Sécheresse : {weather_summary['nb_secheresse']}", ln=True)
        pdf.ln(8)
        
    pdf.set_font("Helvetica", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(190, 5, f"Rapport généré le {datetime.now().strftime('%d/%m/%Y %H:%M')} par Bio-Expert 360. Source Météo: Open-Meteo (ERA5).", align='C')
    
    return pdf.output(dest='S').encode('latin1')


# ══════════════════════════════════════════════════════════════════════════
# 6. PIPELINE CHARGEMENT & TRAITEMENT
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 Bio-Expert 360</h1>
        <p>Analysez vos essais terrain en quelques secondes — comparaison statistique, ACP factorielle, ANOVA spatiale et météo.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
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
    field_names = [f[0] for f in sf.fields[1:]]
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
        st.error("❌ Aucune valeur de rendement exploitable.")
        st.stop()

except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()

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

# En-tête KPIs
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

stat_res = run_main_test(data_p, data_t, alpha_v=alpha_v)
sig = stat_res['p'] < alpha_v

badge_n = '<span class="badge badge-orange">Échantillon réduit</span>' if stat_res['small_sample'] else '<span class="badge badge-blue">Test auto-adapté</span>'
html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
{badge_n} <br><br>
<strong>{'✅ Impact Significatif' if sig else '❌ Impact Non Démontré'}</strong>
— {stat_res['name']} · p = {format_pval(stat_res['p'])} · Cohen's d = {stat_res['d']:.2f} ({stat_res['label']})
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")

# Récupération Météo Amont pour injection ACP + PDF
params_w = {"t_echaudage": t_echaudage, "t_critique": t_critique, "t_gel": t_gel, "precip_min_jour": precip_min_jour}
df_w = fetch_weather(lat_input, lon_input, d_semis, d_recolt)
weather_summary = None
if df_w is not None and not df_w.empty:
    df_w = compute_stress(df_w, params_w, jours_secheresse)
    weather_summary = {
        'nb_chaleur': int(df_w['stress_chaleur'].sum()),
        'nb_critique': int(df_w['stress_critique'].sum()),
        'nb_gel': int(df_w['stress_gel'].sum()),
        'nb_secheresse': int(df_w['stress_secheresse'].sum()),
        'total_jours': len(df_w)
    }

# ══════════════════════════════════════════════════════════════════════════
# 7. ONGLET STRUCTURE DE NAVIGATION
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_pca, tab_meteo = st.tabs([
    "📊 Résultats & Distribution",
    "📐 ANOVA Spatiale",
    "🔮 Analyse ACP & Facteurs",
    "🌦️ Météo & Stress"
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution
# ─────────────────────────────────────────────────────────────────────────
with tab_rdt:
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
                         color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'}, title="Distribution des rendements")
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(df_final, x="grp", y="rdt", color="grp", box=True,
                          color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'}, title="Densité de probabilité")
        st.plotly_chart(fig_viol, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — ANOVA
# ─────────────────────────────────────────────────────────────────────────
with tab_anova:
    if run_anova and HAS_STATSMODELS:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is not None:
            st.subheader(anova_title)
            st.dataframe(anova_table.round(4), use_container_width=True)
            fig_inter = px.box(df_final, x="potentiel", y="rdt", color="grp",
                             color_discrete_map={'Produit': '#2d6a4f', 'Témoin': '#c0392b'}, title="Interaction Sol x Produit")
            st.plotly_chart(fig_inter, use_container_width=True)
        else:
            st.info("ANOVA indisponible : Vérifiez les zones de potentiel (minimum 2 requis).")

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Analyse ACP (Nouveau Composant d'Impact Relatif)
# ─────────────────────────────────────────────────────────────────────────
with tab_pca:
    st.markdown("""
    <div class="vulgarisation">
    🔮 <b>Analyse en Composantes Principales (ACP) :</b> Cette méthode statistique permet d'étudier l'influence combinée 
    du produit face à l'effet du sol (potentiel de sol) et des contraintes climatiques accumulées. 
    Plus les flèches (variables) pointent vers la direction des points de rendement élevés, plus leur contribution est déterminante.
    </div>
    """, unsafe_allow_html=True)
    
    if not HAS_SKLEARN:
        st.warning("Veuillez installer `scikit-learn` pour débloquer l'Analyse en Composantes Principales.")
    else:
        df_pca = df_final.copy()
        df_pca['facteur_produit'] = df_pca['grp'].apply(lambda x: 1 if x == 'Produit' else 0)
        
        if 'potentiel' in df_pca.columns:
            pot_map = {val: idx for idx, val in enumerate(df_pca['potentiel'].dropna().unique())}
            df_pca['facteur_sol'] = df_pca['potentiel'].map(pot_map).fillna(0)
        else:
            df_pca['facteur_sol'] = 0
            
        if weather_summary:
            df_pca['facteur_stress'] = weather_summary['nb_critique'] + weather_summary['nb_secheresse']
        else:
            df_pca['facteur_stress'] = 0
            
        features = ['facteur_produit', 'facteur_sol', 'facteur_stress', 'rdt']
        df_features = df_pca[features].dropna()
        
        if len(df_features) > 5 and df_features['facteur_sol'].std() >= 0:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_features)
            
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled_data)
            
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            fig_biplot = go.Figure()
            fig_biplot.add_trace(go.Scatter(x=pca_res[:, 0], y=pca_res[:, 1], mode='markers',
                                            marker=dict(color=df_features['rdt'], colorscale='Viridis', showscale=True, title="Rdt"),
                                            name="Observations Observations"))
            
            for i, feature in enumerate(features):
                fig_biplot.add_trace(go.Scatter(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]], mode='lines+markers+text',
                                                text=["", feature], textposition="top center",
                                                line=dict(width=3), marker=dict(size=8), name=feature))
                
            fig_biplot.update_layout(title="Biplot ACP : Contribution Relative des Facteurs",
                                     xaxis_title=f"CP1 ({pca.explained_variance_ratio_[0]:.1%})",
                                     yaxis_title=f"CP2 ({pca.explained_variance_ratio_[1]:.1%})",
                                     plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_biplot, use_container_width=True)
            
            # Message d'aide à l'interprétation
            prod_weight = abs(loadings[0, 0])
            sol_weight = abs(loadings[1, 0])
            if prod_weight > sol_weight:
                st.success("💡 L'ACP indique que le **Traitement Produit** montre un impact d'orientation supérieur ou plus direct sur la variance de rendement que l'effet de structure de sol seul.")
            else:
                st.info("💡 L'ACP indique que la variabilité liée au **Potentiel de Sol** reste prépondérante dans l'explication globale des rendements sur cet essai.")
        else:
            st.info("Données factorielles croisées insuffisantes pour projeter l'ACP.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Météo & Stress (Origine Open-Meteo & ERA5 explicitée)
# ─────────────────────────────────────────────────────────────────────────
with tab_meteo:
    st.markdown("""
    <div class="vulgarisation">
    🌦️ <b>Source d'information :</b> Les indicateurs proviennent de l'API **Open-Meteo**. Les données historiques s'appuient sur le modèle de réanalyse mondial à haute résolution <b>ERA5 / ERA5-Land du CEPMMT</b> (Centre européen pour les prévisions météorologiques à moyen terme), garantissant une précision spatio-temporelle optimale pour les parcelles agricoles sans station météo physique.
    </div>
    """, unsafe_allow_html=True)
    
    if df_w is not None and not df_w.empty:
        html_s = f"""<div class="{'stress-high' if (weather_summary['nb_critique'] > 0 or weather_summary['nb_secheresse'] > 0) else 'stress-low'}">
        <strong>Synthèse agro-climatique :</strong> {weather_summary['nb_chaleur']} jour(s) d'échaudage, {weather_summary['nb_critique']} jour(s) critique(s), {weather_summary['nb_gel']} jour(s) de gel subis.
        </div>"""
        st.markdown(html_s, unsafe_allow_html=True)
        
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=df_w['time'], y=df_w['precipitation_sum'], name="Précipitations (mm)", yaxis='y2', opacity=0.4))
        fig_w.add_trace(go.Scatter(x=df_w['time'], y=df_w['temperature_2m_max'], name="T° Max", line=dict(color='red')))
        fig_w.update_layout(title="Suivi Évolution Climatique Parcelle (Source: Open-Meteo / ERA5)", yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.warning("Aucune donnée météo disponible pour la configuration demandée.")

# ══════════════════════════════════════════════════════════════════════════
# 8. EXPORTS COMPLETS & BOUTON PDF RECAPITULATIF
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Exportation des Livrables de Synthèse")

c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    if HAS_FPDF:
        pdf_data = create_pdf_report(culture, val_p, d_semis, d_appli, d_recolt, n_p, n_t, gain, marge, stat_res, sig, alpha_v, weather_summary)
        if pdf_data:
            st.download_button(
                label="⬇️ Télécharger le Bilan Récapitulatif Complet (PDF)",
                data=pdf_data,
                file_name=f"Rapport_Final_BioExpert360_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Veuillez installer le package `fpdf2` pour activer l'impression PDF automatique.")

with c_exp2:
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Exporter les Données Filtrées Brutes (CSV)", csv, file_name="bio_expert_donnees_nettoyees.csv", mime="text/csv")
