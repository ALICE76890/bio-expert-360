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

# ── Imports critiques protégés ──────────────────────────────────────────
try:
    import shapefile as pyshp
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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

if not HAS_PYSHP:
    st.error(f"❌ Le module **pyshp** n'a pas pu être chargé.\n\nDétail : `{PYSHP_ERROR}`")
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
.badge-green { background:#d8f3dc; color:#1b4332; }

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
    st.caption("Analyse statistique d'essais en bandes — v6.0")
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
        run_anova = st.checkbox("Activer l'ANOVA spatiale (si ≥ 2 zones)", value=True)
        run_acp = st.checkbox("Activer l'ACP (Analyse en Composantes Principales)", value=True)

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    with st.expander("🌦️ MÉTÉO (position de la parcelle)", expanded=True):
        st.caption("Indiquez les coordonnées GPS approximatives de votre parcelle.")
        lat_input = st.number_input("Latitude", value=48.8566, format="%.4f")
        lon_input = st.number_input("Longitude", value=2.3522, format="%.4f")

    with st.expander("💧 RÉSERVE UTILE (RU)", expanded=True):
        st.caption("Indiquez la réserve utile moyenne de votre parcelle (en mm).")
        reserve_utile = st.number_input("Réserve Utile (mm)", value=80, min_value=10, max_value=300, step=5)

    with st.expander("🌡️ SEUILS DE STRESS (ajustables)", expanded=True):
        st.caption("Réglez vous-même les seuils selon votre culture, votre région ou votre variété.")
        t_echaudage = st.slider("Seuil chaleur — échaudage (°C)", 15, 45, 25)
        t_critique = st.slider("Seuil chaleur — critique (°C)", t_echaudage, 50, max(t_echaudage + 5, 30))
        t_gel = st.slider("Seuil de gel (°C)", -15, 5, -2)
        precip_min_jour = st.slider("Pluie minimale considérée comme utile (mm/jour)", 0.0, 5.0, 0.5, step=0.1)
        jours_secheresse = st.slider("Nb de jours secs consécutifs = séquence de sécheresse", 3, 21, 7)


# ══════════════════════════════════════════════════════════════════════════
# 3. MOTEUR STATISTIQUE
# ══════════════════════════════════════════════════════════════════════════
def format_pval(p):
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
        return None, "Une seule zone de potentiel détectée. ANOVA nécessite ≥ 2 zones.", None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur ANOVA : {e}", None, False


def run_pca_analysis(df_final, alpha_v=0.05):
    """Analyse en Composantes Principales par groupe et potentiel"""
    if not HAS_SKLEARN:
        return None, "scikit-learn non installé."
    
    results = {}
    
    if 'potentiel' not in df_final.columns or df_final['potentiel'].isna().all():
        return None, "Colonne 'potentiel' absente ou vide. ACP nécessite des zones de potentiel."
    
    zones = df_final['potentiel'].dropna().unique()
    
    for zone in zones:
        df_zone = df_final[df_final['potentiel'] == zone]
        
        for grp in ['Produit', 'Témoin']:
            df_grp = df_zone[df_zone['grp'] == grp]
            if len(df_grp) < 3:
                continue
            
            # Sélectionner colonnes numériques pertinentes
            numeric_cols = [c for c in df_grp.columns if c not in ['bande', 'grp', 'potentiel', 'rdt']]
            if len(numeric_cols) < 2:
                continue
            
            X = df_grp[numeric_cols].dropna()
            if X.empty or len(X) < 2:
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA()
            pca.fit(X_scaled)
            
            results[f"{zone}_{grp}"] = {
                'pca': pca,
                'data': X,
                'zone': zone,
                'group': grp,
                'variance': pca.explained_variance_ratio_,
            }
    
    if not results:
        return None, "Pas assez de données numériques pour l'ACP."
    
    return results, "✅ ACP disponible", True


# ══════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS MÉTÉO
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


def compute_reserve_utile(df_w, reserve_utile_mm):
    """Calcule le bilan RU (réserve utile) jour par jour"""
    df_w = df_w.copy()
    df_w['ru'] = reserve_utile_mm
    depletion = 0
    ru_daily = []
    
    for idx, row in df_w.iterrows():
        precip = row['precipitation_sum']
        # Recharge simple : si pluie > 1mm, on remonte la RU
        if precip > 1:
            depletion = max(0, depletion - precip)
        else:
            # Consommation : approximation simple (mm/jour)
            depletion += max(1, row['temperature_2m_max'] / 30) if row['temperature_2m_max'] > 10 else 0.5
        
        ru_current = max(0, reserve_utile_mm - depletion)
        ru_daily.append(ru_current)
    
    df_w['ru_quotidienne'] = ru_daily
    df_w['stress_ru'] = df_w['ru_quotidienne'] < (reserve_utile_mm * 0.3)  # Stress si < 30% RU
    return df_w


# ══════════════════════════════════════════════════════════════════════════
# 5. CRÉATION PDF RAPPORT
# ══════════════════════════════════════════════════════════════════════════
def generate_pdf_report(df_final, stat_res, sig, alpha_v, culture, val_p, 
                        d_semis, d_appli, d_recolt, data_p, data_t, 
                        gain, marge, prix_vente, cout_prod, n_p, n_t):
    """Génère un rapport PDF professionnel"""
    if not HAS_REPORTLAB:
        return None
    
    pdf_filename = f"/tmp/Bio_Expert_360_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, topMargin=0.7*cm, bottomMargin=0.7*cm)
    styles = getSampleStyleSheet()
    story = []
    
    # Styles personnalisés
    style_title = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1b4332'),
        spaceAfter=6,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
    )
    
    style_heading = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2d6a4f'),
        spaceAfter=10,
        fontName='Helvetica-Bold',
        borderBottom=1,
        borderColor=colors.HexColor('#d8f3dc'),
    )
    
    style_normal = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#33433a'),
        leading=14,
    )
    
    # En-tête
    story.append(Paragraph("🌱 BIO-EXPERT 360", style_title))
    story.append(Paragraph(f"Rapport d'Analyse Statistique d'Essai en Bandes", styles['Normal']))
    story.append(Paragraph(f"<i>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</i>", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))
    
    # Infos essai
    story.append(Paragraph("Informations de l'Essai", style_heading))
    
    essai_data = [
        ['Paramètre', 'Valeur'],
        ['Culture', culture],
        ['Bande Produit', val_p],
        ['Semis', str(d_semis)],
        ['Application', str(d_appli)],
        ['Récolte', str(d_recolt)],
    ]
    
    table_essai = Table(essai_data, colWidths=[3*cm, 5*cm])
    table_essai.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d6a4f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f1f5f2')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d8f3dc')),
    ]))
    story.append(table_essai)
    story.append(Spacer(1, 0.3*cm))
    
    # Résultats principaux
    story.append(Paragraph("Résultats Principaux", style_heading))
    
    results_data = [
        ['Métrique', 'Valeur', 'Interprétation'],
        ['Observations Produit', f"{n_p}", f"n = {n_p}"],
        ['Observations Témoin', f"{n_t}", f"n = {n_t}"],
        ['Rendement Moyen Produit', f"{data_p.mean():.1f} qtx/ha", f"σ = {data_p.std():.2f}"],
        ['Rendement Moyen Témoin', f"{data_t.mean():.1f} qtx/ha", f"σ = {data_t.std():.2f}"],
        ['Gain Moyen', f"+{gain:.2f} qtx/ha", f"{(gain/(data_t.mean())*100):.1f}% de gain relatif"],
        ['Marge Nette', f"{marge:.0f} €/ha", f"Coût prod: {cout_prod}€, Prix: {prix_vente}€/T"],
    ]
    
    table_results = Table(results_data, colWidths=[3*cm, 2.5*cm, 3.5*cm])
    table_results.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d6a4f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f1f5f2')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d8f3dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f1f5f2')]),
    ]))
    story.append(table_results)
    story.append(Spacer(1, 0.3*cm))
    
    # Analyse statistique
    story.append(Paragraph("Analyse Statistique", style_heading))
    
    verdict_text = f"""
    <b>Test Choisi :</b> {stat_res['name']}<br/>
    <b>Significativité :</b> {'✅ SIGNIFICATIF' if sig else '❌ NON SIGNIFICATIF'} (α = {alpha_v})<br/>
    <b>Valeur p :</b> {format_pval(stat_res['p'])}<br/>
    <b>Effect Size (Cohen's d) :</b> {stat_res['d']:.3f} ({stat_res['label']})<br/>
    <br/>
    {'L\'effet observé est statistiquement significatif : la différence n\'est probablement pas due au hasard.' if sig 
     else 'L\'effet observé n\'est pas statistiquement significatif : la variabilité de la parcelle empêche de conclure.'}
    """
    
    story.append(Paragraph(verdict_text, style_normal))
    story.append(Spacer(1, 0.3*cm))
    
    # Recommandations
    story.append(Paragraph("Recommandations", style_heading))
    
    if sig:
        reco = f"✅ L'impact du produit est démontré statistiquement. Envisagez un déploiement à plus grande échelle avec un suivi agronomique."
    else:
        reco = f"⚠️ L'impact n'a pas pu être démontré. Vérifiez les conditions de l'essai ou répétez avec une meilleure maîtrise des hétérogénéités spatiales."
    
    story.append(Paragraph(reco, style_normal))
    story.append(Spacer(1, 0.5*cm))
    
    # Footer
    story.append(Paragraph("─" * 80, styles['Normal']))
    story.append(Paragraph(f"<i>Rapport automatisé Bio-Expert 360 • Données de {datetime.now().strftime('%d/%m/%Y')}</i>", 
                          ParagraphStyle('footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)))
    
    try:
        doc.build(story)
        return pdf_filename
    except Exception as e:
        st.error(f"Erreur génération PDF : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# 6. LECTURE FICHIER
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 Bio-Expert 360</h1>
        <p>Analysez vos essais terrain en quelques secondes — comparaison statistique, ANOVA spatiale, ACP et météo.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
    st.markdown("""
    **Colonnes attendues dans le fichier .shp :**

    | Colonne | Obligatoire | Description |
    |---------|-------------|-------------|
    | `bande` | ✅ | Identifiant de bande |
    | `rdt` | ✅ | Rendement (qtx/ha ou t/ha) |
    | `potentiel` | ⚙️ recommandé | Zone de potentiel sol |

    **Nouvelles fonctionnalités v6.0 :**
    - 📊 Comparaison Produit vs Témoin avec test adaptatif
    - 📐 ANOVA spatiale Traitement × Zone
    - 🎯 ACP pour analyse des variabilités intra-groupe
    - 🌦️ Météo avec bilan de réserve utile
    - 📄 Export PDF complet et professionnel
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
    field_names = [f[0] for f in sf.fields[1:]]
    records = [list(r) for r in sf.records()]
    df = pd.DataFrame(records, columns=field_names)
    df.columns = df.columns.str.lower().str.strip()

    missing = [c for c in ['bande', 'rdt'] if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {missing}.")
        st.stop()

    df['rdt'] = pd.to_numeric(df['rdt'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['rdt'])

    if df.empty:
        st.error("❌ Aucune valeur de rendement exploitable.")
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
# 7. EN-TÊTE & KPIs
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🌱 Bio-Expert 360</h1>
    <p>Culture : <b>{culture}</b> &nbsp;·&nbsp; Bande Produit : <b>{val_p}</b> &nbsp;·&nbsp; Réserve Utile : <b>{reserve_utile} mm</b></p>
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
    st.caption(f"⚠️ {n_removed} points supprimés par nettoyage IQR.")

if not has_enough:
    st.error("Données insuffisantes (< 4 obs. par groupe).")
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
