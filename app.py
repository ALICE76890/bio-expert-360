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
# 1. STYLE & PRESETS PREMIUM (Look UI/UX Épuré)
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱", initial_sidebar_state="expanded")

# Thème Plotly Premium (Fonds blancs, lignes ultra-discrètes, look minimaliste)
THEME_PLOTLY = dict(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(family="Inter, system-ui, sans-serif", color="#1e293b", size=12),
    margin=dict(l=60, r=40, t=50, b=60),
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#cbd5e1', linewidth=1, tickfont=dict(color='#64748b')),
    yaxis=dict(showgrid=True, gridcolor='#f1f5f9', showline=True, linecolor='#cbd5e1', linewidth=1, tickfont=dict(color='#64748b'))
)

PALETTE = {'Produit': '#0f766e', 'Témoin': '#b91c1c'} # Vert canard chic / Rouge brique pro

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
.stApp { background-color: #f8fafc; }

/* Hero Banner Moderne */
.hero {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
    border-radius: 16px; padding: 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(15, 118, 110, 0.15);
}
.hero h1 { color: #ffffff !important; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.025em; }
.hero p  { color: #ccfbf1; margin: 6px 0 0 0; font-size: 1.05rem; font-weight: 400; }

/* Custom KPI Cards */
[data-testid="stMetric"] {
    background: #ffffff !important; border-radius: 12px !important; padding: 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.02) !important;
    border: 1px solid #e2e8f0 !important;
}
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; color: #0f766e !important; }
[data-testid="stMetricLabel"] { font-weight: 600 !important; color: #64748b !important; font-size: 0.85rem !important; text-transform: uppercase; }

/* Badges de Verdict */
.verdict-box { padding: 20px; border-radius: 12px; font-size: 1.05rem; font-weight: 500; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.verdict-sig { background: #f0fdf4; border-left: 5px solid #16a34a; color: #166534; }
.verdict-nosig { background: #fef2f2; border-left: 5px solid #dc2626; color: #991b1b; }

.vulgarisation { background: #f1f5f9; border-left: 4px solid #94a3b8; padding: 14px 18px; margin-bottom: 20px; border-radius: 8px; color: #334155; font-size: 0.95rem; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; margin-right: 6px; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.badge-blue { background: #e0f2fe; color: #0369a1; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 20px; font-weight: 600; color: #64748b; }
.stTabs [aria-selected="true"] { background: #0f766e !important; color: #ffffff !important; border-color: #0f766e !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports Libs critiques sécurisés ──────────────────────────────────────
try:
    import shapefile as pyshp
    HAS_PYSHP = True
except Exception:
    HAS_PYSHP = False

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

def clear_temp():
    if os.path.exists("temp"): shutil.rmtree("temp")
    os.makedirs("temp")

ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}

# ══════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌱 Bio-Expert 360")
    st.caption("Engine v6.2 — Dashboard Premium")
    st.divider()

    with st.expander("📥 SOURCE DES DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🌾 PARAMÈTRES DE L'ESSAI", expanded=True):
        culture = st.selectbox("Culture", ["Blé Tendre", "Maïs", "Orge", "Colza", "Tournesol"])
        d_semis = st.date_input("Date de Semis", date(2024, 10, 20))
        d_appli = st.date_input("Date d'Application", date(2025, 3, 10))
        d_recolt = st.date_input("Date de Récolte", date(2025, 7, 15))
        alpha = st.selectbox("Seuil α", list(ALPHA_LEVELS.keys()))
        alpha_v = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage Outliers (IQR 1.2)", value=True)

    with st.expander("💰 RENDEMENT ÉCONOMIQUE", expanded=False):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    with st.expander("🌦️ LOCALISATION & STRESS", expanded=False):
        lat_input = st.number_input("Latitude", value=48.8566, format="%.4f")
        lon_input = st.number_input("Longitude", value=2.3522, format="%.4f")
        t_echaudage = st.slider("Seuil échaudage (°C)", 15, 45, 25)
        t_critique = st.slider("Seuil critique (°C)", t_echaudage, 50, max(t_echaudage + 5, 30))
        t_gel = st.slider("Seuil de gel (°C)", -15, 5, -2)
        precip_min_jour = st.slider("Pluie utile (mm/j)", 0.0, 5.0, 0.5, step=0.1)
        jours_secheresse = st.slider("Séquence sèche (jours)", 3, 21, 7)

# ══════════════════════════════════════════════════════════════════════════
# 3. FONCTIONS STATS & MÉTÉO SÉCURISÉES
# ══════════════════════════════════════════════════════════════════════════
def format_pval(p):
    if p is None: return "—"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "négligeable"
    if d < 0.5: return "faible"
    if d < 0.8: return "moyen"
    return "fort"

def run_main_test(data_p, data_t, alpha_v=0.05):
    n_p, n_t = len(data_p), len(data_t)
    if n_p < 4 or n_t < 4:
        return {'name': "Données insuffisantes", 'p': None, 'd': 0, 'label': "—", 'small_sample': True}
    
    small_sample = n_p < 8 or n_t < 8
    if small_sample:
        _, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
        test_nom = "Mann-Whitney U (Échantillon réduit)"
    else:
        try:
            _, p_shap_p = stats.shapiro(data_p)
            _, p_shap_t = stats.shapiro(data_t)
            _, p_lev = stats.levene(data_p, data_t)
            normal = p_shap_p > alpha_v and p_shap_t > alpha_v
            homog = p_lev > alpha_v

            if normal and homog:
                _, p_main = stats.ttest_ind(data_p, data_t)
                test_nom = "Test de Student (Paramétrique)"
            elif normal:
                _, p_main = stats.ttest_ind(data_p, data_t, equal_var=False)
                test_nom = "Test de Welch (Variances inégales)"
            else:
                _, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
                test_nom = "Mann-Whitney U (Non-paramétrique)"
        except Exception:
            _, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
            test_nom = "Mann-Whitney U (Repli de secours)"

    d = cohen_d(data_p.values, data_t.values)
    return {'name': test_nom, 'p': p_main, 'd': d, 'label': interpret_d(d), 'small_sample': small_sample}

@st.cache_data(show_spinner=False)
def fetch_weather(lat, lon, start, end):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json().get("daily", {})
        if d:
            df_w = pd.DataFrame(d)
            df_w["time"] = pd.to_datetime(df_w["time"])
            return df_w
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════
# 4. EXPORT PDF FIXÉ (Plus d'erreur fpdf2 AttributeError)
# ══════════════════════════════════════════════════════════════════════════
def create_pdf_report(culture, val_p, d_semis, d_appli, d_recolt, n_p, n_t, gain, marge, stat_res, sig, alpha_v):
    if not HAS_FPDF: return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(190, 12, "Rapport Bio-Expert 360", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(190, 8, "1. Synthèse Agronomique", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(190, 6, f"Culture : {culture} | Bande Produit : {val_p}", ln=True)
    pdf.cell(190, 6, f"Observations : Produit={n_p} / Témoin={n_t}", ln=True)
    pdf.cell(190, 6, f"Gain Moyen : +{gain:.2f} qtx/ha | Marge : {marge:.0f} EUR/ha", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(190, 8, "2. Analyse Statistique", ln=True)
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(190, 6, f"Test : {stat_res['name']}", ln=True)
    pdf.cell(190, 6, f"p-value : {format_pval(stat_res['p'])} | Significatif : {'Oui' if sig else 'Non'}", ln=True)
    
    # fpdf2 Renvoie directement des bytes si aucun nom de fichier n'est fourni à output()
    return pdf.output()

# ══════════════════════════════════════════════════════════════════════════
# 5. DATA INGESTION PIPELINE (Anti-Crash)
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 Bio-Expert 360</h1>
        <p>Analyse d'essais en bandes : Performances agronomiques, ACP vectorielle et données climatiques Copernicus ERA5.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Importez votre fichier QGIS compressé (.zip) dans le volet de gauche pour commencer.")
    st.stop()

try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")
    shp_files = [os.path.join(r, f) for r, _, fs in os.walk("temp") for f in fs if f.lower().endswith('.shp')]
    
    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé à l'intérieur du ZIP.")
        st.stop()

    sf = pyshp.Reader(shp_files[0])
    fields = [f[0].lower().strip() for f in sf.fields[1:]]
    df = pd.DataFrame([list(r) for r in sf.records()], columns=fields)

    if 'bande' not in df.columns or 'rdt' not in df.columns:
        st.error(f"❌ Colonnes 'bande' et 'rdt' introuvables. Colonnes lues : {list(df.columns)}")
        st.stop()

    df['rdt'] = pd.to_numeric(df['rdt'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['rdt'])

except Exception as e:
    st.error(f"❌ Échec critique de lecture de la couche géospatiale : {e}")
    st.stop()

with st.sidebar:
    val_p = st.selectbox("Assigner le groupe 'Produit' à la bande :", sorted(df['bande'].unique().tolist()))

df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

# Nettoyage Outliers
if clean_iqr:
    clean_list = []
    for g in ['Produit', 'Témoin']:
        sub = df[df['grp'] == g]
        if not sub.empty:
            q1, q3 = sub['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            clean_list.append(sub[(sub['rdt'] >= q1 - 1.2 * iqr) & (sub['rdt'] <= q3 + 1.2 * iqr)])
    df_final = pd.concat(clean_list) if clean_list else df.copy()
else:
    df_final = df.copy()

data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()

if len(data_p) < 4 or len(data_t) < 4:
    st.error("❌ Quantité d'observations nettoyées insuffisante pour exécuter les calculs (minimum 4 par groupe).")
    st.stop()

gain = data_p.mean() - data_t.mean()
marge = ((gain / 10) * prix_vente) - cout_prod

# ══════════════════════════════════════════════════════════════════════════
# 6. HEADER DESIGN & VERDICT
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🌱 Bio-Expert 360</h1>
    <p>Culture : <b>{culture}</b> &nbsp;·&nbsp; Traitement : <b>{val_p}</b></p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Obs. Produit", f"{len(data_p)}")
c2.metric("Obs. Témoin", f"{len(data_t)}")
c3.metric("Moy. Produit", f"{data_p.mean():.1f} qtx")
c4.metric("Moy. Témoin", f"{data_t.mean():.1f} qtx")
c5.metric("Gain Moyen", f"+{gain:.2f} qtx")
c6.metric("Marge Nette", f"{marge:.0f} €/ha")

stat_res = run_main_test(data_p, data_t, alpha_v=alpha_v)
sig = stat_res['p'] < alpha_v if stat_res['p'] is not None else False

badge_st = '<span class="badge badge-orange">Échantillon Faible</span>' if stat_res['small_sample'] else '<span class="badge badge-blue">Données Robustes</span>'
v_class = "verdict-sig" if sig else "verdict-nosig"
v_title = "✅ Différence Statisquement Significative" if sig else "❌ Effet Non Démontré"
v_desc = f"Le produit apporte un gain réel non lié au hasard." if sig else "La variabilité interne de la parcelle masque l'effet du produit."

st.markdown(f"""
<div class="verdict-box {v_class}">
    {badge_st} <b>{v_title}</b> — {stat_res['name']}<br>
    <small style="opacity: 0.85;">p-value = {format_pval(stat_res['p'])} · d de Cohen = {stat_res['d']:.2f} ({stat_res['label']})</small><br>
    <span style="font-size:0.95rem; display:inline-block; margin-top:6px;">{v_desc}</span>
</div>
""", unsafe_allow_html=True)

# Météo amont
df_w = fetch_weather(lat_input, lon_input, d_semis.strftime("%Y-%m-%d"), d_recolt.strftime("%Y-%m-%d"))
weather_summary = None
if df_w is not None and not df_w.empty:
    params_w = {"t_echaudage": t_echaudage, "t_critique": t_critique, "t_gel": t_gel, "precip_min_jour": precip_min_jour}
    df_w = compute_stress(df_w, params_w, jours_secheresse)
    weather_summary = {
        'nb_chaleur': int(df_w['stress_chaleur'].sum()), 'nb_critique': int(df_w['stress_critique'].sum()),
        'nb_gel': int(df_w['stress_gel'].sum()), 'nb_secheresse': int(df_w['stress_secheresse'].sum()),
        'total_jours': len(df_w)
    }

# ══════════════════════════════════════════════════════════════════════════
# 7. ONGLETS DESIGN PREMIUM INTERACTIVE
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_pca, tab_meteo = st.tabs(["📊 Distributions", "📐 ANOVA Spatiale", "🔮 Analyse ACP", "🌦️ Suivi Climatique"])

with tab_rdt:
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(df_final, x="grp", y="rdt", color="grp", points="all",
                         color_discrete_map=PALETTE, labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
                         title="Dispersion des Rendements Épurée")
        fig_box.update_layout(**THEME_PLOTLY)
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(df_final, x="grp", y="rdt", color="grp", box=True,
                             color_discrete_map=PALETTE, labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
                             title="Densité et Distribution des Profils")
        fig_viol.update_layout(**THEME_PLOTLY)
        st.plotly_chart(fig_viol, use_container_width=True)

with tab_anova:
    if run_anova and HAS_STATSMODELS:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is not None:
            st.dataframe(anova_table.round(4), use_container_width=True)
            fig_inter = px.box(df_final, x="potentiel", y="rdt", color="grp", color_discrete_map=PALETTE,
                               title="Comportement du Produit par Zone de Sols")
            fig_inter.update_layout(**THEME_PLOTLY)
            st.plotly_chart(fig_inter, use_container_width=True)
        else:
            st.info("ℹ️ Renseignez au moins 2 classes distinctes (ex: Faible/Fort) dans la colonne 'potentiel' pour débloquer l'ANOVA.")

with tab_pca:
    st.markdown('<div class="vulgarisation">🔮 <b>Biplot factoriel :</b> Analyse simultanée des forces d\'impact. Plus les flèches sont longues et proches de l\'axe du Rendement (rdt), plus leur pouvoir explicatif est fort.</div>', unsafe_allow_html=True)
    if HAS_SKLEARN:
        df_pca = df_final.copy()
        df_pca['facteur_produit'] = df_pca['grp'].apply(lambda x: 1 if x == 'Produit' else 0)
        df_pca['facteur_sol'] = df_pca['potentiel'].map({v: i for i, v in enumerate(df_pca['potentiel'].dropna().unique())}).fillna(0) if 'potentiel' in df_pca.columns else 0
        df_pca['facteur_stress'] = (weather_summary['nb_critique'] + weather_summary['nb_secheresse']) if weather_summary else 0
        
        vars_acp = ['facteur_produit', 'facteur_sol', 'facteur_stress', 'rdt']
        df_act = df_pca[vars_acp].dropna()
        
        if len(df_act) > 5:
            scaled = StandardScaler().fit_transform(df_act)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaled)
            vecs = pca.components_.T * np.sqrt(pca.explained_variance_)

            fig_acp = go.Figure()
            fig_acp.add_trace(go.Scatter(x=coords[:,0], y=coords[:,1], mode='markers',
                                         marker=dict(color=df_act['rdt'], colorscale='Tealrose', showscale=True, colorbar=dict(title="Rdt")),
                                         name="Observations"))
            for i, var in enumerate(vars_acp):
                fig_acp.add_trace(go.Scatter(x=[0, vecs[i,0]], y=[0, vecs[i,1]], mode='lines+markers+text',
                                             text=["", f"<b>{var}</b>"], textposition="top center",
                                             line=dict(color='#0f172a', width=2.5), name=var))
            fig_acp.update_layout(**THEME_PLOTLY)
            fig_acp.update_layout(title="Cartographie ACP de l'Essai (Poids des Facteurs)")
            st.plotly_chart(fig_acp, use_container_width=True)

with tab_meteo:
    st.markdown('<div class="vulgarisation">🌦️ <b>Source d\'information :</b> Données agro-climatiques de réanalyse maillée mondiale haute résolution <b>ERA5 / ERA5-Land du CEPMMT</b> (Copernicus), interrogées via l\'API Open-Meteo.</div>', unsafe_allow_html=True)
    if df_w is not None and not df_w.empty:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=df_w['time'], y=df_w['precipitation_sum'], name="Pluie (mm)", yaxis='y2', marker_color='#38bdf8', opacity=0.4))
        fig_w.add_trace(go.Scatter(x=df_w['time'], y=df_w['temperature_2m_max'], name="T° Max", line=dict(color='#ef4444', width=2)))
        fig_w.add_trace(go.Scatter(x=df_w['time'], y=df_w['temperature_2m_min'], name="T° Min", line=dict(color='#60a5fa', width=1.5, dash='solid')))
        fig_w.update_layout(**THEME_PLOTLY)
        fig_w.update_layout(title="Historique Climatique sur la Période du Cycle", yaxis2=dict(overlaying='y', side='right', showgrid=False, title="Précipitations (mm)"))
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("Données météo non disponibles pour cette période.")

# ══════════════════════════════════════════════════════════════════════════
# 8. PANEL DE LIVRAISON & EXPORTS CLEAN
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Livrables et Archivage")
cx1, cx2 = st.columns(2)

with cx1:
    if HAS_FPDF:
        pdf_out = create_pdf_report(culture, val_p, d_semis, d_appli, d_recolt, len(data_p), len(data_t), gain, marge, stat_res, sig, alpha_v)
        if pdf_out:
            st.download_button("⬇️ Imprimer le Rapport Bilan Structuré (PDF)", data=pdf_out, file_name=f"Rapport_BioExpert360_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
    else:
        st.warning("Librairie `fpdf2` manquante.")

with cx2:
    st.download_button("⬇️ Exporter les Données Filtrées (CSV)", df_final.to_csv(index=False).encode('utf-8'), file_name="bio_expert_clean_data.csv", mime="text/csv")
