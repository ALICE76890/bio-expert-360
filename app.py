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
# 1. CONFIG PAGE & SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bio-Expert 360 Numérique",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="expanded"
)

if 'df_propre' not in st.session_state:
    st.session_state.df_propre = None
if 'gdf_brut' not in st.session_state:
    st.session_state.gdf_brut = None
if 'n_initial_points' not in st.session_state:
    st.session_state.n_initial_points = 0

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

def extract_zip(uploaded_zip, folder_name):
    path = os.path.join("temp", folder_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
        z.extractall(path)
    shp_files = [f for f in os.listdir(path) if f.endswith('.shp')]
    if shp_files:
        return os.path.join(path, shp_files[0])
    return None

# ══════════════════════════════════════════════════════════════════════════════
# 2. RÉFÉRENTIEL ARVALIS & CONFIG
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

# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📸 Bio-Expert 360 Numérique")
st.sidebar.caption("Sonde de pixels par matriciel GéoTIFF")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["1. 🚜 Pré-traitement & Nettoyage", "2. 📈 Synthèse & Statistiques"])

with st.sidebar.expander("🌾 CONFIGURATION COMMERCIALE", expanded=False):
    prix_vente = st.number_input("Prix de vente (€/T)", value=210)
    cout_prod  = st.number_input("Coût Produit (€/ha)", value=45)
    culture    = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
    alpha      = st.selectbox("Seuil α", list(ALPHA_LEVELS.keys()))
    alpha_v    = ALPHA_LEVELS[alpha]
    corr_method = st.selectbox("Correction tests multiples", list(CORRECTION_METHODS.keys()))

# 🛠️ FONCTIONS STATISTIQUES
def bootstrap_ci(data, stat_fn=np.mean, n=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boot_stats = np.array([stat_fn(rng.choice(data, len(data), replace=True)) for _ in range(n)])
    lo = (1 - ci) / 2
    return np.percentile(boot_stats, [lo * 100, (1 - lo) * 100]), boot_stats

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def run_statistical_tests(data_p, data_t, alpha_v=0.05, n_boot=5000):
    result = {}
    n_p, n_t = len(data_p), len(data_t)
    _, p_shap_p = stats.shapiro(data_p) if n_p >= 3 else (None, None)
    _, p_shap_t = stats.shapiro(data_t) if n_t >= 3 else (None, None)
    _, p_lev    = stats.levene(data_p, data_t)
    ks_stat, p_ks = stats.ks_2samp(data_p, data_t)
    result['diagnostics'] = {'shapiro_p': (p_shap_p, p_shap_t), 'levene_p': p_lev, 'ks_stat': ks_stat, 'ks_p': p_ks}
    
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
    
    result['main_test'] = {'name': test_nom, 'id': test_id, 'stat': t_stat, 'p': p_main}
    ci_p, boot_p = bootstrap_ci(data_p.values, n=n_boot)
    ci_t, boot_t = bootstrap_ci(data_t.values, n=n_boot)
    boot_diff = boot_p - boot_t
    result['bootstrap'] = {'ci_p': ci_p, 'ci_t': ci_t, 'ci_diff': np.percentile(boot_diff, [2.5, 97.5]), 'boot_diff': boot_diff}
    
    d = cohen_d(data_p.values, data_t.values)
    se = np.sqrt(1/n_p + 1/n_t)
    from scipy.stats import norm
    pwr = 1 - norm.cdf(norm.ppf(1 - alpha_v/2) - abs(d)/se) + norm.cdf(-norm.ppf(1 - alpha_v/2) - abs(d)/se)
    
    lbl = "négligeable" if abs(d) < 0.2 else "faible" if abs(d) < 0.5 else "moyen" if abs(d) < 0.8 else "fort"
    result['effect'] = {'d': d, 'label': lbl, 'power': pwr}
    return result

def run_anova_analysis(df_final, alpha_v=0.05):
    if not HAS_STATSMODELS or 'potentiel' not in df_final.columns: return None, "Données insuffisantes", None, False
    nb_zones = df_final['potentiel'].dropna().nunique()
    if nb_zones <= 1: return None, f"Une seule zone de potentiel détectée ({df_final['potentiel'].dropna().iloc[0] if nb_zones==1 else 'vide'}). Modèle ANOVA bloqué.", None, False
    try:
        model = smf.ols("rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)", data=df_final).fit()
        return sm.stats.anova_lm(model, typ=2), "📐 ANOVA à 2 facteurs : Traitement × Numéro de Zone Be-API", model, True
    except Exception as e: return None, f"Erreur de calcul : {str(e)}", None, False

def apply_correction(p_values_dict, method_key, alpha_v):
    if not HAS_MULTITEST or not method_key:
        return {k: {'p_raw': v, 'p_adj': v, 'reject': v < alpha_v} for k, v in p_values_dict.items()}
    keys = list(p_values_dict.keys())
    pvals = [p_values_dict[k] for k in keys]
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha_v, method=method_key)
    return {k: {'p_raw': pvals[i], 'p_adj': p_adj[i], 'reject': reject[i]} for i, k in enumerate(keys)}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : INTERACTIVE DE NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "1. 🚜 Pré-traitement & Nettoyage":
    st.title("🚜 Étape 1 : Nettoyage & Extraction des Zones de Potentiel")
    st.markdown("""
    Déposez vos points bruts de batteuse et votre carte d'imagerie Be-API exportée en **GeoTIFF (.tif)** depuis QGIS.
    Le système va extraire le vrai numéro de zone gravé dans l'image sous chaque coordonnée de récolte.
    """)

    col_file1, col_file2 = st.columns(2)
    with col_file1:
        file_batteuse = st.file_uploader("1. Points de la Batteuse (.zip QGIS)", type=["zip"], key="batteuse")
    with col_file2:
        file_geotiff = st.file_uploader("2. Carte BE-API Recalée (.tif ou .tiff)", type=["tif", "tiff"], key="beapi_raster")

    if file_batteuse and file_geotiff:
        try:
            import rasterio
            clear_temp()
            path_bat = extract_zip(file_batteuse, "batteuse")
            
            if path_bat:
                gdf_bat = gpd.read_file(path_bat)
                if gdf_bat.crs is None: gdf_bat.crs = "EPSG:2154"
                
                with rasterio.open(io.BytesIO(file_geotiff.read())) as src_raster:
                    crs_raster = src_raster.crs
                    gdf_bat_m = gdf_bat.to_crs(crs_raster)
                    
                    st.success("✅ Fichiers synchronisés ! Ajustez vos curseurs batteuse pour lancer le tri.")
                    st.divider()

                    st.subheader("⚙️ Paramètres de nettoyage automatique de la batteuse")
                    c_clean1, c_clean2, c_clean3 = st.columns(3)
                    with c_clean1:
                        border_dist = st.slider("Exclusion des bordures de la carte (mètres)", 0, 30, 12)
                        vitesse_range = st.slider("Plage de Vitesse autorisée (m/s)", 0.5, 5.0, (1.5, 3.0))
                    with c_clean2:
                        coupe_unit = st.radio("Unité de la batteuse d'origine", ["Feet (Pieds)", "Mètres"])
                        coupe_range = st.slider("Largeur de coupe acceptée (mètres)", 1.0, 15.0, (6.0, 8.0))
                    with c_clean3:
                        humid_range = st.slider("Humidité grain tolérée (%)", 5.0, 25.0, (10.0, 18.0))
                        temps_range = st.slider("Temps de réponse capteur (secondes)", 0.1, 5.0, (0.8, 2.0))

                    if st.button("🚀 Sonder les Numéros de Zones & Filtrer"):
                        with st.spinner("Sondage numérique des pixels en cours..."):
                            df_work = gdf_bat_m.copy()
                            df_work.columns = df_work.columns.str.lower().str.strip()
                            st.session_state.n_initial_points = len(df_work)

                            df_work['id_unique_point'] = range(len(df_work))

                            # 1. Conversions des unités machines
                            if coupe_unit == "Feet (Pieds)" and 'largeur' in df_work.columns:
                                df_work['largeur_m'] = df_work['largeur'] * 0.3048
                            elif 'largeur' in df_work.columns:
                                df_work['largeur_m'] = df_work['largeur']
                            else:
                                df_work['largeur_m'] = 7.0

                            if 'rdt' in df_work.columns and df_work['rdt'].max() < 15:
                                df_work['rdt'] = df_work['rdt'] * 10 
                            
                            # 2. Filtres métiers batteuse
                            if 'vitesse' in df_work.columns:
                                df_work = df_work[(df_work['vitesse'] >= vitesse_range[0]) & (df_work['vitesse'] <= vitesse_range[1])]
                            if 'humidite' in df_work.columns:
                                df_work = df_work[(df_work['humidite'] >= humid_range[0]) & (df_work['humidite'] <= humid_range[1])]
                            if 'temps' in df_work.columns:
                                df_work = df_work[(df_work['temps'] >= temps_range[0]) & (df_work['temps'] <= temps_range[1])]
                            df_work = df_work[(df_work['largeur_m'] >= coupe_range[0]) & (df_work['largeur_m'] <= coupe_range[1])]

                            # 3. LA SONDE DE POTENTIEL PAR NUMÉRO DE PIXEL
                            coord_list = [(pt.x, pt.y) for pt in df_work.geometry]
                            num_zones = [int(x[0]) for x in src_raster.sample(coord_list)]
                            df_work['potentiel'] = [f"Zone {n}" for n in num_zones]

                            if 'bande' not in df_work.columns:
                                df_work['bande'] = 'Inconnu'

                            # 4. Nettoyage statistique (Moyenne +/- 2 Écarts-types)
                            mean_rdt = df_work['rdt'].mean()
                            std_rdt = df_work['rdt'].std()
                            df_work = df_work[(df_work['rdt'] >= mean_rdt - 2*std_rdt) & (df_work['rdt'] <= mean_rdt + 2*std_rdt)]

                            st.session_state.gdf_brut = gdf_bat_m.to_crs(epsg=4326)
                            st.session_state.gdf_brut.columns = st.session_state.gdf_brut.columns.str.lower().str.strip()
                            st.session_state.gdf_brut['id_unique_point'] = range(len(st.session_state.gdf_brut))
                            
                            st.session_state.df_propre = df_work.to_crs(epsg=4326)
                            
                            st.balloons()
                            st.success("🎉 Sondage et filtrage terminés ! Les numéros de zone ont été attribués. Ouvrez la page 2.")

                if st.session_state.df_propre is not None:
                    st.markdown("---")
                    st.subheader("📊 Rapport d'efficacité du filtre")
                    pts_suppr = st.session_state.n_initial_points - len(st.session_state.df_propre)
                    pct_garde = (len(st.session_state.df_propre) / st.session_state.n_initial_points) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Points Initiaux", f"{st.session_state.n_initial_points} pts")
                    m2.metric("Anomalies Jetées", f"{pts_suppr} pts", delta=f"-{100-pct_garde:.1f}%", delta_color="inverse")
                    m3.metric("Points Propres Gardés", f"{len(st.session_state.df_propre)} pts")

        except Exception as e:
            st.error(f"❌ Erreur de lecture des pixels de l'image : {e}")
    else:
        st.info("💡 Chargez vos fichiers pour activer l'analyse spatiale.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : RENDU STATISTIQUE SÉCURISÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2. 📈 Synthèse & Statistiques":
    if st.session_state.df_propre is None:
        st.warning("⚠️ Aucune donnée disponible. Effectuez d'abord le nettoyage en page 1.")
        st.stop()

    df_final = st.session_state.df_propre.copy()
    data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
    data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
    n_p, n_t = len(data_p), len(data_t)
    
    has_enough = n_p > 3 and n_t > 3
    gain = data_p.mean() - data_t.mean() if has_enough else 0.0
    marge = ((gain / 10) * prix_vente) - cout_prod

    st.title("📈 Résultats par analyse matricielle")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pts Produit", f"{n_p}")
    c2.metric("Pts Témoin", f"{n_t}")
    c3.metric("Moy. Produit", f"{data_p.mean():.1f} qtx" if has_enough else "—")
    c4.metric("Moy. Témoin", f"{data_t.mean():.1f} qtx" if has_enough else "—")
    c5.metric("Gain Net", f"+{gain:.2f} qtx" if has_enough else "—")
    c6.metric("Marge nette", f"{marge:.0f} €/ha" if has_enough else "—")

    stat_res = run_statistical_tests(data_p, data_t, alpha_v=alpha_v, n_boot=n_boot)
    p_main = stat_res['main_test']['p']
    sig = p_main < alpha_v
    
    html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
    <strong>{'✅ Impact Significatif Prouvé' if sig else '❌ Impact Non Démontré'}</strong> 
    — {stat_res['main_test']['name']} · p = {p_main:.4f} · Cohen's d = {stat_res['effect']['d']:.2f}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

    tab_rdt, tab_boot, tab_anova, tab_map = st.tabs(["📊 Distributions", "🎲 Bootstrap", "📐 ANOVA par Numéro", "🗺️ Carte diagnostic"])

    with tab_rdt:
        st.markdown('<div class="vulgarisation">💡 <b>Comprendre cette page :</b> Les graphiques montrent la structure de vos points épurés de tout bruit.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.box(df_final, x="grp", y="rdt", color="grp", points="all", notched=True, color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'}), use_container_width=True)
        col2.plotly_chart(px.violin(df_final, x="grp", y="rdt", color="grp", box=True, color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'}), use_container_width=True)

    with tab_boot:
        st.markdown('<div class="vulgarisation">🎲 <b>Comprendre cette page :</b> Si l\'intervalle ci-dessous exclut la valeur zéro, le gain biologique de votre produit est définitivement validé.</div>', unsafe_allow_html=True)
        st.metric("VRAI GAIN NET (IC)", f"[{stat_res['bootstrap']['ci_diff'][0]:.2f} à {stat_res['bootstrap']['ci_diff'][1]:.2f}] qtx/ha")
        fig_boot = go.Figure(go.Histogram(x=stat_res['bootstrap']['boot_diff'], nbinsx=80, marker_color='#3498db', opacity=0.7))
        fig_boot.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_boot, use_container_width=True)

    with tab_anova:
        st.markdown('<div class="vulgarisation">📐 <b>Comprendre cette page :</b> L\'ANOVA analyse l\'interaction avec vos numéros de zones Be-API pour voir si le produit sur-performe sur des zones spécifiques.</div>', unsafe_allow_html=True)
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is None: st.error(f"❌ {anova_title}")
        else:
            st.subheader(anova_title)
            st.dataframe(anova_table.round(4), use_container_width=True)
            if has_pot:
                st.plotly_chart(px.box(df_final, x="potentiel", y="rdt", color="grp", color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'}), use_container_width=True)

    # TAB 6 : CONFIGURATION EXPLICITE DE LA CARTE DE ZONE BE-API
    with tab_map:
        st.markdown('<div class="vulgarisation">💡 <b>Carte de Contrôle Qualité :</b> Chaque couleur représente une <b>Zone Be-API différente</b>. Si vos points changent de couleur en suivant les lignes de votre champ, c’est que la sonde de pixels a fonctionné à 100 % ! Les points gris sont les anomalies supprimées.</div>', unsafe_allow_html=True)
        try:
            gdf_complet = st.session_state.gdf_brut.copy()
            gdf_complet['lat'] = gdf_complet.geometry.centroid.y
            gdf_complet['lon'] = gdf_complet.geometry.centroid.x

            liste_id_propres = df_final['id_unique_point'].tolist()
            gdf_complet['statut_point'] = gdf_complet['id_unique_point'].apply(
                lambda x: 'Point Validé' if x in liste_id_propres else 'Point Supprimé (Filtre)'
            )

            gdf_complet = gdf_complet.merge(
                df_final[['id_unique_point', 'rdt', 'potentiel']], 
                on='id_unique_point', 
                how='left', 
                suffixes=('_brut', '_propre')
            )

            choix_couleur = st.radio(
                "🎨 Mode de coloration de la carte :",
                ["Colorer par Zone Be-API (Pour vérifier l'association 🎯)", "Colorer par niveau de Rendement (Pour voir la performance 🌾)"],
                horizontal=True
            )

            center_lat = float(gdf_complet['lat'].median())
            center_lon = float(gdf_complet['lon'].mean())

            gdf_complet['rdt_propre'] = pd.to_numeric(gdf_complet['rdt_propre'], errors='coerce').fillna(0)
            gdf_complet['taille_bulle'] = gdf_complet['rdt_propre'].apply(lambda x: max(x, 20) if x > 0 else 5)

            if choix_couleur == "Colorer par Zone Be-API (Pour vérifier l'association 🎯)":
                gdf_complet['classe_affichage'] = gdf_complet.apply(
                    lambda row: row['potentiel'] if row['statut_point'] == 'Point Validé' else 'Point Éliminé', axis=1
                )
                palette_couleurs = px.colors.qualitative.Bold
                
                fig_map = px.scatter_mapbox(
                    gdf_complet, lat="lat", lon="lon", color="classe_affichage", size="taille_bulle", size_max=12,
                    color_discrete_sequence=palette_couleurs, mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon},
                    hover_data={'bande': True, 'rdt_brut': ':.1f', 'potentiel': True},
                    labels={'classe_affichage': 'Légende des Zones'}
                )
            else:
                fig_map = go.Figure()
                df_suppr = gdf_complet[gdf_complet['statut_point'] == 'Point Supprimé (Filtre)']
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_suppr['lat'], lon=df_suppr['lon'], mode='markers',
                    marker=dict(size=5, color='#bdc3c7', opacity=0.4), name='Points Éliminés'
                ))

                df_valides = gdf_complet[gdf_complet['statut_point'] == 'Point Validé']
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_valides['lat'], lon=df_valides['lon'], mode='markers',
                    marker=dict(
                        size=8, color=df_valides['rdt_propre'], colorscale='RdYlGn', showscale=True,
                        colorbar=dict(title="Rendement Net"), opacity=0.9
                    ),
                    name='Points Validés',
                    text=[f"Zone : {z}<br>Rdt : {r:.1f} qtx" for z, r in zip(df_valides['potentiel'], df_valides['rdt_propre'])],
                    hoverinfo='text'
                ))
                fig_map.update_layout(mapbox=dict(style="open-street-map", zoom=16, center=dict(lat=center_lat, lon=center_lon)), height=650, margin={"r": 0, "t": 40, "l": 0, "b": 0})

            st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("📋 Nombre de points de récolte capturés par chaque zone")
            df_verif_points = df_final.groupby('potentiel').size().reset_index(name='Nombre de points associés')
            st.dataframe(df_verif_points, use_container_width=True)

        except Exception as e: 
            st.warning(f"Carte diagnostic indisponible : {e}")

    # ── EXPORT DU RAPPORT ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📤 Export du Rapport")
    report_text = f"# Rapport d'essai Numérique Be-API\nGain Moyen : +{gain:.2f} qtx\nMarge Net : {marge:.0f} €/ha\np-value : {p_main:.4f}"
    st.download_button("⬇️ Télécharger le mémoire (.md)", report_text.encode("utf-8"), file_name="Rapport_BioExpert_Photo.md", mime="text/markdown")
