# =============================================================================
# BIO-EXPERT 360 PRO
# Version 7.0
# Analyse d'essais agronomiques
# =============================================================================

import os
import io
import zipfile
import shutil
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import requests

from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORTS OPTIONNELS
# =============================================================================

HAS_PYSHP = False
HAS_STATSMODELS = False
HAS_SKLEARN = False
HAS_FPDF = False

try:
    import shapefile as pyshp
    HAS_PYSHP = True
except:
    pass

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except:
    pass

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except:
    pass

try:
    from fpdf import FPDF
    HAS_FPDF = True
except:
    pass

# =============================================================================
# CONFIGURATION STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Bio-Expert 360 PRO",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PALETTE
# =============================================================================

COLOR_PRODUCT = "#0F766E"
COLOR_CONTROL = "#B91C1C"

PALETTE = {
    "Produit": COLOR_PRODUCT,
    "Témoin": COLOR_CONTROL
}

THEME = dict(

    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        family="Inter",
        size=13,
        color="#1e293b"
    ),

    hovermode="x unified",

    margin=dict(
        l=60,
        r=40,
        t=60,
        b=60
    ),

    xaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1"
    )
)

# =============================================================================
# STYLE CSS PREMIUM
# =============================================================================

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html,body,[class*="css"]{

font-family:Inter;

}

.stApp{

background:#f8fafc;

}

/* HERO */

.hero{

padding:35px;

border-radius:18px;

background:linear-gradient(135deg,#0F766E,#134E4A);

box-shadow:0px 10px 35px rgba(0,0,0,.12);

margin-bottom:25px;

}

.hero h1{

color:white;

font-size:38px;

font-weight:800;

margin-bottom:5px;

}

.hero p{

color:#d1fae5;

font-size:17px;

}

/* KPI */

[data-testid="stMetric"]{

background:white;

padding:20px;

border-radius:15px;

border:1px solid #E2E8F0;

box-shadow:0px 2px 8px rgba(0,0,0,.05);

}

/* TITRES */

h2,h3{

font-weight:700;

}

/* BOITES */

.info-box{

padding:18px;

background:white;

border-radius:12px;

border-left:6px solid #0F766E;

margin-bottom:15px;

}

.success-box{

padding:18px;

background:#ECFDF5;

border-left:6px solid #16A34A;

border-radius:12px;

}

.warning-box{

padding:18px;

background:#FEF2F2;

border-left:6px solid #DC2626;

border-radius:12px;

}

</style>

""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

ALPHA_LEVELS = {

"10 % (Exploratoire)":0.10,
"5 % (Standard)":0.05,
"1 % (Très strict)":0.01

}

# =============================================================================
# DOSSIER TEMPORAIRE
# =============================================================================

TEMP_FOLDER="temp"

def clear_temp():

    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)

    os.makedirs(TEMP_FOLDER)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""

<div class="hero">

<h1>🌱 Bio-Expert 360 PRO</h1>

<p>

Plateforme avancée d'analyse des essais agronomiques

•

Statistiques

•

ACP

•

ANOVA

•

Analyse météo

•

Rapport PDF

</p>

</div>

# =============================================================================
# PARTIE 2 - IMPORT INTELLIGENT DES DONNÉES
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Import des données")

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier ZIP contenant le Shapefile",
    type=["zip"]
)

if uploaded_file is None:
    st.info("Importez un fichier ZIP contenant votre Shapefile.")
    st.stop()

clear_temp()

try:

    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall(TEMP_FOLDER)

except Exception:

    st.error("Impossible d'ouvrir le fichier ZIP.")
    st.stop()

# =============================================================================
# Recherche automatique du SHP
# =============================================================================

shp_files = []

for root, _, files in os.walk(TEMP_FOLDER):

    for f in files:

        if f.lower().endswith(".shp"):

            shp_files.append(os.path.join(root, f))

if len(shp_files) == 0:

    st.error("Aucun fichier .shp trouvé.")

    st.stop()

shp_path = shp_files[0]

# =============================================================================
# Lecture du shapefile
# =============================================================================

try:

    sf = pyshp.Reader(shp_path)

except Exception as e:

    st.error(f"Erreur de lecture : {e}")

    st.stop()

fields = [f[0].lower().strip() for f in sf.fields[1:]]

records = [list(r) for r in sf.records()]

df = pd.DataFrame(records, columns=fields)

st.success(f"✔ {len(df)} observations importées")

# =============================================================================
# Détection intelligente des colonnes
# =============================================================================

COLUMN_PATTERNS = {

    "rendement": [

        "rdt",
        "rend",
        "yield",
        "yield_qx",
        "rendement",
        "qtx",
        "qtxha"

    ],

    "bande": [

        "bande",
        "traitement",
        "trait",
        "modalite",
        "modalité",
        "produit",
        "variant"

    ],

    "potentiel": [

        "potentiel",
        "zone",
        "sol",
        "classe_sol",
        "type_sol",
        "potential"

    ],

    "ndvi":[

        "ndvi"

    ],

    "biomasse":[

        "biomasse",
        "biomass"

    ],

    "conductivite":[

        "ce",
        "conductivite",
        "conductivité",
        "ec"

    ],

    "altitude":[

        "altitude",
        "alt"

    ],

    "pente":[

        "pente",
        "slope"

    ]

}

# =============================================================================
# Fonction de détection automatique
# =============================================================================

def detect_column(patterns):

    for p in patterns:

        for c in df.columns:

            if p in c.lower():

                return c

    return None

# =============================================================================
# Détection
# =============================================================================

detected = {}

for variable, patterns in COLUMN_PATTERNS.items():

    detected[variable] = detect_column(patterns)

# =============================================================================
# Vérification
# =============================================================================

if detected["rendement"] is None:

    st.error("Impossible de trouver une colonne rendement.")

    st.write(df.columns.tolist())

    st.stop()

if detected["bande"] is None:

    st.error("Impossible de trouver une colonne bande.")

    st.write(df.columns.tolist())

    st.stop()

# =============================================================================
# Conversion numérique
# =============================================================================

df[detected["rendement"]] = (

    df[detected["rendement"]]

    .astype(str)

    .str.replace(",", ".")

)

df[detected["rendement"]] = pd.to_numeric(

    df[detected["rendement"]],

    errors="coerce"

)

df = df.dropna(subset=[detected["rendement"]])

# =============================================================================
# Renommage interne
# =============================================================================

df = df.rename(columns={

    detected["rendement"]:"rdt",

    detected["bande"]:"bande"

})

if detected["potentiel"]:

    df = df.rename(columns={

        detected["potentiel"]:"potentiel"

    })

if detected["ndvi"]:

    df = df.rename(columns={

        detected["ndvi"]:"ndvi"

    })

if detected["biomasse"]:

    df = df.rename(columns={

        detected["biomasse"]:"biomasse"

    })

if detected["conductivite"]:

    df = df.rename(columns={

        detected["conductivite"]:"conductivite"

    })

if detected["altitude"]:

    df = df.rename(columns={

        detected["altitude"]:"altitude"

    })

if detected["pente"]:

    df = df.rename(columns={

        detected["pente"]:"pente"

    })

# =============================================================================
# Résumé des colonnes détectées
# =============================================================================

st.subheader("🔍 Colonnes détectées automatiquement")

resume = pd.DataFrame({

    "Variable":[

        "Rendement",

        "Bande",

        "Potentiel",

        "NDVI",

        "Biomasse",

        "Conductivité",

        "Altitude",

        "Pente"

    ],

    "Colonne détectée":[

        detected["rendement"],

        detected["bande"],

        detected["potentiel"],

        detected["ndvi"],

        detected["biomasse"],

        detected["conductivite"],

        detected["altitude"],

        detected["pente"]

    ]

})

st.dataframe(resume, use_container_width=True)

# =============================================================================
# Choix du produit
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🌾 Traitement")

produit = st.sidebar.selectbox(

    "Quelle bande correspond au produit ?",

    sorted(df["bande"].unique())

)

df["grp"] = np.where(

    df["bande"] == produit,

    "Produit",

    "Témoin"

)

st.success("Import terminé.")

# =============================================================================
# PARTIE 3 - NETTOYAGE DES DONNÉES
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Nettoyage des données")

clean_method = st.sidebar.selectbox(
    "Méthode de nettoyage",
    [
        "Aucun",
        "IQR",
        "Z-Score",
        "Isolation Forest"
    ]
)

iqr_coef = st.sidebar.slider(
    "Coefficient IQR",
    1.0,
    3.0,
    1.5,
    0.1
)

z_limit = st.sidebar.slider(
    "Seuil Z-score",
    2.0,
    5.0,
    3.0,
    0.1
)

# =============================================================================
# Sauvegarde des données brutes
# =============================================================================

df_raw = df.copy()

# =============================================================================
# Fonction IQR
# =============================================================================

def clean_iqr(data):

    cleaned = []

    for grp in data["grp"].unique():

        sub = data[data["grp"] == grp].copy()

        q1 = sub["rdt"].quantile(0.25)
        q3 = sub["rdt"].quantile(0.75)

        iqr = q3 - q1

        low = q1 - iqr_coef * iqr
        high = q3 + iqr_coef * iqr

        sub = sub[
            (sub["rdt"] >= low)
            &
            (sub["rdt"] <= high)
        ]

        cleaned.append(sub)

    return pd.concat(cleaned)

# =============================================================================
# Fonction Z-score
# =============================================================================

def clean_zscore(data):

    cleaned = []

    for grp in data["grp"].unique():

        sub = data[data["grp"] == grp].copy()

        z = np.abs(stats.zscore(sub["rdt"]))

        sub = sub[z < z_limit]

        cleaned.append(sub)

    return pd.concat(cleaned)

# =============================================================================
# Fonction Isolation Forest
# =============================================================================

def clean_iforest(data):

    if not HAS_SKLEARN:

        st.warning("Scikit-learn absent.")

        return data

    iso = IsolationForest(

        contamination="auto",

        random_state=42

    )

    pred = iso.fit_predict(data[["rdt"]])

    data = data.copy()

    data["iforest"] = pred

    return data[data["iforest"] == 1].drop(columns="iforest")

# =============================================================================
# Choix de la méthode
# =============================================================================

if clean_method == "Aucun":

    df = df_raw.copy()

elif clean_method == "IQR":

    df = clean_iqr(df_raw)

elif clean_method == "Z-Score":

    df = clean_zscore(df_raw)

elif clean_method == "Isolation Forest":

    df = clean_iforest(df_raw)

# =============================================================================
# Rapport
# =============================================================================

removed = len(df_raw) - len(df)

pct = removed / len(df_raw) * 100

st.subheader("📋 Rapport de nettoyage")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Observations initiales",
    len(df_raw)
)

c2.metric(
    "Observations supprimées",
    removed
)

c3.metric(
    "% supprimé",
    f"{pct:.1f} %"
)

# =============================================================================
# Comparaison graphique
# =============================================================================

col1, col2 = st.columns(2)

with col1:

    fig_before = px.box(
        df_raw,
        x="grp",
        y="rdt",
        color="grp",
        title="Avant nettoyage",
        color_discrete_map=PALETTE
    )

    fig_before.update_layout(**THEME)

    st.plotly_chart(
        fig_before,
        use_container_width=True
    )

with col2:

    fig_after = px.box(
        df,
        x="grp",
        y="rdt",
        color="grp",
        title="Après nettoyage",
        color_discrete_map=PALETTE
    )

    fig_after.update_layout(**THEME)

    st.plotly_chart(
        fig_after,
        use_container_width=True
    )

# =============================================================================
# Distribution
# =============================================================================

fig = px.histogram(

    df,

    x="rdt",

    color="grp",

    nbins=30,

    barmode="overlay",

    opacity=0.6,

    color_discrete_map=PALETTE,

    title="Distribution des rendements après nettoyage"

)

fig.update_layout(**THEME)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =============================================================================
# Vérification
# =============================================================================

n_prod = len(df[df["grp"] == "Produit"])

n_tem = len(df[df["grp"] == "Témoin"])

if n_prod < 4 or n_tem < 4:

    st.error(
        "Il reste moins de 4 observations dans un des groupes. "
        "Les analyses statistiques ne pourront pas être réalisées."
    )

    st.stop()

# =============================================================================
# Jeu de données final
# =============================================================================

df_final = df.copy()

# =============================================================================
# PARTIE 4 - MOTEUR STATISTIQUE AVANCÉ
# =============================================================================

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    kruskal,
    normaltest,
    t
)

# =============================================================================
# INTERVALLE DE CONFIANCE
# =============================================================================

def confidence_interval(data, alpha=0.05):

    n = len(data)

    mean = np.mean(data)

    sd = np.std(data, ddof=1)

    se = sd / np.sqrt(n)

    h = se * t.ppf(1 - alpha/2, n-1)

    return mean-h, mean+h

# =============================================================================
# COHEN D
# =============================================================================

def cohen_d(x,y):

    nx=len(x)
    ny=len(y)

    pooled=np.sqrt(
        (
            ((nx-1)*np.var(x,ddof=1))
            +
            ((ny-1)*np.var(y,ddof=1))
        )/(nx+ny-2)
    )

    if pooled==0:

        return 0

    return (np.mean(x)-np.mean(y))/pooled

# =============================================================================
# HEDGES G
# =============================================================================

def hedges_g(x,y):

    d=cohen_d(x,y)

    n=len(x)+len(y)

    correction=1-(3/(4*n-9))

    return d*correction

# =============================================================================
# GLASS DELTA
# =============================================================================

def glass_delta(x,y):

    sd=np.std(y,ddof=1)

    if sd==0:

        return 0

    return (np.mean(x)-np.mean(y))/sd

# =============================================================================
# INTERPRETATION DES TAILLES D'EFFET
# =============================================================================

def interpret_effect(d):

    d=abs(d)

    if d<0.20:
        return "Négligeable"

    if d<0.50:
        return "Faible"

    if d<0.80:
        return "Moyenne"

    if d<1.20:
        return "Importante"

    return "Très importante"

# =============================================================================
# TEST PRINCIPAL
# =============================================================================

def run_statistics(prod,temoin,alpha=0.05):

    result={}

    result["n_produit"]=len(prod)
    result["n_temoin"]=len(temoin)

    result["mean_prod"]=prod.mean()
    result["mean_tem"]=temoin.mean()

    result["std_prod"]=prod.std()
    result["std_tem"]=temoin.std()

    result["gain"]=prod.mean()-temoin.mean()

    # ---------------------------------------------------
    # NORMALITE
    # ---------------------------------------------------

    if len(prod)>=8:

        p1=shapiro(prod).pvalue

    else:

        p1=np.nan

    if len(temoin)>=8:

        p2=shapiro(temoin).pvalue

    else:

        p2=np.nan

    result["p_shapiro_prod"]=p1
    result["p_shapiro_tem"]=p2

    # ---------------------------------------------------
    # HOMOGENEITE
    # ---------------------------------------------------

    result["p_levene"]=levene(prod,temoin).pvalue

    # ---------------------------------------------------
    # CHOIX DU TEST
    # ---------------------------------------------------

    normal=True

    if not np.isnan(p1):

        if p1<alpha:

            normal=False

    if not np.isnan(p2):

        if p2<alpha:

            normal=False

    homogene=result["p_levene"]>alpha

    if normal:

        if homogene:

            stat,p=ttest_ind(prod,temoin)

            test="Student"

        else:

            stat,p=ttest_ind(
                prod,
                temoin,
                equal_var=False
            )

            test="Welch"

    else:

        stat,p=mannwhitneyu(
            prod,
            temoin,
            alternative="two-sided"
        )

        test="Mann-Whitney"

    result["test"]=test
    result["stat"]=stat
    result["pvalue"]=p

    # ---------------------------------------------------
    # TAILLE D'EFFET
    # ---------------------------------------------------

    result["cohen"]=cohen_d(prod,temoin)

    result["hedges"]=hedges_g(prod,temoin)

    result["glass"]=glass_delta(prod,temoin)

    result["effect"]=interpret_effect(result["cohen"])

    # ---------------------------------------------------
    # IC95
    # ---------------------------------------------------

    result["IC_prod"]=confidence_interval(prod)

    result["IC_tem"]=confidence_interval(temoin)

    return result

    # =============================================================================
# PARTIE 5 - ANOVA MULTIFACTORIELLE
# =============================================================================

st.header("📐 Analyse de la Variance")

if "potentiel" not in df_final.columns:

    st.info(
        """
La colonne 'potentiel' n'a pas été trouvée.

L'ANOVA multifactorielle nécessite une colonne contenant
les classes de potentiel de sol.
"""
    )

else:

    df_anova = df_final.copy()

    df_anova["grp"] = df_anova["grp"].astype("category")

    df_anova["potentiel"] = df_anova["potentiel"].astype("category")

    

""", unsafe_allow_html=True)

# =============================================================================
# PARTIE 6 - ACP SCIENTIFIQUE
# =============================================================================

st.header("🔬 Analyse en Composantes Principales")

df_pca = df_final.copy()
numeric_cols = []

for col in df_pca.columns:

    if pd.api.types.is_numeric_dtype(df_pca[col]):

        numeric_cols.append(col)

remove = [

    "id",

    "fid",

    "objectid"

]

numeric_cols = [

    c

    for c in numeric_cols

    if c.lower() not in remove

]

df_pca["Produit"] = np.where(

    df_pca["grp"]=="Produit",

    1,

    0

)

if "potentiel" in df_pca.columns:

    df_pca["Potentiel"] = (

        df_pca["potentiel"]

        .astype("category")

        .cat.codes

    )

    numeric_cols.append("Potentiel")

    numeric_cols.append("Produit")
    X = df_pca[numeric_cols].dropna()

variables = X.columns
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
pca = PCA()

coords = pca.fit_transform(X_scaled)
variance = pca.explained_variance_ratio_ * 100
fig = px.bar(

    x=np.arange(1,len(variance)+1),

    y=variance,

    labels={

        "x":"Composante",

        "y":"Variance expliquée (%)"

    },

    title="Variance expliquée"

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
loadings = pd.DataFrame(

    pca.components_.T,

    columns=[

        f"PC{i+1}"

        for i in range(len(variables))

    ],

    index=variables

)
fig = go.Figure()

for var in variables:

    fig.add_trace(

        go.Scatter(

            x=[0,loadings.loc[var,"PC1"]],

            y=[0,loadings.loc[var,"PC2"]],

            mode="lines+text",

            text=["",var],

            textposition="top center"

        )

    )

fig.update_layout(

    title="Cercle des corrélations"

)

st.plotly_chart(fig,use_container_width=True)
coord = pd.DataFrame(

    coords[:,0:2],

    columns=["PC1","PC2"]

)

coord["Groupe"] = df_pca.loc[X.index,"grp"].values

fig = px.scatter(

    coord,

    x="PC1",

    y="PC2",

    color="Groupe",

    color_discrete_map=PALETTE

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
contrib = pd.DataFrame(

    {

        "Variable":variables,

        "Contribution":(

            loadings["PC1"]**2+

            loadings["PC2"]**2

        )*100

    }

)

contrib = contrib.sort_values(

    "Contribution",

    ascending=False

)
fig = px.bar(

    contrib,

    x="Contribution",

    y="Variable",

    orientation="h",

    text="Contribution"

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
st.dataframe(

    contrib,

    use_container_width=True,

    hide_index=True

)
top = contrib.iloc[0]

second = contrib.iloc[1]

third = contrib.iloc[2]
st.success(

f"""

### Diagnostic ACP

La variable expliquant le plus la variabilité observée est :

**{top.Variable}**

Contribution :

**{top.Contribution:.1f}%**

Les trois principaux facteurs sont :

🥇 {top.Variable}

🥈 {second.Variable}

🥉 {third.Variable}

"""

)
st.success(

f"""

### Diagnostic ACP

La variable expliquant le plus la variabilité observée est :

**{top.Variable}**

Contribution :

**{top.Contribution:.1f}%**

Les trois principaux facteurs sont :

🥇 {top.Variable}

🥈 {second.Variable}

🥉 {third.Variable}

"""

)
if "Produit" in contrib.Variable.values:

    imp_prod = contrib.loc[

        contrib.Variable=="Produit",

        "Contribution"

    ].values[0]

else:

    imp_prod = 0

if "Potentiel" in contrib.Variable.values:

    imp_sol = contrib.loc[

        contrib.Variable=="Potentiel",

        "Contribution"

    ].values[0]

else:

    imp_sol = 0
    if "Produit" in contrib.Variable.values:

    imp_prod = contrib.loc[

        contrib.Variable=="Produit",

        "Contribution"

    ].values[0]

else:

    imp_prod = 0

if "Potentiel" in contrib.Variable.values:

    imp_sol = contrib.loc[

        contrib.Variable=="Potentiel",

        "Contribution"

    ].values[0]

else:

    imp_sol = 0
    if imp_sol > imp_prod:

    st.info(

f"""

Le potentiel de sol influence davantage le rendement.

Produit :

{imp_prod:.1f} %

Potentiel :

{imp_sol:.1f} %

"""

)

else:

    st.info(

f"""

Le produit influence davantage le rendement.

Produit :

{imp_prod:.1f} %

Potentiel :

{imp_sol:.1f} %

"""

)
df_w["Température moyenne"] = (

    df_w["temperature_2m_max"]

    +

    df_w["temperature_2m_min"]

)/2
df_w["GDD"] = (

    df_w["Température moyenne"]-5

).clip(lower=0)
cumul_gdd = df_w["GDD"].sum()

pluie_totale = df_w["precipitation_sum"].sum()

temperature_max = df_w["temperature_2m_max"].max()

temperature_min = df_w["temperature_2m_min"].min()
jours_30 = (

    df_w["temperature_2m_max"]>30

).sum()

jours_35 = (

    df_w["temperature_2m_max"]>35

).sum()

jours_gel = (

    df_w["temperature_2m_min"]<0

).sum()
jours_secs = (

    df_w["precipitation_sum"]<1

).sum()
c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric(

"Pluie",

f"{pluie_totale:.0f} mm"

)

c2.metric(

"GDD",

f"{cumul_gdd:.0f}"

)

c3.metric(

"T° Max",

f"{temperature_max:.1f}°C"

)

c4.metric(

"T° Min",

f"{temperature_min:.1f}°C"

)

c5.metric(

">30°C",

jours_30

)

c6.metric(

"Gel",

jours_gel

)
fig = go.Figure()

fig.add_trace(

go.Bar(

x=df_w.time,

y=df_w.precipitation_sum,

name="Pluie"

)

)

fig.add_trace(

go.Scatter(

x=df_w.time,

y=df_w.temperature_2m_max,

name="T° Max"

)

)

fig.add_trace(

go.Scatter(

x=df_w.time,

y=df_w.temperature_2m_min,

name="T° Min"

)

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
df_w["GDD cumulé"] = df_w["GDD"].cumsum()

fig = px.line(

df_w,

x="time",

y="GDD cumulé",

title="Somme des degrés-jours"

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
df_w["Stress"] = "Normal"

df_w.loc[

df_w.temperature_2m_max>30,

"Stress"

]="Chaleur"

df_w.loc[

df_w.precipitation_sum<1,

"Stress"

]="Sécheresse"

df_w.loc[

df_w.temperature_2m_min<0,

"Stress"

]="Gel"
fig = px.scatter(

df_w,

x="time",

y="temperature_2m_max",

color="Stress",

size="precipitation_sum"

)

fig.update_layout(**THEME)

st.plotly_chart(fig,use_container_width=True)
diagnostic=[]
if pluie_totale<150:

    diagnostic.append(

"La campagne présente un déficit hydrique marqué."

)

if jours_30>10:

    diagnostic.append(

"Un stress thermique important est observé."

)

if jours_gel>5:

    diagnostic.append(

"Des épisodes de gel peuvent avoir affecté la culture."

)

if cumul_gdd>1800:

    diagnostic.append(

"La somme des températures est élevée."

)
st.success(

" ".join(diagnostic)

)
if stats_results["gain"]>0:

    if jours_30>10:

        st.info(

"Le produit semble avoir conservé un avantage malgré un stress thermique important."

)

    elif pluie_totale<150:

        st.info(

"Le produit paraît intéressant en conditions sèches."

)

else:

    st.warning(

"L'absence de gain peut être liée aux conditions climatiques de la campagne."

)
