import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from scipy import stats
import requests
from datetime import datetime, timedelta
import io
import zipfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")

st.title("🌱 BIO-EXPERT 360")
st.markdown("### Analyse d'essais biostimulants par batteuse")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # LE BOUTON SECRET (Mode Diagnostic)
    mode_debug = st.checkbox("🛠️ Activer le mode diagnostic")
    st.divider()
    
    uploaded_file = st.file_uploader("Charger ZIP (contenant .shp, .dbf, .shx)", type=["zip"])
    prix_vente = st.number_input("Prix Vente (€/T)", value=210)
    cout_prod = st.number_input("Coût Produit (€/ha)", value=45)
    d_appli = st.date_input("Date Application", datetime(2026, 3, 15))

# --- MOTEUR PRINCIPAL ---
if uploaded_file:
    try:
        # 1. Extraction du ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp_shp")
        
        # 2. Identification du fichier SHP
        shp_file = [f for f in os.listdir("temp_shp") if f.endswith('.shp')][0]
        path_to_shp = os.path.join("temp_shp", shp_file)

        # 3. Lecture des données géographiques
        gdf = gpd.read_file(path_to_shp)
        
        # 4. Conversion GPS (WGS84) pour la météo
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        df = pd.DataFrame(gdf_wgs84.drop(columns='geometry'))
        
        # 5. LA LIGNE MAGIQUE : on force tout en minuscules
        df.columns = df.columns.str.lower()
        
        # 6. Ajout des coordonnées exactes
        df['lat'] = gdf_wgs84.geometry.y
        df['lon'] = gdf_wgs84.geometry.x

        # --- LE FAMEUX MODE DIAGNOSTIC ---
        if mode_debug:
            st.info("🔍 MODE DIAGNOSTIC ACTIF")
            st.write(f"Nombre total de points lus : **{len(df)}**")
            st.write("Liste exacte des colonnes :", list(df.columns))
            st.write("Aperçu des 3 premières lignes :")
            st.dataframe(df.head(3))
            st.divider()

        # --- AFFICHAGE DES RÉSULTATS PRO ---
        if 'rdt' in df.columns and 'bande' in df.columns:
            # Calcul des moyennes
            m = df.groupby('bande')['rdt'].mean()
            gain = m.get('produit', 0) - m.get('temoin', 0)
            marge = ((gain/10) * prix_vente) - cout_prod
            
            # KPIs en haut de page
            col1, col2, col3 = st.columns(3)
            col1.metric("Gain de rendement", f"+{round(gain, 2)} qtx/ha")
            col2.metric("Marge Nette", f"{round(marge, 2)} €/ha")
            
            # Test statistique (on enlève les cases vides au cas où avec dropna)
            t_stat, p_val = stats.ttest_ind(df[df['bande']=='produit']['rdt'].dropna(), 
                                            df[df['bande']=='temoin']['rdt'].dropna())
            col3.metric("Fiabilité Statistique", "Prouvé ✅" if p_val < 0.05 else "Incertain ❌", f"p={round(p_val,4)}")

            # Graphique Boxplot interactif
            fig = px.box(df, x="potentiel" if "potentiel" in df.columns else "bande", 
                         y="rdt", color="bande", title="Comparatif Rendement vs Potentiel Sol")
            st.plotly_chart(fig, use_container_width=True)

            # Météo via API Open-Meteo
            st.subheader("🌦️ Analyse Climatique post-application")
            lat_m, lon_m = df['lat'].mean(), df['lon'].mean()
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_m}&longitude={lon_m}&start_date={d_appli}&end_date={d_appli + timedelta(days=7)}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FBerlin"
            
            try:
                res = requests.get(url).json()
                meteo_df = pd.DataFrame(res['daily'])
                st.line_chart(meteo_df.set_index('time')['temperature_2m_max'])
            except:
                st.warning("Données météo indisponibles pour ces dates ou ces coordonnées.")
            
            # Bonus si données de sol détectées
            if 'mg' in df.columns and 'ph' in df.columns:
                st.success("✅ Analyse de sol détectée dans le fichier (pH, Mg, P, K, Ca). Prêt pour le module de fertilité !")
                
        else:
            st.error("Les colonnes 'rdt' et 'bande' sont toujours introuvables. Cochez le mode diagnostic pour vérifier leurs noms.")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
