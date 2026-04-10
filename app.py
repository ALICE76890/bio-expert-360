import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
from datetime import datetime, timedelta
import io
import zipfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱")
st.title("🌱 BIO-EXPERT 360 - Dashboard Expert")

# --- LISTE DES PRODUITS (Tu pourras la modifier ici) ---
LISTE_PRODUITS = ["Bio-Stim 1", "Expert-Grow", "Nitro-Plus", "Fulvic-Acid", "Perso..."]

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("📋 Fiche Parcelle")
    uploaded_file = st.file_uploader("Charger ZIP (shp, dbf, shx)", type=["zip"])
    
    st.divider()
    culture = st.selectbox("Culture", ["Blé", "Orge", "Colza", "Maïs", "Tournesol"])
    d_semis = st.date_input("Date de Semis", datetime(2025, 10, 15))
    
    st.divider()
    st.header("🧪 Application")
    produit_nom = st.selectbox("Produit appliqué", LISTE_PRODUITS)
    dose = st.number_input("Dose (L/ha ou kg/ha)", value=2.0)
    d_appli = st.date_input("Date d'application", datetime(2026, 3, 15))
    
    st.divider()
    st.header("💰 Économie")
    prix_vente = st.number_input("Prix de vente (€/T)", value=210)
    cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    st.divider()
    mode_debug = st.checkbox("🛠️ Mode diagnostic")

# --- LOGIQUE PRINCIPALE ---
if uploaded_file:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp_shp")
        shp_file = [f for f in os.listdir("temp_shp") if f.endswith('.shp')][0]
        gdf = gpd.read_file(os.path.join("temp_shp", shp_file))
        
        # Nettoyage
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        df = pd.DataFrame(gdf_wgs84.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        # --- CONFIGURATION DES BANDES ---
        st.sidebar.header("🎯 Mapping des données")
        liste_valeurs_bande = df['bande'].unique().tolist()
        valeur_produit = st.sidebar.selectbox("Quelle valeur = Produit ?", liste_valeurs_bande)
        
        # Création d'une colonne simplifiée pour le calcul
        df['label_bande'] = df['bande'].apply(lambda x: 'Produit' if x == valeur_produit else 'Temoin')

        if mode_debug:
            st.write("Colonnes détectées :", list(df.columns))
            st.dataframe(df.head(3))

        # --- CALCULS ---
        if 'rdt' in df.columns:
            m = df.groupby('label_bande')['rdt'].mean()
            gain = m.get('Produit', 0) - m.get('Temoin', 0)
            marge = ((gain/10) * prix_vente) - cout_prod
            
            # P-Value réelle
            data_p = df[df['label_bande'] == 'Produit']['rdt'].dropna()
            data_t = df[df['label_bande'] == 'Temoin']['rdt'].dropna()
            t_stat, p_val = stats.ttest_ind(data_p, data_t)

            # --- AFFICHAGE ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gain Rendement", f"{round(gain, 2)} qtx/ha")
            c2.metric("Marge Nette", f"{round(marge, 2)} €/ha")
            c3.metric("P-Value (Fiabilité)", f"{round(p_val, 4)}")
            c4.metric("Confiance", "✅ Élevée" if p_val < 0.05 else "⚠️ Faible")

            tab1, tab2, tab3 = st.tabs(["📊 Analyse Rendement", "🧪 Corrélations Sol", "🌦️ Météo & Climat"])

            with tab1:
                fig_box = px.box(df, x="potentiel" if "potentiel" in df.columns else "label_bande", 
                                 y="rdt", color="label_bande", notched=True,
                                 title=f"Impact de {produit_nom} sur le rendement")
                st.plotly_chart(fig_box, use_container_width=True)

            with tab2:
                st.subheader("Analyse de corrélation (Impact des éléments du sol)")
                # On ne garde que les colonnes numériques pour la corrélation
                cols_sol = [c for c in ['rdt', 'ph', 'p', 'k', 'mg', 'ca'] if c in df.columns]
                if len(cols_sol) > 1:
                    corr = df[cols_sol].corr()
                    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                         title="Est-ce que le sol explique le rendement ?")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.info("Plus le score est proche de 1 ou -1, plus l'élément (ex: Mg) a un impact direct sur le rendement.")
                else:
                    st.warning("Pas assez de données de sol pour corréler.")

            with tab3:
                st.subheader("Conditions climatiques (J+7 après application)")
                lat_m, lon_m = gdf_wgs84.geometry.y.mean(), gdf_wgs84.geometry.x.mean()
                
                # API Météo avec Hygrométrie (relative_humidity_2m)
                url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_m}&longitude={lon_m}&start_date={d_appli}&end_date={d_appli + timedelta(days=7)}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&hourly=relative_humidity_2m&timezone=Europe%2FBerlin"
                
                try:
                    r = requests.get(url).json()
                    # Météo quotidienne
                    meteo_d = pd.DataFrame(r['daily'])
                    st.write(f"🌧️ Pluie totale : {meteo_d['precipitation_sum'].sum()} mm")
                    st.write(f"💨 Vent max : {meteo_d['wind_speed_10m_max'].max()} km/h")
                    
                    # Graphique Hygro
                    meteo_h = pd.DataFrame(r['hourly'])
                    fig_hygro = px.line(meteo_h, x=meteo_h.index, y="relative_humidity_2m", title="Hygrométrie (%) au moment de l'application")
                    st.plotly_chart(fig_hygro, use_container_width=True)
                except:
                    st.error("L'API météo ne répond pas. Vérifiez la date d'application (pas trop dans le futur).")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
