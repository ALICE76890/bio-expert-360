import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import zipfile
import io

st.set_page_config(page_title="Mode Diagnostic", layout="wide")
st.title("🛠️ Mode Diagnostic Bio-Expert")

uploaded_file = st.sidebar.file_uploader("Charger le fichier ZIP", type=["zip"])

if uploaded_file:
    st.subheader("🔍 Étape 1 : Que contient vraiment votre ZIP ?")
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            file_list = z.namelist()
            st.write("Fichiers trouvés dans le ZIP :", file_list)
            z.extractall("temp_shp")
            
        # Vérification de la présence du .dbf
        dbf_files = [f for f in file_list if f.lower().endswith('.dbf')]
        if not dbf_files:
            st.error("🚨 Problème majeur : Aucun fichier .dbf trouvé dans le ZIP ! Le .dbf est obligatoire car c'est lui qui contient les colonnes de rendement.")
            st.stop()
            
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture du ZIP : {e}")
        st.stop()

    st.subheader("🔍 Étape 2 : Quelles sont les colonnes vues par le robot ?")
    try:
        shp_file = [f for f in os.listdir("temp_shp") if f.endswith('.shp')][0]
        path_to_shp = os.path.join("temp_shp", shp_file)
        
        gdf = gpd.read_file(path_to_shp)
        
        # On affiche la liste EXACTE des colonnes
        st.write("Liste brute des colonnes :")
        st.code(list(gdf.columns))
        
        st.write("Aperçu des 5 premières lignes de données :")
        st.dataframe(gdf.head())
        
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Shapefile : {e}")
