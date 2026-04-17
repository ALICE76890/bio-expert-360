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
import numpy as np
import shutil

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Bio-Expert 360 Pro", layout="wide", page_icon="🌱")

# --- FONCTION DE NETTOYAGE ---
def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

# --- SEUILS DE STRESS ARVALIS (BIBLIO) ---
# Blé : 25°C ralentissement métabolique / 30°C arrêt du remplissage (échaudage sévère)
PARAM_CULTURES = {
    "Blé Tendre": {"base_t": 0, "opti": 20, "echaudage": 25, "critique": 30},
    "Maïs": {"base_t": 6, "opti": 30, "echaudage": 35, "critique": 38},
    "Orge": {"base_t": 0, "opti": 18, "echaudage": 25, "critique": 30}
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
    
    with st.expander("🧹 NETTOYAGE IQR (Tukey 1.5x)", expanded=True):
        clean_outliers = st.checkbox("Filtrer les points aberrants", value=True)

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", datetime(2024, 10, 20))
        d_appli = st.date_input("Date d'Application", datetime(2025, 3, 10))
        d_recolte = st.date_input("Date de Récolte", datetime(2025, 7, 15))
        
    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

# --- TRAITEMENT PRINCIPAL ---
if uploaded_file:
    try:
        clear_temp()
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            z.extractall("temp")
        
        shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
        if not shp_files:
            st.error("Aucun fichier .shp trouvé")
            st.stop()
            
        gdf = gpd.read_file(os.path.join("temp", shp_files[0])).to_crs(epsg=4326)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df.columns = df.columns.str.lower()
        
        if 'bande' not in df.columns or 'rdt' not in df.columns:
            st.error("Colonnes 'bande' ou 'rdt' manquantes.")
            st.stop()

        val_p = st.sidebar.selectbox("Bande 'Produit' ?", df['bande'].unique())
        df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')

        # Nettoyage
        if clean_outliers:
            clean_list = []
            for g in ['Produit', 'Témoin']:
                sub = df[df['grp'] == g]
                if not sub.empty:
                    q1, q3 = sub['rdt'].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    sub = sub[(sub['rdt'] >= q1 - 1.5*iqr) & (sub['rdt'] <= q3 + 1.5*iqr)]
                    clean_list.append(sub)
            df_final = pd.concat(clean_list)
        else:
            df_final = df.copy()

        data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
        data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()

        # Stats
        _, p_norm = stats.shapiro(data_p) if len(data_p) > 3 else (0, 0)
        _, p_student = stats.ttest_ind(data_p, data_t)
        gain = data_p.mean() - data_t.mean()

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("GAIN RDT", f"+{round(gain, 2)} qtx/ha")
        k2.metric("FIABILITÉ (P-VALUE)", f"{p_student:.4f}")
        k3.metric("MARGE NETTE", f"{round(((gain/10)*prix_vente)-cout_prod, 2)} €/ha")

        tab_rdt, tab_climat, tab_stats, tab_sol = st.tabs(["📊 Rendement", "🌦️ Climat & Stress", "🔬 Stat Expert", "🧪 Sol"])

        with tab_rdt:
            st.plotly_chart(px.box(df_final, x="grp", y="rdt", color="grp", points="all", title="Boxplot Rendements"), use_container_width=True)

        with tab_climat:
            st.subheader(f"🌦️ Analyse du Stress Thermique : {culture}")
            
            # 1. Préparation des paramètres
            lat, lon = gdf.geometry.y.mean(), gdf.geometry.x.mean()
            p_c = PARAM_CULTURES[culture]
            
            # Sécurité dates : Open-Meteo Archive s'arrête souvent à J-5
            date_max_archive = (datetime.now() - timedelta(days=5)).date()
            safe_end_date = min(d_recolte, date_max_archive)
            
            start_str = d_semis.strftime("%Y-%m-%d")
            end_str = safe_end_date.strftime("%Y-%m-%d")
            
            # URL de l'archive
            url_archive = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
            
            try:
                res = requests.get(url_archive)
                if res.status_code == 200:
                    data = res.json()
                    if 'daily' in data:
                        w_df = pd.DataFrame(data['daily'])
                        w_df['time'] = pd.to_datetime(w_df['time'])
                        
                        # Création du Graphique
                        fig_w = go.Figure()
                        
                        # Pluviométrie
                        fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], 
                                             name="Pluie (mm)", marker_color='rgba(0,0,255,0.2)'))
                        
                        # Courbe Température
                        fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['temperature_2m_max'], 
                                                 name="T° Max", line=dict(color='red', width=2)))
                        
                        # Zones de Stress Arvalis
                        fig_w.add_hrect(y0=p_c['echaudage'], y1=p_c['critique'], 
                                        fillcolor="orange", opacity=0.2, annotation_text="Zone d'échaudage")
                        fig_w.add_hrect(y0=p_c['critique'], y1=max(w_df['temperature_2m_max'].max(), 35), 
                                        fillcolor="red", opacity=0.3, annotation_text="Stress Sévère")
                        
                        # Événements
                        fig_w.add_vline(x=pd.to_datetime(d_appli), line_dash="dash", line_color="green", annotation_text="Application")
                        
                        fig_w.update_layout(title=f"Climat du {start_str} au {end_str}", hovermode="x unified")
                        st.plotly_chart(fig_w, use_container_width=True)

                        # Synthèse dynamique
                        nb_jours_stress = len(w_df[w_df['temperature_2m_max'] >= p_c['echaudage']])
                        st.info(f"💡 **Analyse Expert :** {nb_jours_stress} jours de stress thermique détectés sur la période.")
                    else:
                        st.warning("⚠️ L'API n'a pas renvoyé de données pour ces dates. Vérifie que la date de semis n'est pas trop ancienne.")
                else:
                    st.error(f"❌ Erreur API (Code {res.status_code}). Vérifie ta connexion ou les coordonnées GPS.")
                    
            except Exception as e:
                st.error(f"Erreur technique : {e}")

            # --- THÈSE ÉCRITE (Toujours visible pour ton MFE) ---
            st.markdown("---")
            st.markdown(f"""
            ### 🔬 Note de Synthèse pour le Mémoire
            **Aspect climatique et impact sur les résultats :**
            
            1. **Cinétique de l'échaudage :** Le blé est sensible dès **{p_c['echaudage']}°C**. Si le graphique montre des pics fréquents après l'application du produit, cela justifie l'utilisation de solutions de biostimulation ou de protection de la photosynthèse.
            2. **Remplissage du grain :** Un stress thermique majeur (>{p_c['critique']}°C) provoque un arrêt prématuré de la translocation des sucres vers l'épi, impactant directement le PMG (Poids de Mille Grains).
            3. **Interprétation du gain :** Si le gain de rendement est significatif malgré un fort stress thermique, le produit **BioExpert** a permis de maintenir la vacuité des vaisseaux conducteurs de la plante.
            """)

                    # --- SECTION THÈSE CLIMATIQUE ---
                    st.markdown("---")
                    st.header("🔬 Synthèse Climatique & Agronomique")
                    
                    nb_jours_echaudage = len(w_df[w_df['temperature_2m_max'] >= p_c['echaudage']])
                    
                    col_th1, col_th2 = st.columns(2)
                    with col_th1:
                        st.write("### 1. Analyse de l'Échaudage")
                        st.write(f"Durant le cycle, nous avons détecté **{nb_jours_echaudage} jours** au-dessus du seuil critique de {p_c['echaudage']}°C.")
                        st.write("D'après Arvalis, ces températures provoquent un raccourcissement de la phase de remplissage du grain. Si le gain de rendement est positif, le produit a probablement joué un rôle de **régulateur de stress abiotique**.")
                    
                    with col_th2:
                        st.write("### 2. Interaction Eau-Température")
                        pluie_totale = round(w_df['precipitation_sum'].sum(), 1)
                        st.write(f"Cumul de pluie sur le cycle : **{pluie_totale} mm**.")
                        st.write("Le stress thermique est souvent aggravé par un déficit hydrique. Un faible cumul de pluie durant les pics de chaleur valide l'importance de l'interface Bio-Expert pour cibler les zones à faible réserve utile.")

            except Exception as e:
                st.error(f"Erreur API : {e}")

        with tab_stats:
            st.header("🔬 Robustesse de la donnée")
            st.write(f"- Test de Normalité (Shapiro) : `p={p_norm:.4f}`")
            st.write(f"- Test de Student : `p={p_student:.4f}`")
            st.plotly_chart(px.histogram(df_final, x="rdt", color="grp", barmode="overlay"), use_container_width=True)

        with tab_sol:
            num_cols = df_final.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 1:
                st.plotly_chart(px.imshow(df_final[num_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")
