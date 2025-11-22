import streamlit as st
import pandas as pd

st.title("🎓 Générateur d'Emploi du Temps")
st.write("Bienvenue dans notre application !")

# Upload de fichiers
st.sidebar.header("📁 Données")
profs_file = st.sidebar.file_uploader("Professeurs CSV", type="csv")

if profs_file:
    df = pd.read_csv(profs_file)
    st.write("Aperçu des données :")
    st.dataframe(df)
    
    st.success(f"✅ {len(df)} professeurs chargés !")

st.info("L'application se développera ici étape par étape !")
