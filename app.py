import streamlit as st
import pandas as pd
import requests
import json
from ortools.sat.python import cp_model

st.title("📚 Générateur d'Emploi du Temps Intelligent")
st.write("Chargez vos données CSV et décrivez vos contraintes en français")

# Upload de fichiers CSV
st.sidebar.header("📁 Données d'entrée")
profs_file = st.sidebar.file_uploader("Professeurs (CSV)", type="csv")
classes_file = st.sidebar.file_uploader("Classes (CSV)", type="csv")

if profs_file and classes_file:
    st.success("✅ Fichiers chargés avec succès!")
    
    # Afficher un aperçu
    profs_df = pd.read_csv(profs_file)
    st.write("Aperçu des professeurs:", profs_df.head())