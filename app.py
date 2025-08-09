import streamlit as st
import pandas as pd
from vrp_script import run_vrp

st.title("🚚 Optimisation VRP")

fichier = st.file_uploader("Importer un fichier Excel", type=["xlsx"])

if fichier:
    chemin_temp = "fichier_temp.xlsx"
    with open(chemin_temp, "wb") as f:
        f.write(fichier.getbuffer())

    if st.button("▶️ Lancer"):
        with st.spinner("Calcul en cours..."):
            df_resultats, chemin_carte = run_vrp(chemin_temp)

        st.success("Terminé ✅")
        st.dataframe(df_resultats)

        st.download_button("📥 Télécharger Excel", df_resultats.to_excel(index=False), "resultats.xlsx")

        with open(chemin_carte, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=600)
