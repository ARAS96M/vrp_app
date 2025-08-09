import streamlit as st
import pandas as pd
from io import BytesIO
from vrp_script import run_vrp

st.set_page_config(page_title="Optimisation VRP", layout="wide")
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

        # 📥 Téléchargement Excel corrigé
        buffer = BytesIO()
        df_resultats.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Télécharger Excel",
            data=buffer,
            file_name="resultats.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 🗺 Affichage carte
        try:
            with open(chemin_carte, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=600, width=1500)
        except FileNotFoundError:
            st.error("❌ La carte n'a pas été trouvée. Vérifie que run_vrp génère bien un fichier HTML.")



