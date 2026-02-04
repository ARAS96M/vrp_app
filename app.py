__VERSION__ = "2026-02-04 reload-test"
import os
import importlib
from io import BytesIO

import streamlit as st
import pandas as pd

import vrp_script  # on importe le module (pas seulement la fonction)

st.set_page_config(page_title="Optimisation VRP", layout="wide")
st.title("🚚 Optimisation Trajet d'expédition")

# Optionnel: choix pour forcer reload (utile en dev)
force_reload = st.sidebar.checkbox("Forcer reload vrp_script", value=True)

fichier = st.file_uploader("Importer un fichier Excel", type=["xlsx"])

if fichier:
    chemin_temp = "fichier_temp.xlsx"
    with open(chemin_temp, "wb") as f:
        f.write(fichier.getbuffer())

    if st.button("▶️ Lancer"):
        try:
            # ✅ Evite l'effet "j'ai modifié vrp_script.py mais ça ne change rien"
            if force_reload:
                importlib.reload(vrp_script)

            # ✅ Afficher une signature de version si tu l’ajoutes dans vrp_script.py
            # (sinon ça affiche "inconnue")
            version = getattr(vrp_script, "__VERSION__", "inconnue")
            st.info(f"📌 Version vrp_script utilisée: {version}")

            with st.spinner("Calcul en cours..."):
                df_resultats, chemin_carte = vrp_script.run_vrp(chemin_temp)

            st.success("Terminé ✅")
            st.dataframe(df_resultats, use_container_width=True)

            # 📥 Téléchargement Excel
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
            if chemin_carte and os.path.exists(chemin_carte):
                with open(chemin_carte, "r", encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=600, width=1500)
            else:
                st.error(f"❌ Carte introuvable: {chemin_carte}")

            # ✅ Afficher le log si présent
            log_path = os.path.join("resultats_vrp", "vrp_log.txt")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as lf:
                    st.subheader("🧾 Log d'exécution")
                    st.code(lf.read())
            else:
                st.warning("ℹ️ Aucun log vrp_log.txt trouvé.")

        except Exception as e:
            st.error("❌ Erreur pendant l'exécution.")
            st.exception(e)

        finally:
            # nettoyage fichier temp
            try:
                os.remove(chemin_temp)
            except Exception:
                pass




