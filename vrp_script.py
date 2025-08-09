# vrp_script.py
import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
import pulp
import folium
import traceback

# === Fonction de distance Haversine ===
def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# === Fonction pour formater les trajets en tableau final ===
def formater_trajets(df):
    # On enlève les colonnes inutiles
    df = df.drop(columns=["Cluster", "Véhicule"], errors="ignore").reset_index(drop=True)

    # Création de la nouvelle colonne Véhicule
    vehicules = []
    current_num = 1

    for idx, row in df.iterrows():
        # Ajout du véhicule courant
        vehicules.append(f"V{current_num}")

        # Règle de changement de véhicule
        if "-part" in str(row["Départ"]) or "-part" in str(row["Arrivée"]):
            current_num += 1
        elif row["Départ"].lower() == "depot" and idx != 0:
            current_num += 1

    df.insert(0, "Véhicule", vehicules)

    return df

# === Fonction principale ===
def run_vrp(fichier_excel):
    try:
        eps_km = 60
        min_clients = 2
        max_total_capacite = 10
        time_limit = 60
        dossier_output = "resultats_vrp"
        os.makedirs(dossier_output, exist_ok=True)

        log_lines = []
        def log(s):
            log_lines.append(str(s))

        # --- Chargement des feuilles ---
        xls = pd.ExcelFile(fichier_excel)
        if "client" not in xls.sheet_names or "vehicule" not in xls.sheet_names:
            raise ValueError("Le fichier Excel doit contenir les feuilles 'client' et 'vehicule'.")

        clients_df = pd.read_excel(xls, sheet_name="client")
        vehicules_df = pd.read_excel(xls, sheet_name="vehicule")

        # --- Normalisation des noms de colonnes (gestion accents / casse) ---
        def find_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        # colonnes attendues pour clients
        col_nom = find_col(clients_df, ["Nom", "nom", "NOM"])
        col_lat = find_col(clients_df, ["Latitude", "latitude", "LATITUDE", "Lat"])
        col_lon = find_col(clients_df, ["Longitude", "longitude", "LONGITUDE", "Lon"])
        col_dem = find_col(clients_df, ["Demande", "demande", "DEMANDE", "Qte", "Quantité"])
        col_disp = find_col(clients_df, ["Disponible", "disponible", "DISPONIBLE"])
        col_wilaya = find_col(clients_df, ["Wilaya", "wilaya", "WILAYA"])
        col_zone = find_col(clients_df, ["ZONE", "Zone", "zone"])

        if not all([col_nom, col_lat, col_lon, col_dem, col_disp, col_wilaya, col_zone]):
            raise ValueError("Colonnes client manquantes. Attendues: Nom, Latitude, Longitude, Demande, Disponible, Wilaya, ZONE.")

        # colonnes attendues pour vehicules
        col_v_nom = find_col(vehicules_df, ["Nom", "nom", "NOM"])
        col_v_cap = find_col(vehicules_df, ["Capacité", "Capacite", "capacite", "CAPACITE", "Capacité", "Capacité"])

        if not all([col_v_nom, col_v_cap]):
            raise ValueError("Colonnes vehicule manquantes. Attendues: Nom, Capacité.")

        # renommer colonnes pour usage interne
        clients_df = clients_df.rename(columns={
            col_nom: "Nom",
            col_lat: "Latitude",
            col_lon: "Longitude",
            col_dem: "Demande",
            col_disp: "Disponible",
            col_wilaya: "Wilaya",
            col_zone: "ZONE"
        })
        vehicules_df = vehicules_df.rename(columns={col_v_nom: "Nom", col_v_cap: "Capacite"})

        # nettoyage des valeurs
        clients_df["Nom"] = clients_df["Nom"].astype(str).str.strip()
        vehicules_df["Nom"] = vehicules_df["Nom"].astype(str).str.strip()
        clients_df["Latitude"] = pd.to_numeric(clients_df["Latitude"], errors="coerce")
        clients_df["Longitude"] = pd.to_numeric(clients_df["Longitude"], errors="coerce")
        clients_df["Demande"] = pd.to_numeric(clients_df["Demande"], errors="coerce").fillna(0)
        clients_df["Disponible"] = pd.to_numeric(clients_df["Disponible"], errors="coerce").fillna(0).astype(int)
        vehicules_df["Capacite"] = pd.to_numeric(vehicules_df["Capacite"], errors="coerce").fillna(0)

        log("Colonnes et types vérifiés.")

        # --- Trouver le depot (nom exact) ---
        depot_rows = clients_df[clients_df["Nom"].str.lower() == "depot"]
        if depot_rows.empty:
            raise ValueError("Aucun enregistrement 'depot' trouvé dans la feuille client (colonne 'Nom').")
        depot = depot_rows.iloc[0]
        depot_name = depot["Nom"]

        # --- Séparer clients ---
        clients = clients_df[clients_df["Nom"].str.lower() != "depot"].copy()

        # --- Fractionnement si dépassement capacité max individuelle ---
        capacite_max_vehicule = int(vehicules_df["Capacite"].max())
        clients_fractionnes = []
        for _, row in clients.iterrows():
            if row["Demande"] > capacite_max_vehicule and capacite_max_vehicule > 0:
                q = int(row["Demande"])
                part_num = 1
                while q > 0:
                    quantite = min(capacite_max_vehicule, q)
                    new_row = row.copy()
                    new_row["Nom"] = f"{row['Nom']}_part{part_num}"
                    new_row["Demande"] = quantite
                    clients_fractionnes.append(new_row)
                    q -= quantite
                    part_num += 1
            else:
                clients_fractionnes.append(row)
        clients = pd.DataFrame(clients_fractionnes).reset_index(drop=True)

        # --- Clustering par zone/wilaya ---
        wilayas_specifiques = ["ALGER", "BOUMERDES", "BLIDA", "TIPAZA"]
        clients["Cluster"] = -1
        clients["Cluster"] = clients["Cluster"].astype("object")

        for wilaya in wilayas_specifiques:
            clients_wilaya = clients[clients["Wilaya"] == wilaya].copy()
            if not clients_wilaya.empty:
                clients.loc[clients["Wilaya"] == wilaya, "Cluster"] = clients_wilaya["ZONE"].apply(lambda x: f"{wilaya}_{x}")

        clients_autres_wilayas = clients[~clients["Wilaya"].isin(wilayas_specifiques)].copy()
        if not clients_autres_wilayas.empty:
            coords_autres = clients_autres_wilayas[["Latitude", "Longitude"]].values
            clustering_autres = DBSCAN(eps=eps_km / 111, min_samples=min_clients).fit(coords_autres)
            offset = 1000
            clients.loc[~clients["Wilaya"].isin(wilayas_specifiques), "Cluster"] = (clustering_autres.labels_ + offset).astype(str)

        log("Clustering terminé.")

        # --- Coordonnées dict (inclut depot et clients fractionnés) ---
        coord_dict = {}
        # depuis la feuille client complète (inclut depot)
        for _, row in clients_df.iterrows():
            coord_dict[row["Nom"]] = (row["Latitude"], row["Longitude"])
        # pour les clients fractionnés (si noms modifiés)
        for _, row in clients.iterrows():
            coord_dict[row["Nom"]] = (row["Latitude"], row["Longitude"])

        # --- Capacités par véhicule dict ---
        capacites_vehicules = dict(zip(vehicules_df["Nom"].tolist(), vehicules_df["Capacite"].tolist()))
        liste_vehicules = vehicules_df["Nom"].tolist()

        # --- Fonction de résolution d'un cluster ---
        def resoudre_cluster(cluster_id):
            groupe = clients[clients["Cluster"] == cluster_id]
            if groupe.empty:
                return []

            noms_clients = groupe["Nom"].tolist()
            noms_avec_depot = [depot_name] + noms_clients

            q = dict(zip(clients["Nom"], clients["Demande"]))
            a = dict(zip(clients["Nom"], clients["Disponible"]))

            # Affectation stricte des véhicules selon la feuille vehicule (ordre fixe)
            V_all = liste_vehicules

            # Déterminer combien de véhicules nécessaires selon la demande et la capacité max (max_total_capacite)
            demande_totale = sum(q[n] for n in noms_clients if a.get(n, 0) == 1)
            capacite_cumulee = 0
            V = []
            for idx, v in enumerate(V_all):
                capacite_cumulee += capacites_vehicules[v]
                V.append(v)
                if capacite_cumulee >= max(1, min(demande_totale, max_total_capacite)):
                    break
            if not V:
                # au moins 1 véhicule si existant
                if len(V_all) > 0:
                    V = [V_all[0]]
                else:
                    raise ValueError("Aucun véhicule disponible dans la feuille 'vehicule'.")

            # matrice des distances
            d = {(i, j): haversine(coord_dict[i], coord_dict[j]) for i in noms_avec_depot for j in noms_avec_depot if i != j}

            # --- Modèle pulp ---
            model = pulp.LpProblem(f"VRP_Cluster_{cluster_id}", pulp.LpMinimize)
            x = pulp.LpVariable.dicts("x", (V, noms_avec_depot, noms_avec_depot), cat='Binary')
            u = pulp.LpVariable.dicts("u", (V, noms_clients), lowBound=0, cat='Continuous')

            # Objectif
            model += pulp.lpSum(d[i, j] * x[k][i][j] for k in V for (i, j) in d)

            # Chaque client servi une fois (si disponible)
            for j in noms_clients:
                if a.get(j, 0) == 1:
                    model += pulp.lpSum(x[k][i][j] for k in V for i in noms_avec_depot if i != j) == 1
                    model += pulp.lpSum(x[k][j][i] for k in V for i in noms_avec_depot if i != j) == 1

            # Conservation de flux et capacités
            for k in V:
                for h in noms_avec_depot:
                    model += pulp.lpSum(x[k][i][h] for i in noms_avec_depot if i != h) == pulp.lpSum(x[k][h][j] for j in noms_avec_depot if j != h)
                # Capacité véhicule k
                model += pulp.lpSum(q[j] * x[k][i][j] for j in noms_clients for i in noms_avec_depot if i != j) <= capacites_vehicules[k]

                # Contraintes subtour (Miller-Tucker-Zemlin)
                for i in noms_clients:
                    for j in noms_clients:
                        if i != j:
                            model += u[k][j] >= u[k][i] + q[j] - (1 - x[k][i][j]) * capacites_vehicules[k]
                for j in noms_clients:
                    model += u[k][j] >= q[j]
                    model += u[k][j] <= capacites_vehicules[k]

            # Solve
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

            trajets = []
            for k in V:
                for i in noms_avec_depot:
                    for j in noms_avec_depot:
                        if i != j and pulp.value(x[k][i][j]) is not None and pulp.value(x[k][i][j]) > 0.5:
                            trajets.append({"Cluster": cluster_id, "Véhicule": k, "Départ": i, "Arrivée": j, "Distance (KM)": d[(i, j)]})
            return trajets

        # --- Résolution pour chaque cluster ---
        trajets_par_cluster = []
        for cid in clients["Cluster"].unique():
            trajets = resoudre_cluster(cid)
            trajets_par_cluster.append(trajets)

        df_trajets = pd.DataFrame([t for cluster in trajets_par_cluster for t in cluster])

        # --- Ordonnancement / ré-ordonnage des chemins par véhicule ---
        df_trajets_ordonnes = pd.DataFrame()
        if not df_trajets.empty:
            for cluster_id in df_trajets['Cluster'].unique():
                cluster_df = df_trajets[df_trajets['Cluster'] == cluster_id].copy()
                for vehicle in cluster_df['Véhicule'].unique():
                    vehicle_df = cluster_df[cluster_df['Véhicule'] == vehicle].copy()
                    chemin = []
                    current = depot_name
                    # boucle jusqu'à épuisement
                    while True:
                        next_rows = vehicle_df[vehicle_df['Départ'] == current]
                        if next_rows.empty:
                            break
                        row = next_rows.iloc[0]
                        chemin.append(row)
                        current = row['Arrivée']
                        vehicle_df = vehicle_df.drop(index=next_rows.index)
                        if current == depot_name:
                            break
                    if chemin:
                        ordered_df = pd.DataFrame(chemin)
                        ordered_df['Distance (KM)'] = ordered_df.apply(lambda r: haversine(coord_dict[r['Départ']], coord_dict[r['Arrivée']]), axis=1)
                        df_trajets_ordonnes = pd.concat([df_trajets_ordonnes, ordered_df], ignore_index=True)

        # --- Formater le tableau final ---
        df_final = formater_trajets(df_trajets_ordonnes)

        # sauvegarde Excel
        fichier_resultat = os.path.join(dossier_output, "trajets_vrp_ordonnes_distances.xlsx")
        if not df_final.empty:
            df_final.to_excel(fichier_resultat, index=False)
            log(f"Fichier résultat enregistré: {fichier_resultat}")
        else:
            # fichier vide cohérence
            pd.DataFrame(columns=["Véhicule", "Départ", "Arrivée", "Distance (KM)"]).to_excel(fichier_resultat, index=False)
            log("Aucun trajet trouvé — fichier résultat vide créé.")

        # Carte Folium
        if depot_name not in coord_dict:
            raise KeyError(f"Coordonnées du dépôt ({depot_name}) introuvables.")
        map_center = coord_dict[depot_name]
        m = folium.Map(location=map_center, zoom_start=8, width="100%")
        for nom, coords in coord_dict.items():
            lat, lon = coords
            if pd.isna(lat) or pd.isna(lon):
                continue
            folium.Marker([lat, lon], popup=str(nom), icon=folium.Icon(color="red" if str(nom).lower() == "depot" else "blue")).add_to(m)

        colors = ["blue", "orange", "green", "purple", "darkred", "darkblue", "cadetblue"]
        cluster_ids = df_trajets_ordonnes['Cluster'].unique() if not df_trajets_ordonnes.empty else []
        cluster_to_color_index = {cid: idx % len(colors) for idx, cid in enumerate(cluster_ids)}

        if not df_trajets_ordonnes.empty:
            for _, row in df_trajets_ordonnes.iterrows():
                cid = row['Cluster']
                i = row['Départ']
                j = row['Arrivée']
                if i in coord_dict and j in coord_dict:
                    folium.PolyLine([coord_dict[i], coord_dict[j]],
                                    color=colors[cluster_to_color_index.get(cid, 0)],
                                    weight=2.5,
                                    tooltip=f"{row['Véhicule']}: {i} → {j} ({row['Distance (KM)']:.2f} km)").add_to(m)

        fichier_carte = os.path.join(dossier_output, "carte_clusters_vrp_ordonnee.html")
        m.save(fichier_carte)
        log(f"Carte enregistrée: {fichier_carte}")

        # write log
        with open(os.path.join(dossier_output, "vrp_log.txt"), "w", encoding="utf-8") as lf:
            lf.write("\n".join(log_lines))

        return df_final, fichier_carte

    except Exception as e:
        # log erreur complète pour debug
        os.makedirs("resultats_vrp", exist_ok=True)
        with open(os.path.join("resultats_vrp", "vrp_error.txt"), "w", encoding="utf-8") as ef:
            ef.write("ERREUR RUN VRP\n")
            ef.write(str(e) + "\n\n")
            ef.write(traceback.format_exc())
        raise  # remonter erreur

