import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
import pulp
import folium

# === Fonction de distance Haversine ===
def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = radians(coord2[1]) - radians(coord1[1])
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# === Fonction principale ===
def run_vrp(fichier_excel):
    eps_km = 60
    min_clients = 2
    # max_total_capacite = 10
    time_limit = 60
    dossier_output = "resultats_vrp"
    os.makedirs(dossier_output, exist_ok=True)

    # Chargement
    clients_df = pd.read_excel(fichier_excel, sheet_name="client")
    vehicules_df = pd.read_excel(fichier_excel, sheet_name="vehicule")

    clients_df["Nom"] = clients_df["Nom"].astype(str).str.strip()
    vehicules_df["Nom"] = vehicules_df["Nom"].astype(str).str.strip()

    # Dépôt
    depot = clients_df[clients_df["Nom"].str.lower() == "depot"].iloc[0]
    clients = clients_df[clients_df["Nom"].str.lower() != "depot"].copy()

    capacite_max_vehicule = vehicules_df["Capacité"].max()

    # Fractionnement
    clients_fractionnes = []
    for _, row in clients.iterrows():
        if row["Demande"] > capacite_max_vehicule:
            q = row["Demande"]
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
    clients = pd.DataFrame(clients_fractionnes)

    # Clustering
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

    coord_dict = dict(zip(clients_df["Nom"], zip(clients_df["Latitude"], clients_df["Longitude"])))
    for _, row in clients.iterrows():
        coord_dict[row["Nom"]] = (row["Latitude"], row["Longitude"])

    def resoudre_cluster(cluster_id):
        groupe = clients[clients["Cluster"] == cluster_id]
        noms_clients = groupe["Nom"].tolist()
        noms_avec_depot = ["depot"] + noms_clients

        q = dict(zip(clients["Nom"], clients["Demande"]))
        a = dict(zip(clients["Nom"], clients["Disponible"]))
        o = dict(zip(vehicules_df["Nom"], vehicules_df["Capacité"]))

        # Liste des noms réels des véhicules depuis l'Excel
        V_all = vehicules_df["Nom"].tolist()
        C = noms_clients
        N = noms_avec_depot

        demande_totale = sum(q[n] for n in C if a[n] == 1)
        capacite_cumulee = 0
        V = []
        for idx, v in enumerate(V_all):
            capacite_cumulee += vehicules_df.iloc[idx]["Capacité"]
            V.append(v)
            if capacite_cumulee >= min(demande_totale, max_total_capacite):
                break

        d = {(i, j): haversine(coord_dict[i], coord_dict[j]) for i in N for j in N if i != j}

        model = pulp.LpProblem(f"VRP_Cluster_{cluster_id}", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", ((k, i, j) for k in V for i in N for j in N if i != j), cat='Binary')
        u = pulp.LpVariable.dicts("u", ((k, j) for k in V for j in C), lowBound=0, cat='Continuous')

        model += pulp.lpSum(d[i, j] * x[k, i, j] for k in V for i, j in d)

        for j in C:
            if a[j] == 1:
                model += pulp.lpSum(x[k, i, j] for k in V for i in N if i != j) == 1
                model += pulp.lpSum(x[k, j, i] for k in V for i in N if i != j) == 1

        for k in V:
            for h in N:
                model += pulp.lpSum(x[k, i, h] for i in N if i != h) == pulp.lpSum(x[k, h, j] for j in N if j != h)
            model += pulp.lpSum(q[j] * x[k, i, j] for j in C for i in N if i != j) <= o[k]

            for i in C:
                for j in C:
                    if i != j:
                        model += u[k, j] >= u[k, i] + q[j] - (1 - x[k, i, j]) * o[k]
            for j in C:
                model += u[k, j] >= q[j]
                model += u[k, j] <= o[k]

        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

        trajets = []
        for k in V:
            for i, j in d:
                if pulp.value(x[k, i, j]) > 0.5:
                    trajets.append({"Cluster": cluster_id, "Véhicule": k, "Départ": i, "Arrivée": j})
        return trajets

    trajets_par_cluster = []
    for cid in clients["Cluster"].unique():
        trajets = resoudre_cluster(cid)
        trajets_par_cluster.append(trajets)

    df_trajets = pd.DataFrame([t for cluster in trajets_par_cluster for t in cluster])

    df_trajets_ordonnes = pd.DataFrame()
    for cluster_id in df_trajets['Cluster'].unique():
        cluster_df = df_trajets[df_trajets['Cluster'] == cluster_id].copy()
        for vehicle in cluster_df['Véhicule'].unique():
            vehicle_df = cluster_df[cluster_df['Véhicule'] == vehicle].copy()
            chemin = []
            current = 'depot'
            while True:
                next_rows = vehicle_df[vehicle_df['Départ'] == current]
                if next_rows.empty:
                    break
                row = next_rows.iloc[0]
                chemin.append(row)
                current = row['Arrivée']
                vehicle_df = vehicle_df.drop(index=next_rows.index)
                if current == 'depot':
                    break
            if chemin:
                ordered_df = pd.DataFrame(chemin)
                ordered_df['Distance (KM)'] = ordered_df.apply(lambda r: haversine(coord_dict[r['Départ']], coord_dict[r['Arrivée']]), axis=1)
                df_trajets_ordonnes = pd.concat([df_trajets_ordonnes, ordered_df])

    fichier_resultat = os.path.join(dossier_output, "trajets_vrp_ordonnes_distances.xlsx")
    df_trajets_ordonnes.to_excel(fichier_resultat, index=False)

    map_center = coord_dict["depot"]
    m = folium.Map(location=map_center, zoom_start=8, width="100%")  # largeur max
    for nom, (lat, lon) in coord_dict.items():
        folium.Marker([lat, lon], popup=nom, icon=folium.Icon(color="red" if nom == "depot" else "blue")).add_to(m)

    colors = ["blue", "orange", "green", "purple", "darkred", "darkblue", "cadetblue"]
    cluster_ids = df_trajets_ordonnes['Cluster'].unique()
    cluster_to_color_index = {cid: idx % len(colors) for idx, cid in enumerate(cluster_ids)}

    for _, row in df_trajets_ordonnes.iterrows():
        cid = row['Cluster']
        i = row['Départ']
        j = row['Arrivée']
        folium.PolyLine([coord_dict[i], coord_dict[j]],
                        color=colors[cluster_to_color_index[cid]],
                        weight=2.5,
                        tooltip=f"{row['Véhicule']}: {i} → {j} ({row['Distance (KM)']:.2f} km)").add_to(m)

    fichier_carte = os.path.join(dossier_output, "carte_clusters_vrp_ordonnee.html")
    m.save(fichier_carte)

    return df_trajets_ordonnes, fichier_carte

