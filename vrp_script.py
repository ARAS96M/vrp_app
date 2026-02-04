__VERSION__ = "2026-02-04 v3"
print(">>> VRP_SCRIPT LOADED VERSION =", __VERSION__)

import pandas as pd
import numpy as np
import os
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
import pulp
import folium
import traceback


# === Distance Haversine (KM) ===
def haversine(coord1, coord2):
    R = 6371.0
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# === Format trajets ===
def formater_trajets(df):
    """
    Retourne (df_display, df_for_map)

    Règles :
    - Si '_part' (ou '-part') dans Départ ou Arrivée -> incrément immédiat et la ligne reçoit le nouveau véhicule.
    - Si Départ == 'depot' (et idx != 0) -> incrément avant d'assigner la ligne.
    """
    df2 = df.copy().reset_index(drop=True)

    if "Véhicule" in df2.columns:
        df2 = df2.drop(columns=["Véhicule"])

    numeros = []
    current_num = 1

    for idx, row in df2.iterrows():
        dep = str(row.get("Départ", "")).strip().lower()
        arr = str(row.get("Arrivée", "")).strip().lower()

        is_part = ("_part" in dep) or ("_part" in arr) or ("-part" in dep) or ("-part" in arr)

        if is_part:
            current_num += 1
            numeros.append(current_num)
            continue

        if dep == "depot" and idx != 0:
            current_num += 1
            numeros.append(current_num)
            continue

        numeros.append(current_num)

    vehs = [f"V{n}" for n in numeros]
    df2.insert(0, "Véhicule", vehs)

    df_for_map = df2.copy()

    cols_keep = [c for c in ["Véhicule", "Départ", "Arrivée", "Distance (KM)"] if c in df2.columns]
    df_display = df2[cols_keep].copy().reset_index(drop=True)

    return df_display, df_for_map


def run_vrp(fichier_excel):
    try:
        # ====== Paramètres ======
        eps_km = 80               # DBSCAN en km (vrai)
        min_clients = 2
        time_limit = 60
        dossier_output = "resultats_vrp"
        os.makedirs(dossier_output, exist_ok=True)

        # Pénalité forte => évite d'utiliser 2 véhicules si 1 suffit
        VEHICLE_PENALTY = 5_000_000

        log_lines = []
        def log(s):
            log_lines.append(str(s))

        # --- Chargement ---
        xls = pd.ExcelFile(fichier_excel)
        if "client" not in xls.sheet_names or "vehicule" not in xls.sheet_names:
            raise ValueError("Le fichier Excel doit contenir les feuilles 'client' et 'vehicule'.")

        clients_df = pd.read_excel(xls, sheet_name="client")
        vehicules_df = pd.read_excel(xls, sheet_name="vehicule")

        # --- Normalisation colonnes ---
        def find_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        col_nom = find_col(clients_df, ["Nom", "nom", "NOM"])
        col_lat = find_col(clients_df, ["Latitude", "latitude", "LATITUDE", "Lat"])
        col_lon = find_col(clients_df, ["Longitude", "longitude", "LONGITUDE", "Lon"])
        col_dem = find_col(clients_df, ["Demande", "demande", "DEMANDE", "Qte", "Quantité"])
        col_disp = find_col(clients_df, ["Disponible", "disponible", "DISPONIBLE"])
        col_wilaya = find_col(clients_df, ["Wilaya", "wilaya", "WILAYA"])
        col_zone = find_col(clients_df, ["ZONE", "Zone", "zone"])

        if not all([col_nom, col_lat, col_lon, col_dem, col_disp, col_wilaya, col_zone]):
            raise ValueError("Colonnes client manquantes. Attendues: Nom, Latitude, Longitude, Demande, Disponible, Wilaya, ZONE.")

        col_v_nom = find_col(vehicules_df, ["Nom", "nom", "NOM"])
        col_v_cap = find_col(vehicules_df, ["Capacité", "Capacite", "capacite", "CAPACITE"])

        if not all([col_v_nom, col_v_cap]):
            raise ValueError("Colonnes vehicule manquantes. Attendues: Nom, Capacité.")

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

        # --- Nettoyage ---
        clients_df["Nom"] = clients_df["Nom"].astype(str).str.strip()
        vehicules_df["Nom"] = vehicules_df["Nom"].astype(str).str.strip()

        clients_df["Wilaya"] = clients_df["Wilaya"].astype(str).str.strip().str.upper()
        clients_df["ZONE"] = clients_df["ZONE"].astype(str).str.strip()

        clients_df["Latitude"] = pd.to_numeric(clients_df["Latitude"], errors="coerce")
        clients_df["Longitude"] = pd.to_numeric(clients_df["Longitude"], errors="coerce")
        clients_df["Demande"] = pd.to_numeric(clients_df["Demande"], errors="coerce").fillna(0.0).astype(float)
        clients_df["Disponible"] = pd.to_numeric(clients_df["Disponible"], errors="coerce").fillna(0).astype(int)

        vehicules_df["Capacite"] = pd.to_numeric(vehicules_df["Capacite"], errors="coerce").fillna(0.0).astype(float)
        vehicules_df = vehicules_df[vehicules_df["Capacite"] > 0].copy()
        if vehicules_df.empty:
            raise ValueError("Aucun véhicule avec capacité > 0.")

        log("Colonnes et types OK + nettoyage Wilaya/ZONE OK.")

        # --- Depot ---
        depot_rows = clients_df[clients_df["Nom"].str.lower() == "depot"]
        if depot_rows.empty:
            raise ValueError("Aucun 'depot' trouvé dans la feuille client.")
        depot = depot_rows.iloc[0]
        depot_name = depot["Nom"]

        # --- Séparer clients ---
        clients = clients_df[clients_df["Nom"].str.lower() != "depot"].copy()

        # --- Fractionnement > capacité max ---
        capacite_max_vehicule = float(vehicules_df["Capacite"].max())
        clients_fractionnes = []
        for _, row in clients.iterrows():
            dem = float(row["Demande"])
            if dem > capacite_max_vehicule + 1e-9:
                q = dem
                part_num = 1
                while q > 1e-9:
                    quantite = min(capacite_max_vehicule, q)
                    new_row = row.copy()
                    new_row["Nom"] = f"{row['Nom']}_part{part_num}"
                    new_row["Demande"] = float(quantite)
                    clients_fractionnes.append(new_row)
                    q -= quantite
                    part_num += 1
            else:
                clients_fractionnes.append(row)

        clients = pd.DataFrame(clients_fractionnes).reset_index(drop=True)

        # --- Clustering ---
        wilayas_specifiques = ["ALGER", "BOUMERDES", "BLIDA", "TIPAZA"]
        clients["Cluster"] = None
        clients["Cluster"] = clients["Cluster"].astype("object")

        # Cas spécial wilayas spécifiques
        for wilaya in wilayas_specifiques:
            mask_w = clients["Wilaya"] == wilaya
            if mask_w.any():
                clients.loc[mask_w, "Cluster"] = clients.loc[mask_w, "ZONE"].apply(
                    lambda z: f"{wilaya}_{z if str(z).strip() not in ['', 'nan', 'None'] else 'NA'}"
                )

        # Autres wilayas: DBSCAN en HAVERSINE
        mask_autres = ~clients["Wilaya"].isin(wilayas_specifiques)
        clients_autres = clients[mask_autres].copy()

        if not clients_autres.empty:
            valid_mask = clients_autres[["Latitude", "Longitude"]].notna().all(axis=1)
            invalid = clients_autres[~valid_mask]
            valid = clients_autres[valid_mask]

            for idx, row in invalid.iterrows():
                clients.loc[idx, "Cluster"] = f"NOCOORD_{row['Nom']}"

            if not valid.empty:
                coords_rad = np.radians(valid[["Latitude", "Longitude"]].values)
                eps_rad = eps_km / 6371.0

                clustering = DBSCAN(
                    eps=eps_rad,
                    min_samples=min_clients,
                    metric="haversine"
                ).fit(coords_rad)

                labels = clustering.labels_
                for idx_row, label in zip(valid.index, labels):
                    if label == -1:
                        clients.loc[idx_row, "Cluster"] = f"NOISE_{clients.loc[idx_row, 'Nom']}"
                    else:
                        clients.loc[idx_row, "Cluster"] = f"DB_{label}"

        # Sécurité si None
        none_mask = clients["Cluster"].isna()
        if none_mask.any():
            for idx, row in clients[none_mask].iterrows():
                clients.loc[idx, "Cluster"] = f"UNCL_{row['Nom']}"

        log(f"Clustering terminé (eps_km={eps_km}, haversine).")
        log("Clusters trouvés: " + ", ".join(map(str, clients["Cluster"].unique().tolist())))

        # --- Coordonnées ---
        coord_dict = {depot_name: (float(depot["Latitude"]), float(depot["Longitude"]))}
        for _, row in clients.iterrows():
            coord_dict[row["Nom"]] = (float(row["Latitude"]), float(row["Longitude"]))

        # --- Véhicules ---
        vehicules_df = vehicules_df.sort_values("Capacite", ascending=False).reset_index(drop=True)
        capacites_vehicules = dict(zip(vehicules_df["Nom"].tolist(), vehicules_df["Capacite"].tolist()))
        liste_vehicules = vehicules_df["Nom"].tolist()

        # --- Résoudre un cluster ---
        def resoudre_cluster(cluster_id):
            groupe_all = clients[clients["Cluster"] == cluster_id].copy()
            if groupe_all.empty:
                return []

            groupe = groupe_all[groupe_all["Disponible"] == 1].copy()
            if groupe.empty:
                log(f"Cluster {cluster_id}: aucun client Disponible==1")
                return []

            noms_clients = groupe["Nom"].tolist()
            q = dict(zip(groupe["Nom"], groupe["Demande"]))
            demande_totale = float(sum(q[n] for n in noms_clients))

            log(f"[DEBUG] Cluster={cluster_id} nb_clients={len(noms_clients)} demande_totale={demande_totale:.3f}")
            log(f"[DEBUG] Cluster={cluster_id} clients={noms_clients}")

            capacite_totale = float(sum(capacites_vehicules.values()))
            if capacite_totale + 1e-9 < demande_totale:
                log(f"Cluster {cluster_id}: INFAISABLE (demande {demande_totale:.3f} > cap_totale {capacite_totale:.3f})")
                return []

            V = liste_vehicules
            noms_avec_depot = [depot_name] + noms_clients

            # distances
            d = {}
            for i in noms_avec_depot:
                for j in noms_avec_depot:
                    if i == j:
                        continue
                    if i not in coord_dict or j not in coord_dict:
                        continue
                    if any(pd.isna(x) for x in coord_dict[i]) or any(pd.isna(x) for x in coord_dict[j]):
                        continue
                    d[(i, j)] = haversine(coord_dict[i], coord_dict[j])

            model = pulp.LpProblem(f"VRP_Cluster_{cluster_id}", pulp.LpMinimize)

            x = pulp.LpVariable.dicts("x", (V, noms_avec_depot, noms_avec_depot), lowBound=0, upBound=1, cat="Binary")
            y = pulp.LpVariable.dicts("y", V, lowBound=0, upBound=1, cat="Binary")
            u = pulp.LpVariable.dicts("u", (V, noms_clients), lowBound=0, cat="Continuous")

            model += (
                VEHICLE_PENALTY * pulp.lpSum(y[k] for k in V)
                + pulp.lpSum(d[(i, j)] * x[k][i][j] for k in V for (i, j) in d)
            )

            for k in V:
                for i in noms_avec_depot:
                    model += x[k][i][i] == 0

            # clients 1 fois
            for j in noms_clients:
                model += pulp.lpSum(x[k][i][j] for k in V for i in noms_avec_depot if i != j) == 1
                model += pulp.lpSum(x[k][j][i] for k in V for i in noms_avec_depot if i != j) == 1

            for k in V:
                cap_k = float(capacites_vehicules.get(k, 0.0))

                # 1 tournée max par véhicule (sinon extraction peut devenir compliquée)
                model += pulp.lpSum(x[k][depot_name][j] for j in noms_clients) <= 1
                model += pulp.lpSum(x[k][i][depot_name] for i in noms_clients) <= 1

                # arcs -> y
                for i in noms_avec_depot:
                    for j in noms_avec_depot:
                        if i != j:
                            model += x[k][i][j] <= y[k]

                # flow
                for h in noms_avec_depot:
                    model += (
                        pulp.lpSum(x[k][i][h] for i in noms_avec_depot if i != h)
                        == pulp.lpSum(x[k][h][j] for j in noms_avec_depot if j != h)
                    )

                # capacité
                model += (
                    pulp.lpSum(q[j] * pulp.lpSum(x[k][i][j] for i in noms_avec_depot if i != j) for j in noms_clients)
                    <= cap_k
                )

                # bornes u
                for j in noms_clients:
                    incoming_kj = pulp.lpSum(x[k][i][j] for i in noms_avec_depot if i != j)
                    model += u[k][j] >= q[j] * incoming_kj
                    model += u[k][j] <= cap_k * incoming_kj

                # MTZ
                for i in noms_clients:
                    for j in noms_clients:
                        if i != j:
                            model += u[k][j] >= u[k][i] + q[j] - cap_k * (1 - x[k][i][j])

            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
            model.solve(solver)

            status = pulp.LpStatus.get(model.status, str(model.status))
            log(f"[DEBUG] Cluster={cluster_id} status={status}")

            if status in ["Infeasible", "Unbounded", "Undefined"]:
                return []

            trajets = []
            for k in V:
                for i in noms_avec_depot:
                    for j in noms_avec_depot:
                        if i == j:
                            continue
                        val = pulp.value(x[k][i][j])
                        if val is not None and val > 0.5 and (i, j) in d:
                            trajets.append({
                                "Cluster": cluster_id,
                                "Véhicule": k,
                                "Départ": i,
                                "Arrivée": j,
                                "Distance (KM)": float(d[(i, j)])
                            })

            log(f"Cluster {cluster_id}: {len(trajets)} arcs extraits.")
            return trajets

        # --- Résolution ---
        trajets_par_cluster = []
        for cid in clients["Cluster"].unique():
            trajets_par_cluster.append(resoudre_cluster(cid))

        df_trajets = pd.DataFrame([t for cluster in trajets_par_cluster for t in cluster])

        # --- Ordonnancement ---
        def ordonner_trajets(df_in):
            if df_in.empty:
                return df_in

            out = []
            for cluster_id in df_in["Cluster"].unique():
                df_c = df_in[df_in["Cluster"] == cluster_id].copy()
                for vehicle in df_c["Véhicule"].unique():
                    df_v = df_c[df_c["Véhicule"] == vehicle].copy()

                    # mapping départ -> arrivée
                    next_map = {}
                    for _, r in df_v.iterrows():
                        next_map[str(r["Départ"])] = (str(r["Arrivée"]), float(r["Distance (KM)"]))

                    if depot_name not in next_map:
                        continue

                    cur = depot_name
                    visited = set()

                    while cur in next_map:
                        nxt, dist = next_map[cur]
                        edge = (cur, nxt)
                        if edge in visited:
                            break
                        visited.add(edge)

                        out.append({
                            "Cluster": cluster_id,
                            "Véhicule": vehicle,
                            "Départ": cur,
                            "Arrivée": nxt,
                            "Distance (KM)": dist
                        })

                        cur = nxt
                        if cur == depot_name:
                            break

            return pd.DataFrame(out)

        df_trajets_ordonnes = ordonner_trajets(df_trajets)

        # --- Format ---
        if not df_trajets_ordonnes.empty:
            df_display, df_for_map = formater_trajets(df_trajets_ordonnes)
        else:
            df_display = pd.DataFrame(columns=["Véhicule", "Départ", "Arrivée", "Distance (KM)"])
            df_for_map = pd.DataFrame(columns=["Cluster", "Véhicule", "Départ", "Arrivée", "Distance (KM)"])

        # --- Sauvegarde Excel ---
        fichier_resultat = os.path.join(dossier_output, "trajets_vrp_ordonnes_distances.xlsx")
        df_display.to_excel(fichier_resultat, index=False)
        log(f"Fichier résultat enregistré: {fichier_resultat}")

        # --- Carte (nom unique) ---
        if depot_name not in coord_dict:
            raise KeyError(f"Coordonnées du dépôt ({depot_name}) introuvables.")

        m = folium.Map(location=coord_dict[depot_name], zoom_start=8, width="100%")

        for nom, coords in coord_dict.items():
            lat, lon = coords
            if pd.isna(lat) or pd.isna(lon):
                continue
            folium.Marker(
                [lat, lon],
                popup=str(nom),
                icon=folium.Icon(color="red" if str(nom).lower() == "depot" else "blue")
            ).add_to(m)

        colors = ["blue", "orange", "green", "purple", "darkred", "darkblue", "cadetblue"]
        cluster_ids = df_for_map["Cluster"].unique() if not df_for_map.empty else []
        cluster_to_color_index = {cid: idx % len(colors) for idx, cid in enumerate(cluster_ids)}

        if not df_for_map.empty:
            for _, row in df_for_map.iterrows():
                cid = row["Cluster"]
                i = row["Départ"]
                j = row["Arrivée"]
                if i in coord_dict and j in coord_dict:
                    folium.PolyLine(
                        [coord_dict[i], coord_dict[j]],
                        color=colors[cluster_to_color_index.get(cid, 0)],
                        weight=2.5,
                        tooltip=f"{row['Véhicule']}: {i} → {j} ({row['Distance (KM)']:.2f} km)"
                    ).add_to(m)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fichier_carte = os.path.join(dossier_output, f"carte_{stamp}.html")
        m.save(fichier_carte)
        log(f"Carte enregistrée: {fichier_carte}")

        # log
        with open(os.path.join(dossier_output, "vrp_log.txt"), "w", encoding="utf-8") as lf:
            lf.write("\n".join(log_lines))

        return df_display, fichier_carte

    except Exception as e:
        os.makedirs("resultats_vrp", exist_ok=True)
        with open(os.path.join("resultats_vrp", "vrp_error.txt"), "w", encoding="utf-8") as ef:
            ef.write("ERREUR RUN VRP\n")
            ef.write(str(e) + "\n\n")
            ef.write(traceback.format_exc())
        raise


