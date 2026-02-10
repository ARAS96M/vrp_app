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


def to_float_series(s: pd.Series) -> pd.Series:
    # support décimales "1,23" dans Excel
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )


# === Format trajets ===
def formater_trajets(df):
    """
    - Si '_part' (ou '-part') dans Départ ou Arrivée -> incrément véhicule
    - Si Départ == depot (sauf première ligne) -> incrément véhicule
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

    df2.insert(0, "Véhicule", [f"V{n}" for n in numeros])

    df_for_map = df2.copy()
    cols_keep = [c for c in ["Véhicule", "Départ", "Arrivée", "Distance (KM)"] if c in df2.columns]
    df_display = df2[cols_keep].copy().reset_index(drop=True)
    return df_display, df_for_map


def run_vrp(fichier_excel):
    try:
        # ====== Paramètres ======
        eps_km = 100
        min_clients = 2
        time_limit = 60

        dossier_output = "resultats_vrp"
        os.makedirs(dossier_output, exist_ok=True)

        # pénalité forte => minimise # véhicules utilisés
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

        clients_df["Latitude"] = to_float_series(clients_df["Latitude"])
        clients_df["Longitude"] = to_float_series(clients_df["Longitude"])
        clients_df["Demande"] = to_float_series(clients_df["Demande"]).fillna(0.0).astype(float)
        clients_df["Disponible"] = pd.to_numeric(clients_df["Disponible"], errors="coerce").fillna(0).astype(int)

        vehicules_df["Capacite"] = to_float_series(vehicules_df["Capacite"]).fillna(0.0).astype(float)
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

        # --- Clients ---
        clients = clients_df[clients_df["Nom"].str.lower() != "depot"].copy()

        # coords manquantes
        missing_coords = clients[clients[["Latitude", "Longitude"]].isna().any(axis=1)][["Nom", "Wilaya", "Latitude", "Longitude"]]
        if not missing_coords.empty:
            log("⚠️ Clients avec coordonnées manquantes (non servables) :")
            log(missing_coords.to_string(index=False))

        # --- Fractionnement ---
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

        # wilayas spécifiques: ZONE
        for wilaya in wilayas_specifiques:
            mask_w = clients["Wilaya"] == wilaya
            if mask_w.any():
                clients.loc[mask_w, "Cluster"] = clients.loc[mask_w, "ZONE"].apply(
                    lambda z: f"{wilaya}_{z if str(z).strip() not in ['', 'nan', 'None'] else 'NA'}"
                )

        # autres: DBSCAN haversine
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

        # sécurité
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

        # === Résolution cluster (SCF) ===
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

            # SCF flow
            f = pulp.LpVariable.dicts("f", (V, noms_avec_depot, noms_avec_depot), lowBound=0, cat="Continuous")

            # objectif: #vehicules puis distance
            model += (
                VEHICLE_PENALTY * pulp.lpSum(y[k] for k in V)
                + pulp.lpSum(d[(i, j)] * x[k][i][j] for k in V for (i, j) in d)
            )

            # self loop interdit
            for k in V:
                for i in noms_avec_depot:
                    model += x[k][i][i] == 0

            # bloquer arcs sans distance
            for k in V:
                for i in noms_avec_depot:
                    for j in noms_avec_depot:
                        if i == j:
                            continue
                        if (i, j) not in d:
                            model += x[k][i][j] == 0
                            model += f[k][i][j] == 0  # flow impossible
                        else:
                            # flow seulement si arc utilisé
                            model += f[k][i][j] <= capacites_vehicules[k] * x[k][i][j]

            # chaque client: 1 entrée + 1 sortie (tous véhicules)
            for j in noms_clients:
                model += pulp.lpSum(x[k][i][j] for k in V for i in noms_avec_depot if i != j) == 1
                model += pulp.lpSum(x[k][j][i] for k in V for i in noms_avec_depot if i != j) == 1

            # contraintes par véhicule
            for k in V:
                cap_k = float(capacites_vehicules.get(k, 0.0))

                # FIX: si véhicule utilisé => départ dépôt=1 et retour dépôt=1
                model += pulp.lpSum(x[k][depot_name][j] for j in noms_clients) == y[k]
                model += pulp.lpSum(x[k][i][depot_name] for i in noms_clients) == y[k]

                # arcs -> y
                for i in noms_avec_depot:
                    for j in noms_avec_depot:
                        if i != j:
                            model += x[k][i][j] <= y[k]

                # flow conservation (depot injecte la demande servie par k)
                served_by_k = {j: pulp.lpSum(x[k][i][j] for i in noms_avec_depot if i != j) for j in noms_clients}

                model += pulp.lpSum(f[k][depot_name][j] for j in noms_clients) == pulp.lpSum(q[j] * served_by_k[j] for j in noms_clients)

                for h in noms_clients:
                    model += (
                        pulp.lpSum(f[k][i][h] for i in noms_avec_depot if i != h)
                        - pulp.lpSum(f[k][h][j] for j in noms_avec_depot if j != h)
                        == q[h] * served_by_k[h]
                    )

                # capacité: somme demandes servies par k <= cap_k
                model += pulp.lpSum(q[j] * served_by_k[j] for j in noms_clients) <= cap_k

                # conservation de degré (si un client est sur la route de k => 1 entrée et 1 sortie pour k)
                for h in noms_clients:
                    model += pulp.lpSum(x[k][i][h] for i in noms_avec_depot if i != h) == served_by_k[h]
                    model += pulp.lpSum(x[k][h][j] for j in noms_avec_depot if j != h) == served_by_k[h]

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
                        if val is not None and val > 0.5:
                            trajets.append({
                                "Cluster": cluster_id,
                                "Véhicule": k,
                                "Départ": i,
                                "Arrivée": j,
                                "Distance (KM)": float(d[(i, j)]) if (i, j) in d else np.nan
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
                    if df_v.empty:
                        continue

                    next_map = {}
                    for _, r in df_v.iterrows():
                        dep = str(r["Départ"])
                        arr = str(r["Arrivée"])
                        # si plusieurs arcs sortants (normalement non), on garde le premier
                        if dep not in next_map:
                            next_map[dep] = (arr, float(r["Distance (KM)"]) if pd.notna(r["Distance (KM)"]) else 0.0)

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

        # --- Format affichage ---
        if not df_trajets_ordonnes.empty:
            df_display, df_for_map = formater_trajets(df_trajets_ordonnes)
        else:
            df_display = pd.DataFrame(columns=["Véhicule", "Départ", "Arrivée", "Distance (KM)"])
            df_for_map = pd.DataFrame(columns=["Cluster", "Véhicule", "Départ", "Arrivée", "Distance (KM)"])

        # --- Sauvegarde Excel ---
        fichier_resultat = os.path.join(dossier_output, "trajets_vrp_ordonnes_distances.xlsx")
        df_display.to_excel(fichier_resultat, index=False)
        log(f"Fichier résultat enregistré: {fichier_resultat}")

        # --- Carte ---
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
                if i in coord_dict and j in coord_dict and pd.notna(row["Distance (KM)"]):
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

