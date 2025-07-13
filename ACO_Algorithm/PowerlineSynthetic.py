import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import time
import json
import os
from google.colab import drive

# --- Helper function to convert NumPy types to Python native types for JSON serialization ---
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj


# --- Haversine Distance Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance # in kilometers

# --- Pheromone and Heuristic Initialization ---
def initialize_pheromones(num_entities, tau0=1.0):
    return np.full((num_entities, num_entities), tau0)

def calculate_heuristic(distances):
    # Use np.where to handle division by zero for unreachable nodes
    # Unreachable nodes (inf distance) will have 0 heuristic
    heuristic = np.where(distances > 0, 1.0 / distances, 0.0)
    # Set heuristic for self-loops (distances == 0) to 0
    np.fill_diagonal(heuristic, 0)
    return heuristic

# --- Probability Calculation (for a single source node) ---
def calculate_probabilities_from_node(pheromones_row, heuristics_row, visited_nodes_indices_set, alpha, beta):
    numerator_row = np.power(pheromones_row, alpha) * np.power(heuristics_row, beta)

    # Mask out visited nodes for the current ant's tour
    mask = np.zeros_like(numerator_row, dtype=bool)
    mask[list(visited_nodes_indices_set)] = True
    numerator_row[mask] = 0 # Prevent visiting already visited nodes (for this ant's tour)

    denominator = np.sum(numerator_row)

    # Handle cases where denominator is zero to avoid division by zero
    probabilities_row = np.divide(numerator_row, denominator,
                                  out=np.zeros_like(numerator_row),
                                  where=denominator != 0)
    return probabilities_row

# --- Pheromone Update (Evaporation & Deposition) ---
def update_pheromones(pheromones, paths, costs, rho, Q_deposit):
    pheromones *= (1 - rho)
    for path, cost in zip(paths, costs):
        if cost == 0 or cost == np.inf: # Skip if cost is zero or infinite
            continue
        pheromone_to_add = Q_deposit / cost
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            pheromones[u, v] += pheromone_to_add
            pheromones[v, u] += pheromone_to_add # Assuming symmetric

# --- Build UAV Distances Matrix using NetworkX APSP (for generated edges) ---
def build_uav_distances_matrix_from_edges_nx(aco_nodes, aco_edges):
    """
    Calculates all-pairs shortest paths using NetworkX for a given set of nodes and edges.
    Handles disconnected components by penalizing paths that cannot be traversed via existing edges.
    This is crucial for UAVs which must follow the powerline network or incur a significant penalty for off-line travel.
    Returns the distance matrix and the NetworkX graph itself for later path extraction.
    """
    num_nodes = len(aco_nodes)
    id_to_idx = {node['id']: i for i, node in enumerate(aco_nodes)}

    G_nx = nx.Graph()
    for node in aco_nodes:
        G_nx.add_node(node['id'], lat=node['lat'], lon=node['lon'])

    for u_id, v_id, dist in aco_edges:
        if u_id in G_nx.nodes and v_id in G_nx.nodes:
            G_nx.add_edge(u_id, v_id, weight=dist)

    print("Calculating all-pairs shortest paths using NetworkX (repeated Dijkstra's)...")
    start_apsp_time = time.time()

    all_pairs_paths_gen = nx.all_pairs_dijkstra_path_length(G_nx, weight='weight')
    all_pairs_paths = {source: dict(paths) for source, paths in all_pairs_paths_gen}

    end_apsp_time = time.time()
    print(f"NetworkX APSP computation time: {end_apsp_time - start_apsp_time:.2f} seconds.")

    uav_distances = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(uav_distances, 0)

    for u_id, paths in all_pairs_paths.items():
        u_idx = id_to_idx.get(u_id)
        if u_idx is None: continue

        for v_id, dist in paths.items():
            v_idx = id_to_idx.get(v_id)
            if v_idx is None: continue
            uav_distances[u_idx, v_idx] = dist
            uav_distances[v_idx, u_idx] = dist

    PENALTY_FACTOR = 5.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue

            if uav_distances[i, j] == np.inf:
                lat1, lon1 = aco_nodes[i]['lat'], aco_nodes[i]['lon']
                lat2, lon2 = aco_nodes[j]['lat'], aco_nodes[j]['lon']
                direct_air_dist = haversine_distance(lat1, lon1, lat2, lon2)

                if direct_air_dist == 0:
                    uav_distances[i, j] = 1e-6 * PENALTY_FACTOR
                else:
                    uav_distances[i, j] = direct_air_dist * PENALTY_FACTOR
                uav_distances[j, i] = uav_distances[i, j]

    return uav_distances, G_nx # Return the graph


# --- LV Network Generation (Included for completeness, but not used when LOAD_FROM_SAVED_DATASET is True) ---
# These parameters are for the network generation function, which won't be called if loading from file.
ACCRA_LAT_RANGE = 0.02
ACCRA_LON_RANGE = 0.02
DEPOT_COORD = {"lat": 5.5900, "lon": -0.2100, "id": -1}
ACCRA_LAT_MIN = DEPOT_COORD['lat'] - ACCRA_LAT_RANGE / 2
ACCRA_LAT_MAX = DEPOT_COORD['lat'] + ACCRA_LAT_RANGE / 2
ACCRA_LON_MIN = DEPOT_COORD['lon'] - ACCRA_LON_RANGE / 2
ACCRA_LON_MAX = DEPOT_COORD['lon'] + ACCRA_LON_RANGE / 2
MIN_LV_SEGMENT_KM = 0.01
MAX_LV_SEGMENT_KM = 0.20

def generate_lv_powerline_network(num_nodes, num_obstacle_zones=10, obstacle_zone_size_factor=0.20, meshing_prob=0.001):
    """
    Generates a synthetic LV powerline network by creating regions of suppressed
    connectivity (simulating buildings/dense areas).
    This function is included for completeness but will not be called in this setup
    as LOAD_FROM_SAVED_DATASET is True.
    """
    nodes = []
    edges = []
    current_node_id = 0
    lat_min_gen, lat_max_gen = ACCRA_LAT_MIN, ACCRA_LAT_MAX
    lon_min_gen, lon_max_gen = ACCRA_LON_MIN, ACCRA_LON_MAX
    depot_node_id = current_node_id
    nodes.append({'id': current_node_id, 'lat': DEPOT_COORD['lat'], 'lon': DEPOT_COORD['lon'], 'is_depot': True})
    current_node_id += 1
    remaining_nodes_to_generate = num_nodes - 1
    if num_nodes < 1: raise ValueError("num_nodes must be at least 1")
    if num_nodes == 1: return nodes, depot_node_id, [], (lat_min_gen, lat_max_gen, lon_min_gen, lon_max_gen), []
    obstacle_zones = []
    depot_zone_buffer = 0.002
    for _ in range(num_obstacle_zones):
        zone_width_lat = ACCRA_LAT_RANGE * obstacle_zone_size_factor * random.uniform(0.8, 1.2)
        zone_height_lon = ACCRA_LON_RANGE * obstacle_zone_size_factor * random.uniform(0.8, 1.2)
        attempts = 0
        max_attempts_zone = 100
        while attempts < max_attempts_zone:
            zone_lat_min = random.uniform(lat_min_gen, lat_max_gen - zone_width_lat)
            zone_lon_min = random.uniform(lon_min_gen, lon_max_gen - zone_height_lon)
            zone_lat_max = zone_lat_min + zone_width_lat
            zone_lon_max = zone_lon_min + zone_height_lon
            if not (DEPOT_COORD['lat'] + depot_zone_buffer > zone_lat_min and \
                    DEPOT_COORD['lat'] - depot_zone_buffer < zone_lat_max and \
                    DEPOT_COORD['lon'] + depot_zone_buffer > zone_lon_min and \
                    DEPOT_COORD['lon'] - depot_zone_buffer < zone_lon_max):
                obstacle_zones.append({
                    'lat_min': zone_lat_min, 'lat_max': zone_lat_max,
                    'lon_min': zone_lon_min, 'lon_max': zone_lon_max
                })
                break
            attempts += 1
    def is_point_in_obstacle_zone(lat, lon):
        for zone in obstacle_zones:
            if zone['lat_min'] <= lat <= zone['lat_max'] and \
               zone['lon_min'] <= lon <= zone['lon_max']:
                return True
        return False
    def does_edge_cross_obstacle_zone(lat1, lon1, lat2, lon2, num_checks=5):
        for i in range(num_checks + 1):
            t = i / num_checks
            check_lat = lat1 + t * (lat2 - lat1)
            check_lon = lon1 + t * (lon2 - lon1)
            if is_point_in_obstacle_zone(check_lat, check_lon):
                return True
        return False
    expandable_nodes = [depot_node_id]
    growth_angles = [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    JITTER_LAT = 0.000001
    JITTER_LON = 0.000001
    global_attempts = 0
    max_global_attempts = num_nodes * 500
    while remaining_nodes_to_generate > 0 and global_attempts < max_global_attempts:
        global_attempts += 1
        if not expandable_nodes:
            if len(nodes) > 1:
                from_node_id = random.choice(nodes)['id']
            else:
                break
        else:
            from_node_id = random.choice(expandable_nodes)
        if from_node_id >= len(nodes): continue
        from_lat, from_lon = nodes[from_node_id]['lat'], nodes[from_node_id]['lon']
        num_branches_attempt = 1
        if random.random() < 0.2:
            num_branches_attempt = 2
        chosen_angles = random.sample(growth_angles, k=min(num_branches_attempt, len(growth_angles)))
        if random.random() < 0.02:
            chosen_angles.append(random.uniform(0, 2*np.pi))
        for angle in chosen_angles:
            if remaining_nodes_to_generate <= 0: break
            segment_length_km = random.uniform(MIN_LV_SEGMENT_KM, MAX_LV_SEGMENT_KM)
            avg_lat_rad = radians((lat_min_gen + lat_max_gen) / 2)
            lat_offset = segment_length_km * np.sin(angle) / 111
            lon_offset = segment_length_km * np.cos(angle) / (111 * np.cos(avg_lat_rad))
            target_lat = from_lat + lat_offset
            target_lon = from_lon + lon_offset
            new_lat = target_lat + random.uniform(-JITTER_LAT, JITTER_LAT)
            new_lon = target_lon + random.uniform(-JITTER_LON, JITTER_LON)
            new_lat = max(lat_min_gen, min(lat_max_gen, new_lat))
            new_lon = max(lon_min_gen, min(lon_max_gen, new_lon))
            if is_point_in_obstacle_zone(new_lat, new_lon) or \
               does_edge_cross_obstacle_zone(from_lat, from_lon, new_lat, new_lon):
                continue
            dist_calculated = haversine_distance(from_lat, from_lon, new_lat, new_lon)
            is_too_close = False
            for existing_node in nodes:
                if haversine_distance(new_lat, new_lon, existing_node['lat'], existing_node['lon']) < MIN_LV_SEGMENT_KM * 0.5:
                    is_too_close = True
                    break
            if is_too_close: continue
            if MIN_LV_SEGMENT_KM <= dist_calculated <= MAX_LV_SEGMENT_KM and current_node_id < num_nodes:
                nodes.append({'id': current_node_id, 'lat': new_lat, 'lon': new_lon})
                edges.append((from_node_id, current_node_id, dist_calculated))
                if current_node_id not in expandable_nodes:
                    expandable_nodes.append(current_node_id)
                current_node_id += 1
                remaining_nodes_to_generate -= 1
            else:
                if from_node_id not in expandable_nodes:
                    expandable_nodes.append(from_node_id)
    temp_graph = nx.Graph()
    for u,v,d in edges:
        temp_graph.add_edge(u,v, weight=d)
    for u_node in nodes:
        for v_node in nodes:
            if u_node['id'] == v_node['id'] or temp_graph.has_edge(u_node['id'], v_node['id']):
                continue
            dist = haversine_distance(u_node['lat'], u_node['lon'], v_node['lat'], v_node['lon'])
            if MIN_LV_SEGMENT_KM * 0.8 <= dist <= MAX_LV_SEGMENT_KM * 1.5:
                if random.random() < meshing_prob and \
                   not does_edge_cross_obstacle_zone(u_node['lat'], u_node['lon'], v_node['lat'], v_node['lon']):
                    edges.append((u_node['id'], v_node['id'], dist))
                    temp_graph.add_edge(u_node['id'], v_node['id'], weight=dist)
    old_ids = sorted([n['id'] for n in nodes])
    id_map = {old_id: new_idx for new_idx, old_id in enumerate(old_ids)}
    reindexed_nodes = []
    for node in nodes:
        new_node = {'id': id_map[node['id']], 'lat': node['lat'], 'lon': node['lon']}
        if 'is_depot' in node: new_node['is_depot'] = node['is_depot']
        reindexed_nodes.append(new_node)
    reindexed_edges = []
    for u_id, v_id, dist in edges:
        if u_id in id_map and v_id in id_map:
            new_u_id = id_map[u_id]
            new_v_id = id_map[v_id]
            reindexed_edges.append((new_u_id, new_v_id, dist))
            reindexed_edges.append((new_v_id, new_u_id, dist)) # For undirected graph
    unique_edges = set()
    final_edges = []
    for u, v, d in reindexed_edges:
        edge_key = tuple(sorted((u, v)))
        if edge_key not in unique_edges:
            unique_edges.add(edge_key)
            final_edges.append((u, v, d))
    new_depot_id = id_map[depot_node_id] if depot_node_id in id_map else 0
    actual_lats = [n['lat'] for n in reindexed_nodes]
    actual_lons = [n['lon'] for n in reindexed_nodes]
    bbox = (min(actual_lats), max(actual_lats), min(actual_lons), max(actual_lons))
    return reindexed_nodes, new_depot_id, final_edges, bbox, obstacle_zones


# --- Inner ACO Function (adjusted for "one tower per sortie" rule) ---
def solve_inner_aco(cluster_id, nodes_in_cluster, num_uavs, uav_range, uav_distances_all, depot_coords, uav_aco_params, all_nodes_global):
    """
    Solves the UAV routing problem for a single cluster.
    Assumes "one tower per sortie": each UAV flight is from the OV location (cluster centroid)
    to a single inspection node and back to the OV location.
    The inner ACO's goal is to assign all nodes in the cluster to the available UAVs
    such that total UAV flight distance is minimized and individual sortie ranges are met.
    """
    iters_uav = uav_aco_params['iters_uav']
    n_ants_uav = uav_aco_params['n_ants_uav']
    rho_uav = uav_aco_params['rho_uav']
    alpha_uav = uav_aco_params['alpha_uav']
    beta_uav = uav_aco_params['beta_uav']
    q_deposit_uav = uav_aco_params['q_deposit_uav']

    local_nodes = [{'id': -1, 'lat': depot_coords['lat'], 'lon': depot_coords['lon'], 'is_virtual_depot': True}] + nodes_in_cluster
    original_id_to_local_idx = {node['id']: i for i, node in enumerate(local_nodes)}
    global_id_to_global_idx = {node['id']: i for i, node in enumerate(all_nodes_global)}

    num_local_nodes = len(local_nodes)
    local_uav_distances = np.zeros((num_local_nodes, num_local_nodes))

    for i in range(num_local_nodes):
        for j in range(num_local_nodes):
            if i == j:
                local_uav_distances[i, j] = 0
                continue

            if local_nodes[i].get('is_virtual_depot') or local_nodes[j].get('is_virtual_depot'):
                local_uav_distances[i, j] = haversine_distance(
                    local_nodes[i]['lat'], local_nodes[i]['lon'],
                    local_nodes[j]['lat'], local_nodes[j]['lon']
                )
            else:
                id_i = local_nodes[i]['id']
                id_j = local_nodes[j]['id']
                try:
                    global_idx_i = global_id_to_global_idx[id_i]
                    global_idx_j = global_id_to_global_idx[id_j]
                    local_uav_distances[i, j] = uav_distances_all[global_idx_i, global_idx_j]
                except KeyError:
                   print(f"Error: Node ID {id_i} or {id_j} not found in global_id_to_global_idx map for cluster {cluster_id}.")
                   local_uav_distances[i, j] = float('inf')


    pheromones_uav = initialize_pheromones(num_local_nodes, tau0=uav_aco_params['tau0_uav'])
    heuristics_uav = calculate_heuristic(local_uav_distances)

    best_cluster_uav_cost = float('inf')
    best_cluster_uav_routes = {}

    for iter_uav in range(iters_uav):
        ant_current_total_uav_cost = 0.0
        ant_current_uav_routes = {q: [] for q in range(num_uavs)}

        nodes_to_inspect_local_idx = [original_id_to_local_idx[node['id']] for node in nodes_in_cluster]
        random.shuffle(nodes_to_inspect_local_idx)

        nodes_assigned_in_this_ant_tour = []
        successful_assignments_in_ant = 0

        for node_local_idx in nodes_to_inspect_local_idx:
            try:
                sortie_cost = local_uav_distances[0, node_local_idx] + local_uav_distances[node_local_idx, 0]
            except IndexError:
                sortie_cost = float('inf')


            if sortie_cost <= uav_range:
                assigned_uav_q = len(nodes_assigned_in_this_ant_tour) % num_uavs

                depot_original_id = f"OV_C{cluster_id}"
                target_original_id = local_nodes[node_local_idx]['id']

                ant_current_uav_routes[assigned_uav_q].extend([depot_original_id, target_original_id, depot_original_id])

                ant_current_total_uav_cost += sortie_cost
                nodes_assigned_in_this_ant_tour.append(node_local_idx)
                successful_assignments_in_ant += 1

        if successful_assignments_in_ant == len(nodes_in_cluster):
            if ant_current_total_uav_cost < best_cluster_uav_cost:
                best_cluster_uav_cost = ant_current_total_uav_cost
                best_cluster_uav_routes = ant_current_uav_routes
        else:
            ant_current_total_uav_cost = float('inf')

        pheromones_uav *= (1 - rho_uav)

        if ant_current_total_uav_cost != float('inf') and ant_current_total_uav_cost > 0:
            pheromone_to_add = q_deposit_uav / ant_current_total_uav_cost
            for uav_q, q_route_original_ids in ant_current_uav_routes.items():
                for i in range(0, len(q_route_original_ids), 3):
                    if i + 1 < len(q_route_original_ids):
                        v_original_id = q_route_original_ids[i+1]
                        v_local = original_id_to_local_idx.get(v_original_id)

                        if v_local is not None:
                            pheromones_uav[0, v_local] += pheromone_to_add
                            pheromones_uav[v_local, 0] += pheromone_to_add

    if best_cluster_uav_cost == float('inf'):
        print(f"Warning: Cluster {cluster_id} (num_nodes: {len(nodes_in_cluster)}) could not find a valid set of UAV sorties to cover all nodes within range {uav_range:.2f}km.")
        if not nodes_in_cluster:
            return 0.0, {}
        return float('inf'), {}

    return best_cluster_uav_cost, best_cluster_uav_routes


# --- Outer ACO Function ---
def solve_outer_aco(depot_cluster_id, all_clusters_data, all_nodes_coords, uav_distances_all, num_uavs, uav_range, ov_aco_params, uav_aco_params):
    iters_ov = ov_aco_params['iters_ov']
    n_ants_ov = ov_aco_params['n_ants_ov']
    rho_ov = ov_aco_params['rho_ov']
    alpha_ov = ov_aco_params['alpha_ov']
    beta_ov = ov_aco_params['beta_ov']
    q_deposit_ov = ov_aco_params['q_deposit_ov']

    cluster_ids_ordered = sorted(all_clusters_data['centroids'].keys())
    cluster_id_to_idx = {c_id: i for i, c_id in enumerate(cluster_ids_ordered)}
    idx_to_cluster_id = {i: c_id for i, c_id in enumerate(cluster_ids_ordered)}

    num_clusters = len(cluster_ids_ordered)

    ov_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i == j: continue
            c_id_i = idx_to_cluster_id[i]
            c_id_j = idx_to_cluster_id[j]
            ov_distances[i, j] = haversine_distance(
                all_clusters_data['centroids'][c_id_i]['lat'], all_clusters_data['centroids'][c_id_i]['lon'],
                all_clusters_data['centroids'][c_id_j]['lat'], all_clusters_data['centroids'][c_id_j]['lon']
            )

    pheromones_ov = initialize_pheromones(num_clusters, tau0=ov_aco_params['tau0_ov'])
    heuristics_ov = calculate_heuristic(ov_distances)

    best_total_cost = float('inf')
    best_ov_path_indices = []
    best_uav_paths_in_clusters = {}

    convergence_data = []

    depot_cluster_idx = cluster_id_to_idx[depot_cluster_id]


    for iter_ov in range(iters_ov):
        ant_ov_paths_indices = []
        ant_total_costs = []

        current_iter_best_ov_cost = float('inf')

        for ant_idx in range(n_ants_ov):
            current_ov_path_indices = [depot_cluster_idx]
            visited_cluster_indices = {depot_cluster_idx}
            current_cluster_idx = depot_cluster_idx

            ant_tour_uav_paths = {}
            ant_tour_total_uav_cost = 0.0

            while len(visited_cluster_indices) < num_clusters:
                probabilities = calculate_probabilities_from_node(
                    pheromones_ov[current_cluster_idx, :],
                    heuristics_ov[current_cluster_idx, :],
                    visited_cluster_indices,
                    alpha_ov, beta_ov
                )

                unvisited_indices = [idx for idx in range(num_clusters) if idx not in visited_cluster_indices]

                if not unvisited_indices:
                    break

                valid_probabilities = np.array([probabilities[unvisited_idx] for unvisited_idx in unvisited_indices])

                if np.sum(valid_probabilities) == 0:
                    next_cluster_idx = random.choice(unvisited_indices)
                else:
                    valid_probabilities_normalized = valid_probabilities / np.sum(valid_probabilities)
                    next_cluster_idx = random.choices(unvisited_indices, weights=valid_probabilities_normalized, k=1)[0]

                current_ov_path_indices.append(next_cluster_idx)
                visited_cluster_indices.add(next_cluster_idx)
                current_cluster_idx = next_cluster_idx

            if len(visited_cluster_indices) < num_clusters:
                current_ant_total_cost = float('inf')
                ant_ov_paths_indices.append(current_ov_path_indices)
                ant_total_costs.append(current_ant_total_cost)
                continue

            current_ov_path_indices.append(depot_cluster_idx)

            total_ov_path_length = 0.0
            for i in range(len(current_ov_path_indices) - 1):
                u_idx, v_idx = current_ov_path_indices[i], current_ov_path_indices[i+1]
                total_ov_path_length += ov_distances[u_idx, v_idx]

            processed_clusters_in_ant_tour = set()
            for cluster_idx in current_ov_path_indices[:-1]:
                cluster_id = idx_to_cluster_id[cluster_idx]

                if cluster_id not in processed_clusters_in_ant_tour:
                    nodes_for_this_cluster = [node for node in all_nodes_coords if 'cluster_id' in node and node['cluster_id'] == cluster_id]

                    depot_coords_inner = all_clusters_data['centroids'][cluster_id]

                    cluster_uav_cost, cluster_uav_routes = solve_inner_aco(
                        cluster_id, nodes_for_this_cluster, num_uavs, uav_range,
                        uav_distances_all, depot_coords_inner, uav_aco_params, all_nodes_coords
                    )

                    if cluster_uav_cost == float('inf'):
                        ant_tour_total_uav_cost = float('inf')
                        break

                    ant_tour_total_uav_cost += cluster_uav_cost
                    ant_tour_uav_paths[cluster_id] = cluster_uav_routes
                    processed_clusters_in_ant_tour.add(cluster_id)

            current_ant_total_cost = float('inf')
            if ant_tour_total_uav_cost != float('inf'):
                current_ant_total_cost = total_ov_path_length + ant_tour_total_uav_cost

            ant_ov_paths_indices.append(current_ov_path_indices)
            ant_total_costs.append(current_ant_total_cost)

            if current_ant_total_cost < current_iter_best_ov_cost:
                current_iter_best_ov_cost = current_ant_total_cost
                if current_ant_total_cost < best_total_cost:
                    best_total_cost = current_ant_total_cost
                    best_ov_path_indices = current_ov_path_indices
                    best_uav_paths_in_clusters = ant_tour_uav_paths


        pheromones_ov *= (1 - rho_ov)

        if ant_total_costs:
            best_iter_ant_idx = np.argmin(ant_total_costs)
            best_iter_total_cost = ant_total_costs[best_iter_ant_idx]

            if best_iter_total_cost != float('inf') and best_iter_total_cost > 0:
                best_iter_ant_path_indices = ant_ov_paths_indices[best_iter_ant_idx]
                pheromone_to_add = q_deposit_ov / best_iter_total_cost
                for j in range(len(best_iter_ant_path_indices) - 1):
                    u_idx, v_idx = best_iter_ant_path_indices[j], best_iter_ant_path_indices[j+1]
                    pheromones_ov[u_idx, v_idx] += pheromone_to_add
                    pheromones_ov[v_idx, u_idx] += pheromone_to_add

        convergence_data.append(best_total_cost)

    best_ov_path_ids = [idx_to_cluster_id[idx] for idx in best_ov_path_indices]

    return best_total_cost, best_ov_path_ids, best_uav_paths_in_clusters, convergence_data


# --- Plotting Functions ---

# --- 4.1. Visualization of Dataset BEFORE Clustering (Raw Network) ---
def plot_raw_powerline_network(all_nodes, generated_edges, bbox_coords, DEPOT_COORD, depot_actual_node_id, obstacle_zones_data):
    plt.figure(figsize=(12, 10))
    G = nx.Graph()
    G.add_nodes_from([node['id'] for node in all_nodes])

    pos = {node['id']: (node['lon'], node['lat']) for node in all_nodes}

    raw_edges_nx_format = [(u, v) for u, v, _ in generated_edges]
    G.add_edges_from(raw_edges_nx_format)

    for zone in obstacle_zones_data:
        rect = plt.Rectangle((zone['lon_min'], zone['lat_min']),
                             zone['lon_max'] - zone['lon_min'],
                             zone['lat_max'] - zone['lat_min'],
                             facecolor='lightcoral', edgecolor='red', alpha=0.3, zorder=0)
        plt.gca().add_patch(rect)

    nx.draw_networkx_nodes(G, pos, nodelist=[n_id for n_id in G.nodes() if isinstance(n_id, int)], node_size=20, node_color='darkblue', edgecolors='none', alpha=0.8)

    nx.draw_networkx_edges(G, pos, edgelist=raw_edges_nx_format, width=0.8, edge_color='darkgray', alpha=0.7)

    depot_node_raw = next((n for n in all_nodes if n['id'] == depot_actual_node_id), None)
    if depot_node_raw and depot_node_raw['id'] in pos:
        nx.draw_networkx_nodes(G, pos={depot_node_raw['id']: (depot_node_raw['lon'], depot_node_raw['lat'])},
                               nodelist=[depot_node_raw['id']],
                               node_size=150, node_color='darkred', edgecolors='black', linewidths=1.0, label='Main Depot')
        plt.text(depot_node_raw['lon'], depot_node_raw['lat'] + 0.00001, 'Depot', fontsize=10, ha='center', va='bottom', color='darkred', fontweight='bold')

    plt.title("Synthetic LV Powerline Network: Raw Nodes and Segments (Urban Obstacles)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.xlim(bbox_coords[2], bbox_coords[3])
    plt.ylim(bbox_coords[0], bbox_coords[1])
    plt.legend(loc='lower left', handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=5, label='Inspection Nodes'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', markersize=10, label='Main Depot'),
        plt.Line2D([0], [0], color='darkgray', linewidth=1, label='LV Powerline Segments'),
        plt.Rectangle((0, 0), 1, 1, fc="lightcoral", alpha=0.3, label="Obstacle Zones (No Lines)")
    ], framealpha=0.9)
    plt.tight_layout()
    plt.savefig("synthetic_lv_network_raw.png")
    plt.show()

# --- 4.2. Visualization of Dataset AFTER Clustering (Clustered Network) ---
def plot_clustered_powerline_network_nx(all_nodes, cluster_centroids, depot_cluster_id, cluster_colors_map,
                                        plot_nodes_coords, node_labels_map, centroid_plot_ids_map, generated_edges, bbox_coords, depot_actual_node_id, obstacle_zones_data, num_clusters_for_title):
    plt.figure(figsize=(12, 10))
    G = nx.Graph()

    node_ids_only = [node['id'] for node in all_nodes]
    G.add_nodes_from(node_ids_only)

    centroid_plot_id_list = list(centroid_plot_ids_map.values())
    G.add_nodes_from(centroid_plot_id_list)

    lv_edges_nx_format = [(u, v) for u, v, dist in generated_edges]
    G.add_edges_from(lv_edges_nx_format)

    for zone in obstacle_zones_data:
        rect = plt.Rectangle((zone['lon_min'], zone['lat_min']),
                             zone['lon_max'] - zone['lon_min'],
                             zone['lat_max'] - zone['lat_min'],
                             facecolor='lightcoral', edgecolor='red', alpha=0.3, zorder=0)
        plt.gca().add_patch(rect)

    node_colors_list = []
    node_sizes_list = []

    valid_plot_node_ids = set(plot_nodes_coords.keys())

    for n_id in G.nodes():
        if isinstance(n_id, int):
            if n_id not in valid_plot_node_ids:
                node_colors_list.append('lightgray')
                node_sizes_list.append(0)
                continue

            node_data = next((node for node in all_nodes if node['id'] == n_id), None)
            if node_data and 'cluster_id' in node_data:
                node_colors_list.append(mcolors.rgb2hex(cluster_colors_map(node_data['cluster_id'])))
            else:
                node_colors_list.append('lightgray')
            node_sizes_list.append(25)

            if n_id == depot_actual_node_id:
                node_colors_list[-1] = 'darkred'
                node_sizes_list[-1] = 70

        else:
            if n_id not in valid_plot_node_ids:
                node_colors_list.append('lightgray')
                node_sizes_list.append(0)
                continue

            if n_id == f"DepotC{depot_cluster_id}":
                node_colors_list.append('darkred')
                node_sizes_list.append(200)
            else:
                cluster_id_from_str = int(n_id.replace('C', '').replace('Depot', ''))
                node_colors_list.append(mcolors.rgb2hex(cluster_colors_map(cluster_id_from_str)))
                node_sizes_list.append(120)

    nx.draw_networkx_nodes(G, plot_nodes_coords, node_size=node_sizes_list, node_color=node_colors_list, edgecolors='black', linewidths=0.4)

    centroid_label_pos = {p_id: (x, y + 0.000005) for p_id, (x, y) in plot_nodes_coords.items() if not isinstance(p_id, int) and p_id in G.nodes()}
    nx.draw_networkx_labels(G, plot_nodes_coords, labels={p_id: node_labels_map[p_id] for p_id in centroid_plot_ids_map.values() if p_id in G.nodes()},
                             font_size=7, font_weight='bold', font_color='darkblue')

    nx.draw_networkx_edges(G, plot_nodes_coords, edgelist=lv_edges_nx_format, width=1, edge_color='gray', alpha=0.6, label='Generated LV Powerline Segments')

    plt.title(f"Synthetic LV Powerline Network: Clustered Nodes (K={num_clusters_for_title})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray', markersize=5, label='Inspection Nodes (by cluster color)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=7, label='Cluster Centroids'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', markersize=9, label='Depot & Depot Cluster Centroid'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, label='LV Powerline Segments'),
        plt.Rectangle((0, 0), 1, 1, fc="lightcoral", alpha=0.3, label="Obstacle Zones (No Lines)")
    ]
    for i in range(num_clusters_for_title): # Use num_clusters_for_title to match the clustering
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.rgb2hex(cluster_colors_map(i)),
                                           markersize=7, label=f'Cluster {i}'))

    plt.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize='small', framealpha=0.9)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.xlim(bbox_coords[2], bbox_coords[3])
    plt.ylim(bbox_coords[0], bbox_coords[1])
    plt.tight_layout()
    plt.savefig(f"synthetic_lv_network_clustered_K{num_clusters_for_title}.png")
    plt.show()

# --- 4.3. Visualization of Best Paths (OV + UAVs) ---
def plot_best_paths_nx(all_nodes, best_ov_path_ids, best_uav_paths_in_clusters, cluster_centroids, depot_cluster_id, plot_nodes_coords, Q_UAVS, bbox_coords, depot_actual_node_id, obstacle_zones_data, generated_lv_edges, num_clusters_for_title, num_uavs_for_title):
    plt.figure(figsize=(14, 12))
    ax = plt.gca()

    for zone in obstacle_zones_data:
        rect = plt.Rectangle((zone['lon_min'], zone['lat_min']),
                             zone['lon_max'] - zone['lon_min'],
                             zone['lat_max'] - zone['lat_min'],
                             facecolor='lightcoral', edgecolor='red', alpha=0.2, zorder=0)
        ax.add_patch(rect)

    for u, v, _ in generated_lv_edges:
        node_u = next((n for n in all_nodes if n['id'] == u), None)
        node_v = next((n for n in all_nodes if n['id'] == v), None)
        if node_u and node_v:
            ax.plot([node_u['lon'], node_v['lon']], [node_u['lat'], node_v['lat']],
                    color='lightgray', linewidth=0.5, alpha=0.8, zorder=1)

    for node_id, (lon, lat) in plot_nodes_coords.items():
        if isinstance(node_id, int):
            ax.plot(lon, lat, 'o', markersize=2, color='lightgray', alpha=0.6, zorder=2)

    all_visited_nodes_by_uavs = set()
    for cluster_id, uav_routes in best_uav_paths_in_clusters.items():
        for uav_q, route_nodes_ids in uav_routes.items():
            for i in range(0, len(route_nodes_ids), 3):
                if i + 1 < len(route_nodes_ids):
                    target_node_id = route_nodes_ids[i+1]
                    all_visited_nodes_by_uavs.add(target_node_id)

    for node_id in all_visited_nodes_by_uavs:
        node_data = next((n for n in all_nodes if n['id'] == node_id), None)
        if node_data:
            ax.plot(node_data['lon'], node_data['lat'], 'o', markersize=5, color='darkgreen', markeredgecolor='black', linewidth=0.5, zorder=4, label='_nolegend_')

    for c_id, centroid in cluster_centroids.items():
        if c_id == depot_cluster_id:
            ax.plot(centroid['lon'], centroid['lat'], 'X', markersize=18, color='darkred', markeredgecolor='black', linewidth=1.5, zorder=8, label='Main Depot (OV Base)')
            ax.text(centroid['lon'], centroid['lat'] + 0.00003, f"Depot C{c_id}", fontsize=10, ha='center', va='bottom', color='darkred', fontweight='bold', zorder=9)
        else:
            ax.plot(centroid['lon'], centroid['lat'], 's', markersize=12, color='skyblue', markeredgecolor='black', linewidth=0.7, alpha=0.8, zorder=7, label='_nolegend_')
            ax.text(centroid['lon'], centroid['lat'] + 0.00003, f"C{c_id}", fontsize=9, ha='center', va='bottom', color='darkblue', fontweight='bold', zorder=8)

    ax.plot([], [], 's', markersize=12, color='skyblue', markeredgecolor='black', linewidth=0.7, alpha=0.8, label='Cluster Centroids (OV Bases)')

    ov_path_coords = []
    for cluster_id in best_ov_path_ids:
        centroid_loc = cluster_centroids[cluster_id]
        ov_path_coords.append((centroid_loc['lon'], centroid_loc['lat']))

    for i in range(len(ov_path_coords) - 1):
        x1, y1 = ov_path_coords[i]
        x2, y2 = ov_path_coords[i+1]
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                 head_width=0.00005, head_length=0.00008, length_includes_head=True,
                 color='blue', linewidth=2.5, alpha=0.9, zorder=6)
    ax.plot([p[0] for p in ov_path_coords], [p[1] for p in ov_path_coords],
             'b-', linewidth=2.5, alpha=0.7, label='Operating Vehicle (OV) Path', zorder=5)

    for i in range(len(best_ov_path_ids) - 1):
        cluster_id = best_ov_path_ids[i]
        centroid_loc = cluster_centroids[cluster_id]
        ax.text(centroid_loc['lon'], centroid_loc['lat'], str(i+1),
                 color='white', fontsize=10, fontweight='bold', ha='center', va='center',
                 bbox=dict(facecolor='blue', edgecolor='darkblue', boxstyle='circle,pad=0.15', alpha=0.9),
                 zorder=10)

    uav_path_colors = plt.get_cmap('Paired', Q_UAVS) # Q_UAVS is the current num UAVs being plotted
    uav_labels_added = set()

    for cluster_id, uav_routes in best_uav_paths_in_clusters.items():
        cluster_centroid_loc = cluster_centroids[cluster_id]

        for uav_q, route_nodes_ids in uav_routes.items():
            if not route_nodes_ids:
                continue

            for i in range(0, len(route_nodes_ids), 3):
                if i + 2 < len(route_nodes_ids):
                    target_node_id = route_nodes_ids[i+1]
                    node_data = next((n for n in all_nodes if n['id'] == target_node_id), None)

                    if node_data:
                        uav_sortie_coords = [
                            (cluster_centroid_loc['lon'], cluster_centroid_loc['lat']),
                            (node_data['lon'], node_data['lat']),
                            (cluster_centroid_loc['lon'], cluster_centroid_loc['lat'])
                        ]

                        if Q_UAVS > 1: # Use Q_UAVS from function argument for colormap scaling
                            normalized_uav_q = uav_q / (Q_UAVS - 1)
                        else:
                            normalized_uav_q = 0.5

                        color_for_uav = uav_path_colors(normalized_uav_q)
                        label = f'UAV {uav_q+1} Sorties' if uav_q not in uav_labels_added else "_nolegend_"

                        ax.plot([p[0] for p in uav_sortie_coords], [p[1] for p in uav_sortie_coords],
                                  linestyle='--', linewidth=0.8, color=color_for_uav, alpha=0.6, label=label, zorder=3)
                        ax.arrow(uav_sortie_coords[0][0], uav_sortie_coords[0][1],
                                 uav_sortie_coords[1][0] - uav_sortie_coords[0][0],
                                 uav_sortie_coords[1][1] - uav_sortie_coords[0][1],
                                 head_width=0.00002, head_length=0.00003, length_includes_head=True,
                                 color=color_for_uav, linewidth=0.8, linestyle='--', alpha=0.6, zorder=3)
                        uav_labels_added.add(uav_q)


    ax.legend(loc='upper left', fontsize='small', framealpha=0.9, bbox_to_anchor=(1.02, 1))
    plt.title(f"Optimized OV and UAV Paths (K={num_clusters_for_title}, Q={num_uavs_for_title})", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.xlim(bbox_coords[2], bbox_coords[3])
    plt.ylim(bbox_coords[0], bbox_coords[1])
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(f"optimized_paths_ov_uav_synthetic_lv_enhanced_K{num_clusters_for_title}_Q{num_uavs_for_title}.png", bbox_inches='tight')
    plt.show()

# --- Evaluation Functions ---
def calculate_node_coverage(all_nodes, depot_actual_node_id, best_uav_paths_in_clusters):
    """
    Calculates the percentage of inspection nodes covered by UAV sorties.
    Excludes the main depot node from the total count of 'inspectable' nodes.
    """
    total_inspectable_nodes = [node for node in all_nodes if node['id'] != depot_actual_node_id]
    total_inspectable_node_ids = {node['id'] for node in total_inspectable_nodes}

    covered_node_ids = set()
    for cluster_id, uav_routes in best_uav_paths_in_clusters.items():
        for uav_q, route_nodes_ids in uav_routes.items():
            for i in range(0, len(route_nodes_ids), 3): # Iterate over (OV->Node->OV) segments
                if i + 1 < len(route_nodes_ids):
                    target_node_id = route_nodes_ids[i+1]
                    if isinstance(target_node_id, int):
                        covered_node_ids.add(target_node_id)

    effectively_covered_node_ids = covered_node_ids.intersection(total_inspectable_node_ids)

    if not total_inspectable_node_ids:
        return 0.0

    coverage = (len(effectively_covered_node_ids) / len(total_inspectable_node_ids)) * 100
    return coverage

def calculate_edge_coverage(all_nodes, generated_lv_edges, best_uav_paths_in_clusters, cluster_centroids, uav_graph_nx):
    """
    Calculates the percentage of unique powerline edges traversed by UAVs.
    Assumes UAVs follow the shortest path along the network from the closest network node
    to their OV base, to the inspection node.
    """
    if not generated_lv_edges:
        return 0.0

    all_original_edges_canonical = set()
    for u, v, _ in generated_lv_edges:
        all_original_edges_canonical.add(tuple(sorted((u, v))))

    covered_edges_canonical = set()

    node_id_to_coords = {node['id']: (node['lat'], node['lon']) for node in all_nodes}

    for cluster_id, uav_routes in best_uav_paths_in_clusters.items():
        ov_base_lat, ov_base_lon = cluster_centroids[cluster_id]['lat'], cluster_centroids[cluster_id]['lon']

        closest_node_to_ov_id = None
        min_dist_to_ov = float('inf')
        for node_id_in_graph in uav_graph_nx.nodes():
            n_lat, n_lon = node_id_to_coords.get(node_id_in_graph)
            if n_lat is None or n_lon is None:
                continue
            dist = haversine_distance(ov_base_lat, ov_base_lon, n_lat, n_lon)
            if dist < min_dist_to_ov:
                min_dist_to_ov = dist
                closest_node_to_ov_id = node_id_in_graph

        if closest_node_to_ov_id is None:
            continue

        for uav_q, route_nodes_ids in uav_routes.items():
            for i in range(0, len(route_nodes_ids), 3):
                if i + 1 < len(route_nodes_ids):
                    target_node_id = route_nodes_ids[i+1]

                    if not isinstance(target_node_id, int) or target_node_id not in uav_graph_nx.nodes():
                        continue

                    try:
                        path_nodes = nx.shortest_path(uav_graph_nx, source=closest_node_to_ov_id, target=target_node_id, weight='weight')
                        for k in range(len(path_nodes) - 1):
                            u, v = path_nodes[k], path_nodes[k+1]
                            covered_edges_canonical.add(tuple(sorted((u, v))))
                    except nx.NetworkXNoPath:
                        pass

    if not all_original_edges_canonical:
        return 0.0

    coverage = (len(covered_edges_canonical) / len(all_original_edges_canonical)) * 100
    return coverage


# --- Optimal Cluster Analysis Functions (Elbow and Silhouette) ---

def plot_elbow_method(data_coords, max_clusters=15):
    """
    Plots the Elbow Method for K-Means to help determine optimal K.
    """
    sse = []
    if len(data_coords) < 1:
        print("Not enough data points for Elbow method.")
        return

    max_k_effective = min(max_clusters, len(data_coords))
    k_range = range(1, max_k_effective + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_coords)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kmeans_elbow_method.png")
    plt.show()
    print("Elbow Method plot saved to kmeans_elbow_method.png")

    if len(sse) > 2:
        diffs = np.diff(sse)
        diff_of_diffs = np.diff(diffs)
        optimal_k_elbow = np.argmax(diff_of_diffs) + 2
        print(f"Suggested K from Elbow Method (heuristic): {optimal_k_elbow}")
    return sse

def plot_silhouette_score(data_coords, max_clusters=15):
    """
    Plots the Silhouette Score for K-Means to help determine optimal K.
    """
    silhouette_scores = []

    if len(data_coords) < 2:
        print("Not enough data points for Silhouette Score (requires at least 2).")
        return

    max_k_effective = min(max_clusters, len(data_coords) - 1)
    if max_k_effective < 2:
        print("Not enough data points to compute Silhouette Score for multiple clusters.")
        return

    k_range = range(2, max_k_effective + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_coords)
        if len(set(labels)) < 2:
            silhouette_scores.append(-1)
            print(f"Skipping Silhouette for K={k} as only 1 cluster formed.")
            continue
        try:
            score = silhouette_score(data_coords, labels)
            silhouette_scores.append(score)
        except ValueError as e:
            print(f"Could not calculate Silhouette Score for K={k}: {e}")
            silhouette_scores.append(-1)

    if not silhouette_scores or all(s == -1 for s in silhouette_scores):
        print("No valid Silhouette Scores could be computed.")
        return

    valid_k_range = [k_range[i] for i, score in enumerate(silhouette_scores) if score != -1]
    valid_scores = [score for score in silhouette_scores if score != -1]

    if not valid_scores:
        print("No valid Silhouette Scores to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(valid_k_range, valid_scores, marker='o')
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(list(valid_k_range))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kmeans_silhouette_score.png")
    plt.show()
    print("Silhouette Score plot saved to kmeans_silhouette_score.png")

    optimal_k_silhouette = valid_k_range[np.argmax(valid_scores)]
    print(f"Suggested K from Silhouette Score: {optimal_k_silhouette}")
    return silhouette_scores

def plot_convergence_analysis_varying_q(convergence_data_by_q, ov_aco_iterations):
    """
    Plots the convergence curves for different Q_UAVS values.
    """
    plt.figure(figsize=(12, 8))
    for q_val, history in convergence_data_by_q.items():
        if history:
            plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', markersize=4, label=f'Q={q_val}')

    plt.title('Convergence of Hierarchical ACO Algorithm for Different Number of UAVs (Q)')
    plt.xlabel('Outer ACO Iteration')
    plt.ylabel('Best Total Path Cost (km)')
    plt.grid(True)
    plt.xticks(np.arange(0, ov_aco_iterations + 1, max(1, ov_aco_iterations // 10)))
    plt.legend(title='Number of UAVs (Q)')
    plt.tight_layout()
    plt.savefig("aco_convergence_plot_all_q.png")
    plt.show()
    print("Combined convergence plot saved to aco_convergence_plot_all_q.png")

def plot_runtime_comparison_varying_q(runtimes_data):
    """
    Plots a bar chart comparing the runtime for different Q_UAVS values.
    """
    q_values = sorted(runtimes_data.keys())
    runtimes = [runtimes_data[q] for q in q_values]

    plt.figure(figsize=(10, 6))
    plt.bar([str(q) for q in q_values], runtimes, color='lightgreen', edgecolor='black')
    plt.xlabel('Number of UAVs (Q)')
    plt.ylabel('Runtime (seconds)')
    plt.title('ACO Runtime Comparison for Different Number of UAVs (Q)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("aco_runtime_comparison_q_values.png")
    plt.show()
    print("Runtime comparison plot for Q values saved to aco_runtime_comparison_q_values.png")


# --- Main Execution Flow ---

# Configuration: Always load from JSON for network data. ACO parameters are defined here.
LOAD_FROM_SAVED_DATASET = True

# Ensure this path matches where your JSON file is located in Google Drive.
SAVED_FILE_PATH = '/content/drive/MyDrive/PowerLine/getn/ACO_LV_Network_Simulation_dataset.json'

# --- ACO and Simulation Parameters (DEFINED HERE) ---
N_NODES_CONFIG = 100 # Original number of nodes when dataset was generated (informational)
FIXED_M_CLUSTERS = 12 # <--- FIXED NUMBER OF CLUSTERS
R_UAV_KM = 5.0 # UAV flight range in km

# Define fixed depot location (example central Accra)
DEPOT_COORD = {"lat": 5.5900, "lon": -0.2100, "id": -1}

OV_ACO_PARAMS = {
    'iters_ov': 75,
    'n_ants_ov': 7,
    'rho_ov': 0.1,
    'alpha_ov': 1.0,
    'beta_ov': 3.0,
    'q_deposit_ov': 250.0,
    'tau0_ov': 1.0
}

UAV_ACO_PARAMS = {
    'iters_uav': 50,
    'n_ants_uav': 6,
    'rho_uav': 0.1,
    'alpha_uav': 1.0,
    'beta_uav': 3.0,
    'q_deposit_uav': 200.0,
    'tau0_uav': 1.0
}
# --- END ACO and Simulation Parameters ---

# --- UAV counts for Analysis ---
Q_UAVS_FOR_ANALYSIS = [2, 4, 6, 8, 10, 12]
VISUALIZATION_Q_UAVS = 2 # Generate full visualizations for this Q value

# Global variables for loaded network data
all_nodes_original = [] # Will hold the original loaded nodes
depot_actual_node_id_original = -1
generated_lv_edges_original = []
bbox_coords_original = (0, 0, 0, 0)
obstacle_zones_data_original = []
uav_distances_matrix_original = None
uav_graph_nx_original = None

# Variables that hold the results for the VISUALIZATION_Q_UAVS setting
# These will be populated inside the loop and used for specific plots/textual output
visual_all_nodes = []
visual_cluster_centroids = {}
visual_depot_cluster_id = -1
visual_best_overall_cost = float('inf')
visual_best_ov_path_ids = []
visual_best_uav_paths_in_clusters = {}
visual_convergence_history = []


# Store results for each Q during the analysis loop
convergence_histories_all_q = {}
runtimes_all_q = {}


print("\n--- Mounting Google Drive ---")
try:
    if not os.path.exists('/content/drive/My Drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted.")
    else:
        print("Google Drive already mounted.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}. Please manually mount or ensure connectivity.")
    exit()


print("\n--- Loading Network Data from Google Drive ---")
if not os.path.exists(SAVED_FILE_PATH):
    print(f"Error: Saved file not found at {SAVED_FILE_PATH}")
    print("Please check the path or generate a new dataset first (by setting LOAD_FROM_SAVED_DATASET=False in a previous run).")
    exit()

try:
    with open(SAVED_FILE_PATH, 'r') as f:
        loaded_data = json.load(f)

    globals()['all_nodes_original'] = loaded_data['nodes']
    globals()['depot_actual_node_id_original'] = loaded_data['depot_actual_node_id']
    globals()['generated_lv_edges_original'] = loaded_data['generated_lv_edges']
    globals()['bbox_coords_original'] = loaded_data['bbox_coords']
    globals()['obstacle_zones_data_original'] = loaded_data['obstacle_zones_data']

    if 'simulation_parameters' in loaded_data and 'N_NODES' in loaded_data['simulation_parameters']:
        globals()['N_NODES_CONFIG'] = loaded_data['simulation_parameters']['N_NODES']


    print("Network data loaded successfully. ACO parameters are defined in the code.")
    print(f"Loaded network contains {len(all_nodes_original)} nodes.")

except Exception as e:
    print(f"Error loading network data from file: {e}")
    print("Please check the file content or path.")
    exit()

# Build UAV distances matrix and NetworkX graph once for the loaded network
print("\nBuilding UAV distances matrix and NetworkX graph for the loaded network...")
uav_distances_matrix_original, uav_graph_nx_original = build_uav_distances_matrix_from_edges_nx(all_nodes_original, generated_lv_edges_original)
print("UAV distances matrix and NetworkX graph built.")

# --- Perform Optimal K-Means Clustering Analysis on the loaded data (informational) ---
print("\n--- Performing Optimal K-Means Cluster Analysis (Elbow Method & Silhouette Score) ---")
coords_for_kmeans_analysis = np.array([[node['lat'], node['lon']] for node in all_nodes_original])

if len(coords_for_kmeans_analysis) == 0:
    print("No data points available for cluster analysis. Skipping.")
else:
    max_clusters_to_test = min(len(coords_for_kmeans_analysis) - 1, 15)
    if max_clusters_to_test < 2:
        print(f"Warning: Only {len(coords_for_kmeans_analysis)} nodes. Cannot perform comprehensive cluster analysis.")
    else:
        print(f"Testing K from 1 to {max_clusters_to_test} for Elbow Method, and 2 to {max_clusters_to_test} for Silhouette Score.")
        print("\nRunning Elbow Method...")
        plot_elbow_method(coords_for_kmeans_analysis, max_clusters=max_clusters_to_test)
        print("\nRunning Silhouette Score analysis...")
        plot_silhouette_score(coords_for_kmeans_analysis, max_clusters=max_clusters_to_test)
        print("\n--- K-Means Cluster Analysis Complete (for reference) ---")
        print(f"\nThis analysis is for reference. The ACO simulation will proceed with a FIXED_M_CLUSTERS = {FIXED_M_CLUSTERS}.")


# --- Perform K-Means Clustering ONCE for FIXED_M_CLUSTERS ---
print(f"\n--- Performing K-means clustering ONCE with FIXED_M_CLUSTERS = {FIXED_M_CLUSTERS} ---")
current_all_nodes_clustered = [node.copy() for node in all_nodes_original] # Deep copy for modification
coords_fixed_k = np.array([[node['lat'], node['lon']] for node in current_all_nodes_clustered])

if len(current_all_nodes_clustered) < FIXED_M_CLUSTERS:
    print(f"Error: Number of nodes ({len(current_all_nodes_clustered)}) is less than FIXED_M_CLUSTERS ({FIXED_M_CLUSTERS}). Cannot proceed with clustering. Exiting.")
    exit()

kmeans_fixed_k = KMeans(n_clusters=FIXED_M_CLUSTERS, random_state=42, n_init=10)
labels_fixed_k = kmeans_fixed_k.fit_predict(coords_fixed_k)

fixed_clusters_data = {i: [] for i in range(FIXED_M_CLUSTERS)}
for i, label in enumerate(labels_fixed_k):
    current_all_nodes_clustered[i]['cluster_id'] = label
    fixed_clusters_data[label].append(current_all_nodes_clustered[i])

fixed_cluster_centroids = {i: {'lat': centroid[0], 'lon': centroid[1], 'id': i} for i, centroid in enumerate(kmeans_fixed_k.cluster_centers_)}

fixed_depot_node_obj = next(node for node in current_all_nodes_clustered if node['id'] == depot_actual_node_id_original)
fixed_depot_cluster_id = fixed_depot_node_obj['cluster_id']
fixed_cluster_centroids[fixed_depot_cluster_id]['lat'] = DEPOT_COORD['lat']
fixed_cluster_centroids[fixed_depot_cluster_id]['lon'] = DEPOT_COORD['lon']
fixed_cluster_centroids[fixed_depot_cluster_id]['id'] = f"D{fixed_depot_cluster_id}" # Assign a string ID

fixed_all_clusters_for_aco = {
    'centroids': fixed_cluster_centroids,
    'node_sets': fixed_clusters_data
}
print(f"Clustering for FIXED_M_CLUSTERS = {FIXED_M_CLUSTERS} complete.")


# --- Main Loop for ACO Convergence and Runtime Analysis (Varying Q) ---
print("\n--- Starting ACO Convergence and Runtime Analysis for various Q_UAVS ---")

for current_q_uavs in Q_UAVS_FOR_ANALYSIS:
    print(f"\nProcessing Q_UAVS = {current_q_uavs} (with K={FIXED_M_CLUSTERS})...")

    # ACO will use the fixed clustering results
    start_time_aco = time.time()
    iter_best_cost, iter_best_ov_path_ids, iter_best_uav_paths_in_clusters, iter_convergence_history = solve_outer_aco(
        fixed_depot_cluster_id,
        fixed_all_clusters_for_aco,
        current_all_nodes_clustered, # Use the already clustered nodes
        uav_distances_matrix_original,
        current_q_uavs, # Varying Q
        R_UAV_KM,
        OV_ACO_PARAMS,
        UAV_ACO_PARAMS
    )
    end_time_aco = time.time()
    iter_runtime = end_time_aco - start_time_aco

    print(f"  ACO for Q={current_q_uavs} finished. Runtime: {iter_runtime:.2f}s, Best Cost: {iter_best_cost:.2f}km")

    convergence_histories_all_q[current_q_uavs] = iter_convergence_history
    runtimes_all_q[current_q_uavs] = iter_runtime

    # Store full results if this is the Q for visualization
    if current_q_uavs == VISUALIZATION_Q_UAVS:
        globals()['visual_all_nodes'] = current_all_nodes_clustered # Store the nodes with cluster_id for plotting
        globals()['visual_cluster_centroids'] = fixed_cluster_centroids
        globals()['visual_depot_cluster_id'] = fixed_depot_cluster_id
        globals()['visual_best_overall_cost'] = iter_best_cost
        globals()['visual_best_ov_path_ids'] = iter_best_ov_path_ids
        globals()['visual_best_uav_paths_in_clusters'] = iter_best_uav_paths_in_clusters
        globals()['visual_convergence_history'] = iter_convergence_history


print("\n--- All ACO runs for Convergence and Runtime Analysis Complete ---")


# --- Final Visualizations and Textual Output (Only for VISUALIZATION_Q_UAVS) ---
if visual_best_overall_cost != float('inf'): # Check if visual_results were actually populated
    print(f"\n--- Generating detailed visualizations and textual output for K={FIXED_M_CLUSTERS}, Q_UAVS = {VISUALIZATION_Q_UAVS} ---")

    # Re-derive plot_nodes_coords and node_labels_map for the visual K/Q
    plot_nodes_coords = {node['id']: (node['lon'], node['lat']) for node in visual_all_nodes}
    node_labels_map = {node['id']: str(node['id']) for node in visual_all_nodes}

    centroid_plot_ids_map = {}
    for c_id, centroid in visual_cluster_centroids.items():
        plot_id = f"C{c_id}"
        if c_id == visual_depot_cluster_id:
            plot_id = f"DepotC{c_id}"
        plot_nodes_coords[plot_id] = (centroid['lon'], centroid['lat'])
        node_labels_map[plot_id] = str(c_id)
        centroid_plot_ids_map[c_id] = plot_id

    cluster_colors_map = plt.get_cmap('tab20', FIXED_M_CLUSTERS) # Use fixed K for cluster colors

    # 1. Plot raw network (always the same, using original loaded data)
    plot_raw_powerline_network(all_nodes_original, generated_lv_edges_original, bbox_coords_original, DEPOT_COORD, depot_actual_node_id_original, obstacle_zones_data_original)

    # 2. Plot clustered network for FIXED_M_CLUSTERS
    plot_clustered_powerline_network_nx(visual_all_nodes, visual_cluster_centroids, visual_depot_cluster_id, cluster_colors_map,
                                        plot_nodes_coords, node_labels_map, centroid_plot_ids_map, generated_lv_edges_original, bbox_coords_original, depot_actual_node_id_original, obstacle_zones_data_original, num_clusters_for_title=FIXED_M_CLUSTERS)

    # 3. Plot best paths for VISUALIZATION_Q_UAVS (with FIXED_M_CLUSTERS)
    plot_best_paths_nx(visual_all_nodes, visual_best_ov_path_ids, visual_best_uav_paths_in_clusters, visual_cluster_centroids, visual_depot_cluster_id, plot_nodes_coords, VISUALIZATION_Q_UAVS, bbox_coords_original, depot_actual_node_id_original, obstacle_zones_data_original, generated_lv_edges_original, num_clusters_for_title=FIXED_M_CLUSTERS, num_uavs_for_title=VISUALIZATION_Q_UAVS)

    # --- Evaluation Metrics for VISUALIZATION_Q_UAVS ---
    print(f"\n--- Running Evaluation Metrics for K={FIXED_M_CLUSTERS}, Q_UAVS = {VISUALIZATION_Q_UAVS} ---")
    node_coverage = calculate_node_coverage(visual_all_nodes, depot_actual_node_id_original, visual_best_uav_paths_in_clusters)
    print(f"Node Coverage: {node_coverage:.2f}% of inspection nodes visited.")
    edge_coverage = calculate_edge_coverage(visual_all_nodes, generated_lv_edges_original, visual_best_uav_paths_in_clusters, visual_cluster_centroids, uav_graph_nx_original)
    print(f"Edge Coverage: {edge_coverage:.2f}% of LV powerline segments traversed.")


    # --- Optimal Path Details (Textual) for VISUALIZATION_Q_UAVS ---
    print(f"\n--- Optimal Path Details for K={FIXED_M_CLUSTERS}, Q_UAVS = {VISUALIZATION_Q_UAVS} ---")
    print(f"Best Total Path Cost: {visual_best_overall_cost:.2f} km")

    print("\nOV Best Path Order (Cluster IDs):")
    ov_path_str_display = []
    for c_id in visual_best_ov_path_ids:
        if c_id == visual_depot_cluster_id:
            ov_path_str_display.append(f"DEPOT({c_id})")
        else:
            ov_path_str_display.append(f"C{c_id}")
    print(" -> ".join(ov_path_str_display))

    print("\nUAV Best Paths within Clusters (each sortie: OV Base -> Node -> OV Base):")
    for cluster_id, uav_routes in visual_best_uav_paths_in_clusters.items():
        centroid_display = visual_cluster_centroids[cluster_id]
        print(f"  Cluster {cluster_id} (OV Base at [{centroid_display['lat']:.4f}, {centroid_display['lon']:.4f}]):")
        for uav_q, route_nodes_ids in uav_routes.items():
            if route_nodes_ids:
                display_sorties = []
                for i in range(0, len(route_nodes_ids), 3):
                    if i + 1 < len(route_nodes_ids):
                        display_sorties.append(f"{route_nodes_ids[i]} -> {route_nodes_ids[i+1]} -> {route_nodes_ids[i+2]}")
                print(f"    UAV {uav_q}: {'; '.join(display_sorties)}")
            else:
                print(f"    UAV {uav_q}: (No assigned nodes in this cluster)")
else:
    print(f"\nNo detailed visualizations or textual output for Q_UAVS={VISUALIZATION_Q_UAVS} because no valid results were stored for it.")


# --- Combined Convergence Analysis Visualization (for all Q values) ---
print("\n--- Plotting Combined Convergence Analysis for all Q values ---")
plot_convergence_analysis_varying_q(convergence_histories_all_q, OV_ACO_PARAMS['iters_ov'])


# --- Runtime Comparison Visualization ---
print("\n--- Plotting Runtime Comparison for all Q values ---")
plot_runtime_comparison_varying_q(runtimes_all_q)


# --- Save final results ---
print("\n--- Updating and Saving Final Simulation Results to Google Drive ---")
try:
    final_data_to_save_raw = {
        "nodes": visual_all_nodes, # Nodes with cluster_id for FIXED_M_CLUSTERS
        "depot_actual_node_id": depot_actual_node_id_original,
        "generated_lv_edges": generated_lv_edges_original,
        "bbox_coords": bbox_coords_original,
        "obstacle_zones_data": obstacle_zones_data_original,
        "cluster_centroids": visual_cluster_centroids, # For FIXED_M_CLUSTERS
        "best_ov_path_ids": visual_best_ov_path_ids, # For VISUALIZATION_Q_UAVS
        "best_uav_paths_in_clusters": visual_best_uav_paths_in_clusters, # For VISUALIZATION_Q_UAVS
        "best_overall_cost": visual_best_overall_cost, # For VISUALIZATION_Q_UAVS
        "convergence_history": visual_convergence_history, # For VISUALIZATION_Q_UAVS (its individual history)
        "convergence_histories_all_q": convergence_histories_all_q, # All convergence histories by Q
        "runtimes_all_q": runtimes_all_q, # All runtimes by Q
        "simulation_parameters": {
            "N_NODES_CONFIG": N_NODES_CONFIG,
            "FIXED_M_CLUSTERS": FIXED_M_CLUSTERS,
            "Q_UAVS_VISUALIZED": VISUALIZATION_Q_UAVS,
            "Q_UAVS_FOR_ANALYSIS": Q_UAVS_FOR_ANALYSIS,
            "R_UAV_KM": R_UAV_KM,
            "OV_ACO_PARAMS": OV_ACO_PARAMS,
            "UAV_ACO_PARAMS": UAV_ACO_PARAMS
        }
    }
    final_data_to_save_converted = convert_numpy_types(final_data_to_save_raw)

    drive_folder_path = '/content/drive/MyDrive/PowerLine/getn/'
    full_file_path = os.path.join(drive_folder_path, "ACO_LV_Network_Simulation_dataset.json")

    print(f"Saving final simulation results to: {full_file_path}")
    with open(full_file_path, 'w') as f:
        json.dump(final_data_to_save_converted, f, indent=4)
    print("Final data successfully saved to Google Drive!")
    print(f"File location: {full_file_path}")
except Exception as e:
    print(f"Error saving final data to file: {e}")
