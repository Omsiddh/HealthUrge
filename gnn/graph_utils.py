import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None
    Data = None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_temporal_features():
    """Extract time-based features (hour, day of week, is_weekend)"""
    now = datetime.now()
    hour = now.hour / 24.0  # Normalize to [0, 1]
    day_of_week = now.weekday() / 6.0  # Monday=0, Sunday=6, normalized
    is_weekend = 1.0 if now.weekday() >= 5 else 0.0
    return hour, day_of_week, is_weekend

def build_hospital_graph(hospitals, k_neighbors=5, max_distance_km=50.0, scaler=None, fit_scaler=False):
    """
    Converts a list of hospital dicts into a PyG Data object with improved features.
    
    Args:
        hospitals: List of dicts with keys: id, beds_capacity, predicted_load, latitude, longitude
        k_neighbors: Connect each hospital to its K nearest neighbors
        max_distance_km: Maximum distance for edge connection (prevents very distant connections)
        scaler: StandardScaler for feature normalization (pass existing scaler for inference)
        fit_scaler: Whether to fit the scaler (True for training, False for inference)
    
    Returns:
        Data object with node features, edges, and the scaler used
    """
    if torch is None or Data is None:
        return None, None
    
    if not hospitals:
        raise ValueError("Hospital list cannot be empty")
    
    # Extract temporal features (same for all hospitals in this snapshot)
    hour, day_of_week, is_weekend = get_temporal_features()
    
    # Build raw node features: [capacity, load, load_ratio, available_beds, lat, lon, hour, day, weekend]
    node_features = []
    
    for h in hospitals:
        cap = float(h.get('beds_capacity', 1))  # Avoid division by zero
        load = float(h.get('predicted_load', 0))
        ratio = min(load / cap, 2.0) if cap > 0 else 0  # Cap at 200% to prevent outliers
        available = max(cap - load, 0)
        lat = float(h.get('latitude', 0))
        lon = float(h.get('longitude', 0))
        
        node_features.append([
            cap,
            load,
            ratio,
            available,
            lat,
            lon,
            hour,
            day_of_week,
            is_weekend
        ])
    
    node_features = np.array(node_features, dtype=np.float32)
    
    # Normalize features using StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    
    if fit_scaler:
        node_features_normalized = scaler.fit_transform(node_features)
    else:
        node_features_normalized = scaler.transform(node_features)
    
    x = torch.tensor(node_features_normalized, dtype=torch.float)
    
    # Build edges using K-nearest neighbors + distance threshold
    edge_index = []
    edge_attr = []
    
    num_nodes = len(hospitals)
    
    # Compute all pairwise distances
    distances = {}
    for i in range(num_nodes):
        distances[i] = []
        for j in range(num_nodes):
            if i != j:
                dist = calculate_distance(
                    hospitals[i]['latitude'], hospitals[i]['longitude'],
                    hospitals[j]['latitude'], hospitals[j]['longitude']
                )
                distances[i].append((j, dist))
        
        # Sort by distance
        distances[i].sort(key=lambda x: x[1])
    
    # Connect each node to its K nearest neighbors (within max distance)
    for i in range(num_nodes):
        connected = 0
        for j, dist in distances[i]:
            if connected >= k_neighbors:
                break
            if dist <= max_distance_km:
                # Add bidirectional edge
                edge_index.append([i, j])
                
                # Edge features: [distance, inverse_distance, capacity_difference]
                inv_dist = 1.0 / (dist + 0.1)
                cap_diff = abs(node_features[i][0] - node_features[j][0]) / 1000.0  # Normalized capacity difference
                
                edge_attr.append([dist / max_distance_km, inv_dist, cap_diff])
                connected += 1
    
    if not edge_index:
        # If no edges, create self-loops to prevent errors
        print("Warning: No edges found. Creating self-loops.")
        edge_index = [[i, i] for i in range(num_nodes)]
        edge_attr = [[0.0, 1.0, 0.0] for _ in range(num_nodes)]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), scaler


def create_stress_labels(hospitals):
    """
    Create ground truth stress labels for training.
    This is a heuristic - in production, use actual historical stress data.
    
    Returns: Tensor of shape (num_hospitals, 1) with values in [0, 1]
    """
    labels = []
    for h in hospitals:
        cap = float(h.get('beds_capacity', 1))
        load = float(h.get('predicted_load', 0))
        ratio = load / cap if cap > 0 else 0
        
        # Stress mapping:
        # 0-70% load -> 0.0-0.3 stress
        # 70-90% load -> 0.3-0.6 stress
        # 90-100% load -> 0.6-0.8 stress
        # 100%+ load -> 0.8-1.0 stress
        if ratio <= 0.7:
            stress = ratio / 0.7 * 0.3
        elif ratio <= 0.9:
            stress = 0.3 + (ratio - 0.7) / 0.2 * 0.3
        elif ratio <= 1.0:
            stress = 0.6 + (ratio - 0.9) / 0.1 * 0.2
        else:
            stress = min(0.8 + (ratio - 1.0) / 0.5 * 0.2, 1.0)
        
        labels.append([stress])
    
    return torch.tensor(labels, dtype=torch.float)