import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import time
from geopy.distance import geodesic
import psutil
import os

warnings.simplefilter("ignore", ConvergenceWarning)

CRIME_SCORES = {
    'HOMICIDE': 7,
    'SHOOTING': 7,
    'RAPE': 6,
    'AGG. ASSAULT': 6,
    'ROBBERY - CARJACKING': 6,
    'ROBBERY - COMMERCIAL': 5,
    'ROBBERY - RESIDENCE': 5,
    'ROBBERY - STREET': 5,
    'BURGLARY': 4,
    'ASSAULT BY THREAT': 4,
    'AUTO THEFT': 3,
    'ARSON': 3,
    'COMMON ASSAULT': 2,
    'LARCENY': 1,
    'LARCENY FROM AUTO': 1
}

class SafestPathGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(SafestPathGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def create_clusters(df, n_clusters=7):
    """Create clusters within each area"""
    area_groups = df.groupby('Area')
    clustered_data = []
    
    for area, group in area_groups:
        coords = group[['Latitude', 'Longitude']].values
        if len(coords) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(coords)
            group['Cluster'] = kmeans.labels_
        else:
            group['Cluster'] = 0
        clustered_data.append(group)
    
    return pd.concat(clustered_data, ignore_index=True)

def calculate_cluster_danger_index(df, crime_scores):
    """Calculate danger index for each cluster"""
    df['Weighted_Count'] = df['Crime Type'].map(crime_scores) * df['Count']
    cluster_stats = df.groupby(['Area', 'Cluster']).agg({
        'Weighted_Count': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    cluster_stats['Danger_Index'] = (cluster_stats['Weighted_Count'] - cluster_stats['Weighted_Count'].min()) / \
                                   (cluster_stats['Weighted_Count'].max() - cluster_stats['Weighted_Count'].min())
    return cluster_stats

def find_safest_path_gnn(start, goal, nodes, model, graph_data, node_list, centroids):
    """Find safest path using trained GNN model"""
    model.eval()
    with torch.no_grad():
        node_scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr).squeeze().tolist()

    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    previous = {node: None for node in nodes}
    unvisited = set(nodes.keys())
    
    total_danger = 0
    path_distances = []

    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        if current == goal:
            break

        unvisited.remove(current)
        current_idx = node_list.index(current)

        for neighbor in nodes[current]['neighbors']:
            if neighbor in unvisited:
                neighbor_idx = node_list.index(neighbor)
                
                # Calculate physical distance
                current_coords = (nodes[current]['latitude'], nodes[current]['longitude'])
                neighbor_coords = (nodes[neighbor]['latitude'], nodes[neighbor]['longitude'])
                physical_distance = geodesic(current_coords, neighbor_coords).km
                
                # Combine physical distance with GNN score
                distance = distances[current] + (physical_distance * (1 + node_scores[neighbor_idx]))
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current

    # Reconstruct path and calculate metrics
    path = []
    current = goal
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()

    # Calculate detailed metrics
    for i in range(len(path)-1):
        current_node = path[i]
        next_node = path[i+1]
        
        # Get danger indices
        current_area, current_cluster = current_node.split('_cluster_')
        current_data = centroids[(centroids['Area'] == current_area) & 
                               (centroids['Cluster'] == int(current_cluster))]
        total_danger += current_data['Danger_Index'].iloc[0]
        
        # Calculate physical distance between consecutive nodes
        current_coords = (nodes[current_node]['latitude'], nodes[current_node]['longitude'])
        next_coords = (nodes[next_node]['latitude'], nodes[next_node]['longitude'])
        path_distances.append(geodesic(current_coords, next_coords).km)

    # Add final node's danger
    final_area, final_cluster = path[-1].split('_cluster_')
    final_data = centroids[(centroids['Area'] == final_area) & 
                          (centroids['Cluster'] == int(final_cluster))]
    total_danger += final_data['Danger_Index'].iloc[0]

    metrics = {
        'path': path,
        'total_cost': distances[goal],
        'total_danger': total_danger,
        'avg_danger': total_danger / len(path),
        'path_length': len(path),
        'physical_distances': path_distances,
        'total_physical_distance': sum(path_distances)
    }

    return metrics

def main():
    print("\nRunning GNN Algorithm...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Load data
    df = pd.read_csv('baltimore_test_1.csv')
    if 'Count' not in df.columns:
        df['Count'] = 1
    
    # Create clusters
    df = create_clusters(df)
    print("Clustering complete")
    
    # Calculate danger indices and create nodes
    cluster_stats = calculate_cluster_danger_index(df, CRIME_SCORES)
    
    # Create nodes dictionary
    nodes = {}
    for _, row in cluster_stats.iterrows():
        cluster_id = f"{row['Area']}_cluster_{row['Cluster']}"
        nodes[cluster_id] = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'danger_index': row['Danger_Index'],
            'neighbors': []
        }

    # Connect nodes
    threshold_distance = 1.5  # For intra-area connections
    inter_area_threshold_distance = 6.0  # For inter-area connections

    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                coords1 = (nodes[node1]['latitude'], nodes[node1]['longitude'])
                coords2 = (nodes[node2]['latitude'], nodes[node2]['longitude'])
                distance = geodesic(coords1, coords2).km
                
                if (node1.split('_cluster_')[0] == node2.split('_cluster_')[0] and distance < threshold_distance) or \
                   (node1.split('_cluster_')[0] != node2.split('_cluster_')[0] and distance < inter_area_threshold_distance):
                    nodes[node1]['neighbors'].append(node2)

    # Prepare data for GNN
    node_list = list(nodes.keys())
    edge_index = []
    edge_attr = []
    
    for node_id, node_data in nodes.items():
        for neighbor_id in node_data['neighbors']:
            source_idx = node_list.index(node_id)
            target_idx = node_list.index(neighbor_id)
            edge_index.append([source_idx, target_idx])
            
            distance = geodesic(
                (nodes[node_id]['latitude'], nodes[node_id]['longitude']),
                (nodes[neighbor_id]['latitude'], nodes[neighbor_id]['longitude'])
            ).km
            danger = nodes[neighbor_id]['danger_index']
            edge_attr.append([distance, danger])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_features = torch.tensor([[nodes[node]['latitude'], nodes[node]['longitude']] 
                                for node in nodes], dtype=torch.float)
    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Train model
    model = SafestPathGNN(num_node_features=2, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training GNN model...")
    training_start = time.time()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        
        # Use danger indices as target
        target = torch.tensor([[nodes[node]['danger_index']] for node in node_list], dtype=torch.float)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/200, Loss: {loss.item():.4f}")
    
    training_time = time.time() - training_start
    print("Model training complete")

    # Define start and end points
    start_area = "Carroll Park"
    end_area = "Overlea"
    
    # Get clusters in start and end areas
    start_clusters = cluster_stats[cluster_stats['Area'] == start_area]
    end_clusters = cluster_stats[cluster_stats['Area'] == end_area]
    
    if start_clusters.empty or end_clusters.empty:
        print("Start or end area not found")
        return
    
    start_cluster = f"{start_area}_cluster_{start_clusters.iloc[0]['Cluster']}"
    end_cluster = f"{end_area}_cluster_{end_clusters.iloc[0]['Cluster']}"
    
    # Find path
    path_metrics = find_safest_path_gnn(start_cluster, end_cluster, nodes, model, 
                                      graph_data, node_list, cluster_stats)
    
    # Calculate final metrics
    execution_time = time.time() - start_time
    peak_memory = get_memory_usage() - initial_memory
    
    # Print results
    print("\nGNN Algorithm Results:")
    print("=" * 80)
    print("\nPath Taken:")
    areas_crossed = []
    
    for i, node in enumerate(path_metrics['path']):
        area, cluster = node.split('_cluster_')
        cluster_data = cluster_stats[(cluster_stats['Area'] == area) & 
                                   (cluster_stats['Cluster'] == int(cluster))]
        danger_index = cluster_data['Danger_Index'].iloc[0]
        areas_crossed.append(area)
        
        print(f"{i+1}. {node} (Danger Index: {danger_index:.2f})")
        if i < len(path_metrics['path']) - 1:
            print(f"   Distance to next: {path_metrics['physical_distances'][i]:.2f}km")

    unique_areas = " → ".join(areas_crossed)
    
    print("\nMetrics:")
    print("-" * 60)
    print(f"Total Path Cost: {path_metrics['total_cost']:.2f}")
    print(f"Average Danger Index: {path_metrics['avg_danger']:.2f}")
    print(f"Number of Clusters Traversed: {path_metrics['path_length']}")
    print(f"Total Physical Distance: {path_metrics['total_physical_distance']:.2f}km")
    print(f"Areas Crossed: {unique_areas}")
    print("\nPerformance Metrics:")
    print("-" * 60)
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    print(f"Number of Graph Nodes: {len(nodes)}")
    print(f"Number of Graph Edges: {len(edge_index[0])}")

if __name__ == "__main__":
    main()