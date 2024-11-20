import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
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

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def cluster_within_areas(df, n_clusters=7):
    """Create clusters within each area"""
    if 'Count' not in df.columns:
        df['Count'] = 1
    
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

def build_safety_graph_optimized(clusters_df):
    """Build graph with optimized spatial indexing"""
    # Pre-calculate cluster centroids and danger indices
    centroids = clusters_df.groupby(['Area', 'Cluster']).agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Count': 'sum'
    }).reset_index()
    
    centroids['Weighted_Count'] = centroids['Count']  # Simplified for speed
    centroids['Danger_Index'] = (centroids['Weighted_Count'] - centroids['Weighted_Count'].min()) / \
                               (centroids['Weighted_Count'].max() - centroids['Weighted_Count'].min())
    
    # Build graph
    graph = {}
    edge_count = 0
    
    for i, row1 in centroids.iterrows():
        node1 = f"{row1['Area']}_cluster_{row1['Cluster']}"
        graph[node1] = {}
        coords1 = (row1['Latitude'], row1['Longitude'])
        
        for j, row2 in centroids.iterrows():
            if i != j:
                coords2 = (row2['Latitude'], row2['Longitude'])
                distance = geodesic(coords1, coords2).km
                
                # Connect if within threshold
                threshold = 1.5 if row1['Area'] == row2['Area'] else 6.0
                if distance < threshold:
                    node2 = f"{row2['Area']}_cluster_{row2['Cluster']}"
                    weight = distance * (1 + row2['Danger_Index'])
                    graph[node1][node2] = weight
                    edge_count += 1
    
    return graph, centroids, edge_count

def find_safest_path(graph, start, end):
    """Find safest path using Dijkstra's algorithm with detailed metrics"""
    path_metrics = {
        'path_distances': [],
        'total_physical_distance': 0,
        'total_danger': 0
    }
    
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    unvisited = set(graph.keys())
    
    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        
        if current == end:
            break
            
        unvisited.remove(current)
        
        for neighbor, weight in graph[current].items():
            if neighbor in unvisited:
                new_distance = distances[current] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, distances[end], path_metrics

def main():
    print("\nRunning Dijkstra Algorithm...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Load data
    df = pd.read_csv('baltimore_test_1.csv')
    if 'Count' not in df.columns:
        df['Count'] = 1
    print("Data loaded successfully")
    
    preprocessing_start = time.time()
    # Create clusters
    df = cluster_within_areas(df)
    print("Clustering complete")
    
    # Build graph
    graph, centroids, edge_count = build_safety_graph_optimized(df)
    preprocessing_time = time.time() - preprocessing_start
    print("Graph construction complete")
    
    # Define start and end points
    start_area = "Carroll Park"
    end_area = "Overlea"
    
    # Get clusters in start and end areas
    start_clusters = centroids[centroids['Area'] == start_area]
    end_clusters = centroids[centroids['Area'] == end_area]
    
    if start_clusters.empty or end_clusters.empty:
        print("Start or end area not found")
        return
    
    start_cluster = f"{start_area}_cluster_{start_clusters.iloc[0]['Cluster']}"
    end_cluster = f"{end_area}_cluster_{end_clusters.iloc[0]['Cluster']}"
    
    print(f"Start Cluster: {start_cluster}")
    print(f"End Cluster: {end_cluster}")
    
    # Find path
    pathfinding_start = time.time()
    path, total_cost, path_metrics = find_safest_path(graph, start_cluster, end_cluster)
    pathfinding_time = time.time() - pathfinding_start
    
    # Calculate final metrics
    total_execution_time = time.time() - start_time
    peak_memory = get_memory_usage() - initial_memory
    
    # Print results
    if path:
        print("\nDijkstra Algorithm Results:")
        print("=" * 80)
        print("\nPath Taken:")
        areas_crossed = []

        for i, node in enumerate(path):
            area, cluster = node.split('_cluster_')
            cluster_data = centroids[(centroids['Area'] == area) & (centroids['Cluster'] == int(cluster))]
            danger_index = cluster_data['Danger_Index'].iloc[0]
            path_metrics['total_danger'] += danger_index
            areas_crossed.append(area)

            print(f"{i+1}. {node} (Danger Index: {danger_index:.2f})")
            
            if i < len(path) - 1:
                next_node = path[i + 1]
                next_area, next_cluster = next_node.split('_cluster_')
                next_data = centroids[(centroids['Area'] == next_area) & (centroids['Cluster'] == int(next_cluster))]

                distance = geodesic(
                    (cluster_data['Latitude'].iloc[0], cluster_data['Longitude'].iloc[0]),
                    (next_data['Latitude'].iloc[0], next_data['Longitude'].iloc[0])
                ).km
                path_metrics['path_distances'].append(distance)
                print(f"   Distance to next: {distance:.2f}km")
        
        path_metrics['total_physical_distance'] = sum(path_metrics['path_distances'])
        areas_crossed = " → ".join(areas_crossed)
        
        print("\nPath Metrics:")
        print("-" * 60)
        print(f"Total Physical Distance: {path_metrics['total_physical_distance']:.2f} km")
        print(f"Total Path Cost (including danger weights): {total_cost:.2f}")
        print(f"Total Danger Index: {path_metrics['total_danger']:.2f}")
        print(f"Average Danger per Cluster: {path_metrics['total_danger']/len(path):.2f}")
        print(f"Number of Clusters Traversed: {len(path)}")
        print(f"Areas Crossed: {areas_crossed}")
        
        print("\nPerformance Metrics:")
        print("-" * 60)
        print(f"Total Execution Time: {total_execution_time:.2f} seconds")
        print(f"Preprocessing Time: {preprocessing_time:.2f} seconds")
        print(f"Pathfinding Time: {pathfinding_time:.2f} seconds")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")
        print(f"Number of Graph Nodes: {len(graph)}")
        print(f"Number of Graph Edges: {edge_count}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()