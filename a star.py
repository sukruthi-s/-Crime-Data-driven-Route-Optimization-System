import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
import heapq
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

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    return geodesic((lat1, lon1), (lat2, lon2)).km

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
    df['Weighted Count'] = df['Crime Type'].map(crime_scores) * df['Count']
    cluster_stats = df.groupby(['Area', 'Cluster']).agg({
        'Weighted Count': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    cluster_stats['Danger Index'] = (cluster_stats['Weighted Count'] - cluster_stats['Weighted Count'].min()) / \
                                   (cluster_stats['Weighted Count'].max() - cluster_stats['Weighted Count'].min())
    return cluster_stats

def astar_shortest_paths(start, goal, nodes, danger_index):
    """A* algorithm implementation with detailed metrics"""
    open_set = []
    heapq.heappush(open_set, (0, start, [start], 0))  # Cost, node, path, distance
    g_score = {node: float('inf') for node in nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in nodes}
    f_score[start] = calculate_distance(
        nodes[start]['latitude'], nodes[start]['longitude'],
        nodes[goal]['latitude'], nodes[goal]['longitude']
    )
    
    path_metrics = {
        'expanded_nodes': 0,
        'path_distances': [],
        'total_physical_distance': 0,
        'total_danger': 0
    }

    while open_set:
        path_metrics['expanded_nodes'] += 1
        _, current, path, total_distance = heapq.heappop(open_set)

        if current == goal:
            # Calculate final path metrics
            for i in range(len(path)-1):
                current_node = path[i]
                next_node = path[i+1]
                distance = calculate_distance(
                    nodes[current_node]['latitude'], nodes[current_node]['longitude'],
                    nodes[next_node]['latitude'], nodes[next_node]['longitude']
                )
                path_metrics['path_distances'].append(distance)
                path_metrics['total_danger'] += nodes[current_node]['danger_index']
            
            # Add final node danger
            path_metrics['total_danger'] += nodes[path[-1]]['danger_index']
            path_metrics['total_physical_distance'] = sum(path_metrics['path_distances'])
            
            return path, path_metrics, g_score[goal]

        for neighbor in nodes[current]['neighbors']:
            distance = calculate_distance(
                nodes[current]['latitude'], nodes[current]['longitude'],
                nodes[neighbor]['latitude'], nodes[neighbor]['longitude']
            )
            tentative_g_score = g_score[current] + distance + danger_index.get(neighbor, 0)

            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + calculate_distance(
                    nodes[neighbor]['latitude'], nodes[neighbor]['longitude'],
                    nodes[goal]['latitude'], nodes[goal]['longitude']
                )
                new_total_distance = total_distance + distance
                heapq.heappush(open_set, (f_score[neighbor], neighbor, path + [neighbor], new_total_distance))

    return None, path_metrics, float('inf')

def main():
    print("\nRunning A* Algorithm...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Load data
    df = pd.read_csv('baltimore_test_1.csv')
    if 'Count' not in df.columns:
        df['Count'] = 1
    print("Data loaded successfully")

    preprocessing_start = time.time()
    # Create clusters
    df = create_clusters(df)
    print("Clustering complete")
    
    # Calculate danger indices
    cluster_stats = calculate_cluster_danger_index(df, CRIME_SCORES)
    danger_index = dict(zip(
        [f"{row['Area']}_cluster_{row['Cluster']}" for _, row in cluster_stats.iterrows()],
        cluster_stats['Danger Index']
    ))
    print("Danger indices calculated")

    # Create nodes
    nodes = {}
    for _, row in cluster_stats.iterrows():
        cluster_id = f"{row['Area']}_cluster_{row['Cluster']}"
        nodes[cluster_id] = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'danger_index': row['Danger Index'],
            'neighbors': []
        }

    # Connect nodes
    threshold_distance = 1.5  # For intra-area connections
    inter_area_threshold_distance = 6.0  # For inter-area connections

    edge_count = 0
    for cluster1 in nodes:
        for cluster2 in nodes:
            if cluster1 != cluster2:
                distance = calculate_distance(
                    nodes[cluster1]['latitude'], nodes[cluster1]['longitude'],
                    nodes[cluster2]['latitude'], nodes[cluster2]['longitude']
                )
                if (cluster1.split('_cluster_')[0] == cluster2.split('_cluster_')[0] and distance < threshold_distance) or \
                   (cluster1.split('_cluster_')[0] != cluster2.split('_cluster_')[0] and distance < inter_area_threshold_distance):
                    nodes[cluster1]['neighbors'].append(cluster2)
                    edge_count += 1
    
    preprocessing_time = time.time() - preprocessing_start
    print("Graph construction complete")

    # Define start and end points
    start_area = "Carroll Park"
    end_area = "Overlea"

    # Get clusters in start and end areas
    start_clusters = [node for node in nodes if node.startswith(f"{start_area}_")]
    end_clusters = [node for node in nodes if node.startswith(f"{end_area}_")]

    if not start_clusters or not end_clusters:
        print("Start or end area not found")
        return

    start_cluster = start_clusters[0]
    end_cluster = end_clusters[0]

    print(f"Start Cluster: {start_cluster}")
    print(f"End Cluster: {end_cluster}")

    # Find path
    pathfinding_start = time.time()
    path, path_metrics, total_cost = astar_shortest_paths(start_cluster, end_cluster, nodes, danger_index)
    pathfinding_time = time.time() - pathfinding_start
    
    # Calculate final metrics
    total_execution_time = time.time() - start_time
    peak_memory = get_memory_usage() - initial_memory

    # Print results
    if path:
        print("\nA* Algorithm Results:")
        print("=" * 80)
        print("\nPath Taken:")
        
        for i, node in enumerate(path):
            print(f"{i+1}. {node} (Danger Index: {nodes[node]['danger_index']:.2f})")
            if i < len(path) - 1 and path_metrics['path_distances']:
                print(f"   Distance to next: {path_metrics['path_distances'][i]:.2f}km")

        areas_crossed = " → ".join([node.split('_cluster_')[0] for node in path])
        
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
        print(f"Number of Nodes Expanded: {path_metrics['expanded_nodes']}")
        print(f"Number of Graph Nodes: {len(nodes)}")
        print(f"Number of Graph Edges: {edge_count}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()