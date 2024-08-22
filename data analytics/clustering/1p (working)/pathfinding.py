import pandas as pd
import numpy as np
import heapq
from scipy.spatial import distance
import folium
from folium import Popup
from data_processing import load_clustered_data, calculate_centroids, calculate_crime_counts

def build_graph(centroids, crime_counts):
    """Build the graph with nodes as cluster centroids and edges with weights."""
    graph = {}
    for i, row in centroids.iterrows():
        cluster_id = row['Cluster']
        graph[cluster_id] = {}
        for (start, end), count in crime_counts.items():
            if start == cluster_id:
                graph[start][end] = count
            elif end == cluster_id:
                graph[end][start] = count
    return graph

def dijkstra(graph, start, end):
    """Apply Dijkstra's algorithm to find the shortest path from start to end."""
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
    
    # Reconstruct path
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    path.reverse()
    
    return path, distances[end]

def find_nearest_centroid(lat, lon, centroids):
    """Find the nearest centroid to the given latitude and longitude."""
    distances = centroids.apply(
        lambda row: distance.euclidean((lat, lon), (row['Centroid_Latitude'], row['Centroid_Longitude'])),
        axis=1
    )
    nearest_centroid = centroids.iloc[distances.idxmin()]
    return nearest_centroid['Cluster'], (nearest_centroid['Centroid_Latitude'], nearest_centroid['Centroid_Longitude'])

def visualize_path(centroids, path, output_file='shortest_path_map.html'):
    """Visualize the path on a map."""
    # Initialize the map centered around the first point in the path
    start_lat, start_lon = centroids[centroids['Cluster'] == path[0]][['Centroid_Latitude', 'Centroid_Longitude']].values[0]
    crime_map = folium.Map(location=[start_lat, start_lon], zoom_start=12)
    
    # Add markers for centroids
    for i, row in centroids.iterrows():
        folium.Marker(
            location=[row['Centroid_Latitude'], row['Centroid_Longitude']],
            popup=f"Cluster {row['Cluster']}",
            icon=folium.Icon(color='blue')
        ).add_to(crime_map)
    
    # Add markers for the path
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        start_coords = centroids[centroids['Cluster'] == start_node][['Centroid_Latitude', 'Centroid_Longitude']].values[0]
        end_coords = centroids[centroids['Cluster'] == end_node][['Centroid_Latitude', 'Centroid_Longitude']].values[0]
        
        folium.Marker(
            location=start_coords,
            popup=f"Cluster {start_node}",
            icon=folium.Icon(color='green')
        ).add_to(crime_map)
        
        folium.Marker(
            location=end_coords,
            popup=f"Cluster {end_node}",
            icon=folium.Icon(color='red')
        ).add_to(crime_map)
        
        # Draw a line between the nodes
        folium.PolyLine(
            locations=[start_coords, end_coords],
            color='blue',
            weight=2.5,
            opacity=1
        ).add_to(crime_map)
    
    # Save the map to an HTML file
    crime_map.save(output_file)
    print(f"Map saved as '{output_file}'.")

def main():
    # File paths
    input_file_path = 'clustered_sampled_dataset_by_area.csv'
    
    # Load data
    clustered_data = load_clustered_data(input_file_path)
    
    # Calculate centroids
    centroids = calculate_centroids(clustered_data)
    print("Centroids calculated:")
    print(centroids)
    
    # Calculate crime counts
    crime_counts = calculate_crime_counts(clustered_data, centroids)
    print("Crime counts:")
    print(crime_counts)
    
    # Build the graph
    graph = build_graph(centroids, crime_counts)
    print("Graph built:")
    print(graph)
    
    # User input for start and end points
    start_lat = float(input("Enter the latitude for the start point: "))
    start_lon = float(input("Enter the longitude for the start point: "))
    end_lat = float(input("Enter the latitude for the end point: "))
    end_lon = float(input("Enter the longitude for the end point: "))
    
    # Find nearest centroids
    start_centroid, _ = find_nearest_centroid(start_lat, start_lon, centroids)
    end_centroid, _ = find_nearest_centroid(end_lat, end_lon, centroids)
    print(f"Nearest centroids: start={start_centroid}, end={end_centroid}")
    
    # Find shortest path
    path, shortest_path_distance = dijkstra(graph, start_centroid, end_centroid)
    print(f"The shortest path distance from cluster {start_centroid} to cluster {end_centroid} is {shortest_path_distance}.")
    
    # Visualize the path on a map
    visualize_path(centroids, path)

if __name__ == "__main__":
    main()
