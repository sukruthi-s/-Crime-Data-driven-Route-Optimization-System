import pandas as pd

def load_clustered_data(file_path):
    """Load clustered data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_centroids(clustered_data):
    """Calculate the centroid of each cluster."""
    centroids = clustered_data.groupby('Cluster').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
    centroids.rename(columns={'Latitude': 'Centroid_Latitude', 'Longitude': 'Centroid_Longitude'}, inplace=True)
    return centroids


def calculate_crime_counts(clustered_data, centroids):
    """Calculate the number of crimes between different clusters."""
    crime_counts = {}
    for (start_cluster, end_cluster), group in clustered_data.groupby(['Start_Cluster', 'End_Cluster']):
        crime_count = len(group)
        crime_counts[(start_cluster, end_cluster)] = crime_count
        print(f"Crime count between cluster {start_cluster} and {end_cluster}: {crime_count}")
    return crime_counts
