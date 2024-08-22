import pandas as pd

# Load your dataset from the CSV file, stripping whitespace from headers and handling mixed types
data = pd.read_csv('merged dataset.csv', dtype=str).rename(columns=lambda x: x.strip())

# Convert Latitude and Longitude columns to numeric, forcing errors to NaN
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

# Drop any rows with missing or invalid Latitude/Longitude values
data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from Latitude and Longitude
data = remove_outliers(data, 'Latitude')
data = remove_outliers(data, 'Longitude')

# Define the percentage of points to sample (between 20% and 30%)
sample_percentage = 0.02  # For example, 1%

# Function to sample points from each area
def sample_area(df, percentage):
    return df.sample(frac=percentage, random_state=1)  # Set a random_state for reproducibility

# Apply the sampling function to each group
sampled_data = data.groupby('Area').apply(sample_area, percentage=sample_percentage).reset_index(drop=True)

# Print the number of instances in the sampled dataset
print(f"Number of instances in the sampled dataset: {len(sampled_data)}")

# Save the sampled data to a new CSV file
sampled_data.to_csv('sampled_dataset_2p_wo_outliers.csv', index=False)

print(f"Sampled dataset saved as 'sampled_dataset_2p_wo_outliers.csv' with {sample_percentage*100}% of points from each area.")
