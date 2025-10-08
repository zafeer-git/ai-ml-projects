import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import kagglehub
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os # Import the os module to list directory contents
import seaborn as sns

# Download latest version
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")

print("Path to dataset files:", path)

csv_file_name = "Mall_Customers.csv" # Based on common practice and other Kaggle notebooks for this dataset
full_csv_path = os.path.join(path, csv_file_name)

# Load the CSV into a pandas DataFrame
try:
    df = pd.read_csv(full_csv_path)
    print("Dataset loaded successfully!")
    print(df.columns)
    print(df.head()) # Display the first few rows to verify
except Exception as e:
    print(f"Error loading dataset: {e}")


# --- Feature Selection ---
# As discussed, 'Annual Income (k$)' and 'Spending Score (1-100)' are ideal for this segmentation.
features_for_clustering = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features_for_clustering]

print(f"\nFeatures selected for clustering: {features_for_clustering}")
print("\nFirst 5 rows of selected features:\n", X.head())

# --- Correlation Matrix Analysis ---
print("\n--- Correlation Matrix of Selected Features ---")
correlation_matrix = X.corr()
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap for better interpretation
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Annual Income and Spending Score')
plt.show()



# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5]) # Display the first 5 rows of the scaled data




# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11): # Testing for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Apply K-Means with the optimal number of clusters (5)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = clusters

print(df.head())


# Visualize the clusters and centroids
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', legend='full')

# Plot the centroids
# Need to unscale the centroids to plot them on the original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

