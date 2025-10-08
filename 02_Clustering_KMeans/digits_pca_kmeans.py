import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np


digits = load_digits()
X = digits.data
y_true = digits.target  # True labels for evaluation

n_samples, n_features = X.shape
n_digits = len(np.unique(y_true)) # Number of clusters we expect (0-9, so 10)

print(f"Original dataset shape: {X.shape}")
print(f"Number of true digits (classes): {n_digits}")
print("-" * 30)

# It's good practice to scale the data, especially for PCA and K-Means.
# K-Means is sensitive to feature scales.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. K-Means Clustering on Original Dataset (without PCA) ---
print("Performing K-Means on original dataset...")
kmeans_original = KMeans(n_clusters=n_digits, random_state=42, n_init=10)
kmeans_original.fit(X_scaled)
labels_original = kmeans_original.labels_

# Evaluate clustering without PCA
ari_original = adjusted_rand_score(y_true, labels_original)
silhouette_original = silhouette_score(X_scaled, labels_original)

print(f"K-Means (Original Data) - Adjusted Rand Index (ARI): {ari_original:.3f}")
print(f"K-Means (Original Data) - Silhouette Score: {silhouette_original:.3f}")
print("-" * 30)

# --- 3. Apply PCA (Principal Component Analysis) to reduce dimensionality ---
# Let's reduce to a smaller number of components, e.g., 20 or 30
# We can also choose n_components such that it explains a certain variance,
# but for demonstration, let's pick a fixed number.
n_components = 20 # A common choice for this dataset to retain significant variance

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced dataset shape (after PCA): {X_pca.shape}")
print(f"Explained variance ratio by {n_components} components: {np.sum(pca.explained_variance_ratio_):.3f}")
print("-" * 30)

# Visualize the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# If you want to visualize in 2D (for K-Means after PCA, limited to 2 components)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_true, cmap='Paired', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Digits Dataset after PCA (2 Components) with True Labels')
plt.colorbar(scatter, label='True Digit Label')
plt.grid(True)
plt.show()


# --- 4. Perform K-Means Clustering on the Reduced Dataset ---
print(f"Performing K-Means on PCA-reduced dataset (n_components={n_components})...")
kmeans_pca = KMeans(n_clusters=n_digits, random_state=42, n_init=10)
kmeans_pca.fit(X_pca)
labels_pca = kmeans_pca.labels_

# Evaluate clustering with PCA
ari_pca = adjusted_rand_score(y_true, labels_pca)
silhouette_pca = silhouette_score(X_pca, labels_pca)

print(f"K-Means (PCA Reduced Data) - Adjusted Rand Index (ARI): {ari_pca:.3f}")
print(f"K-Means (PCA Reduced Data) - Silhouette Score: {silhouette_pca:.3f}")
print("-" * 30)

# --- 5. Compare Clustering Results ---
print("\n--- Comparison of Clustering Results ---")
print(f"ARI (Original Data): {ari_original:.3f}")
print(f"ARI (PCA Reduced Data): {ari_pca:.3f}")
print(f"Silhouette Score (Original Data): {silhouette_original:.3f}")
print(f"Silhouette Score (PCA Reduced Data): {silhouette_pca:.3f}")

# --- Optional: Visualize K-Means clusters on 2D PCA for reduced data ---
# (Using the 2-component PCA done earlier for visualization)
kmeans_pca_2d = KMeans(n_clusters=n_digits, random_state=42, n_init=10)
kmeans_pca_2d.fit(X_pca_2d)
labels_pca_2d = kmeans_pca_2d.labels_

plt.figure(figsize=(10, 8))
scatter_kmeans = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_pca_2d, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_pca_2d.cluster_centers_[:, 0], kmeans_pca_2d.cluster_centers_[:, 1],
            marker='X', s=200, color='red', label='Cluster Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters on PCA-Reduced Digits (2 Components)')
plt.colorbar(scatter_kmeans, label='Cluster Label')
plt.legend()
plt.grid(True)
plt.show()

# Additional comparison: Cluster purity (simple mapping to true labels)
# This isn't a formal metric but can give a sense of how well clusters align with true labels.
from scipy.stats import mode

def purity_score(y_true, y_pred):
    # compute contingency matrix (number of occurrences of each cluster-true label pair)
    contingency_matrix = np.zeros((np.unique(y_true).shape[0], np.unique(y_pred).shape[0]), dtype=int)
    for i, cluster_id in enumerate(np.unique(y_pred)):
        for j, true_label in enumerate(np.unique(y_true)):
            contingency_matrix[j, i] = np.sum((y_pred == cluster_id) & (y_true == true_label))

    # find optimal mapping between clusters and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) # maximize sum of entries
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

from scipy.optimize import linear_sum_assignment

purity_original = purity_score(y_true, labels_original)
purity_pca = purity_score(y_true, labels_pca)

print(f"\nPurity Score (Original Data): {purity_original:.3f}")
print(f"Purity Score (PCA Reduced Data): {purity_pca:.3f}")
