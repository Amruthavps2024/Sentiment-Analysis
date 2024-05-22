import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA


# Read the dataset
df = pd.read_csv(r'data/dermatology.csv', delimiter='\t')
df.replace('?', np.nan, inplace=True)
df = df.astype(float)

# handling missing values with mean
df.fillna(df.mean(), inplace=True)

# Selecting relevant attributes for clustering
attributes = df.iloc[:, :33]  # Include all 33 attributes for clustering

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(attributes)

#//code adapted from Anakin,2021
# Estimate bandwidth
bandwidth = estimate_bandwidth(X_normalized, quantile=0.2)
# Apply Mean Shift clustering algorithm with optimized bandwidth
clustering = MeanShift(bandwidth=bandwidth)
cluster_labels = clustering.fit_predict(X_normalized)
#//end of adapted code

# Print cluster analysis and cluster coordinates
for cluster in np.unique(cluster_labels):
    cluster_points = X_normalized[cluster_labels == cluster]
    cluster_centroid = cluster_points.mean(axis=0)
    print(f"Cluster {cluster+1}:")
    print(f"Number of points: {len(cluster_points)}")
    print(f"Centroid coordinates: {cluster_centroid}")
    print()

# Visualize clusters
plt.figure(figsize=(10, 8))
for cluster in np.unique(cluster_labels):
    cluster_points = X_normalized[
    cluster_labels == cluster]
    num_points = min(200, len(cluster_points))
    np.random.seed(42)  # Set seed for reproducibility
    random_indices = np.random.choice(len(cluster_points),
    num_points, replace=False)
    plt.scatter(cluster_points[random_indices, 0], 
    cluster_points[random_indices, 1], label=
    f'Cluster {cluster+1}', alpha=0.7)
    plt.title('Mean Shift Clustering')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# visualization of clusters using  PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
plt.figure(figsize=(10, 8))
for cluster in np.unique(cluster_labels):
    cluster_points = X_pca[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points
    [:, 1], label=f'Cluster {cluster+1}', alpha=0.7)
plt.title('Mean Shift Clustering (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# Calculate confusion matrix
conf_matrix = confusion_matrix(df.iloc[:, -1]
.astype(int) - 1, cluster_labels)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate accuracy of the model
accuracy = accuracy_score(df.iloc[:, -1].astype(int) 
- 1, cluster_labels)
print("Accuracy of the model:", accuracy)
# Calculate silhouette score
silhouette_avg = silhouette_score(X_normalized,
cluster_labels)
print("Silhouette Score:", silhouette_avg)
# Dunn Index
dunn_index = davies_bouldin_score(X_normalized,
cluster_labels)
print("Dunn Index:", dunn_index)
# Calinski-Harabasz Index (Variance Ratio Criterion)
calinski_harabasz_index = calinski_harabasz_score(
X_normalized, cluster_labels)
print("Calinski-Harabasz Index:", calinski_harabasz_index)
# Measure execution time
def time_meanshift():
    clustering.fit(X_normalized)
execution_time = timeit.timeit(time_meanshift, number=1)
print(f"Execution Time: {execution_time} seconds")

