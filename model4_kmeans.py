import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import  silhouette_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import timeit
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Read the dataset
df = pd.read_csv(r'data/dermatology.csv', delimiter='\t')
df.replace('?', np.nan, inplace=True)
df = df.astype(float)

#  missing values with mean
df.fillna(df.mean(), inplace=True)

# Selecting relevant attributes for clustering
attributes = df.iloc[:, :-2]  

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(attributes)

# Choose the number of clusters (K)
num_clusters = 6  #  6 disease types
# Apply K-means clustering algorithm
#//code adapted from Babitz,2023
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X_normalized)
#//end of adapted code

# Get cluster labels
cluster_labels = kmeans.labels_

# Visualize the data in the original feature space
plt.figure(figsize=(12, 8))
for i in range(attributes.shape[1]):
    plt.subplot(4, 9, i + 1)  # Adjust the subplot layout according to the number of features
    plt.hist(attributes.iloc[:, i].dropna(),
    bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(attributes.columns[i])
    plt.ylabel('Frequency')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize clusters and centroids
plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Colors for each cluster
markers = ['o', 's', 'D', '^', 'v', 'P']  # Marker styles for centroids
for cluster in range(num_clusters):
    cluster_points = X_normalized[
    cluster_labels == cluster]
    num_points = min(200, len(cluster_points))  # Maximum of 200 points per cluster
    np.random.seed(42)  # Set seed for reproducibility
    random_indices = np.random.choice(len(
    cluster_points), num_points, replace=False)
    plt.scatter(cluster_points[random_indices, 0],
    cluster_points[random_indices, 1],
    c=colors[cluster], label=f'Cluster {cluster+1}', alpha=0.7)
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1],
    marker=markers[i], color='black', s=200, label=f'Centroid {i+1}')
plt.title('Clustering Results')
plt.legend()
plt.grid(True)
plt.show()


# Visualize clusters in 2D using matplotlib
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
plt.figure(figsize=(10, 8))
for cluster in range(num_clusters):
    cluster_points = X_pca[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points
    [:, 1], label=f'Cluster {cluster+1}')
plt.title('Clustering Results')
plt.legend()
plt.grid(True)
plt.show()


# Calculate confusion matrix
conf_matrix = confusion_matrix(df.iloc[:, -1].astype(int) - 1, cluster_labels)
print("confusion matrix=\n", conf_matrix)

# Visualize the clusters and confusion matrix
# Calculate confusion matrix
true_labels = df.iloc[:, -1].astype(int) - 1  # Assuming the true labels are available
conf_matrix = confusion_matrix(true_labels, cluster_labels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
for cluster in range(num_clusters):
    cluster_points = X_pca[cluster_labels == 
    cluster]
    ax1.scatter(cluster_points[:, 0],
    cluster_points[:, 1], label=f'Cluster {cluster+1}')
ax1.set_title('Clustering Results')
ax1.legend()
ax1.grid(True)

# Calculate and visualize confusion matrix
conf_matrix = confusion_matrix(df.iloc
[:, -1].astype(int) - 1, cluster_labels)
sns.heatmap(conf_matrix, annot=True, fmt='d',
cmap='Blues', ax=ax2)
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
plt.tight_layout()
plt.show()

# Evaluation metrics
accuracy = np.sum(conf_matrix.max(axis=1)) / np.sum(
    conf_matrix)
print("Accuracy of the K-means clustering model =",accuracy)
# Calculate Silhouette
silhouette_avg = silhouette_score(X_normalized, 
    cluster_labels)
print("Silhouette Score:", silhouette_avg)
# Dunn Index
dunn_index = davies_bouldin_score(X_normalized, cluster_labels)
print("Dunn Index:", dunn_index)
# Calinski-Harabasz Index (Variance Ratio Criterion)
calinski_harabasz_index = calinski_harabasz_score(X_normalized,
    cluster_labels)
print("Calinski-Harabasz Index:", calinski_harabasz_index)


def time_kmeans():
    kmeans.fit(X_normalized)

# Measure execution time
execution_time = timeit.timeit(time_kmeans, number=1)
print(f"Execution Time: {execution_time} seconds")

