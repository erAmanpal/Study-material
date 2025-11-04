import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv("Mall_Customers.csv")
X = data[['Age', 'Annual Income (k$)']]

# Start with all points in one cluster
labels = [0] * len(X)  # Everyone starts in cluster 0
clusters = {0: X.copy()}  # Dictionary to hold cluster data
next_label = 1  # Label for the next new cluster

# Keep splitting until we have 4 clusters
while len(clusters) < 4:
    # Find the largest cluster to split
    largest = max(clusters, key=lambda k: len(clusters[k]))
    data_to_split = clusters.pop(largest)

    # Apply KMeans with 2 clusters
    km = KMeans(n_clusters=2, random_state=42)
    split = km.fit_predict(data_to_split)

    # Assign new labels
    idx = data_to_split.index
    for i, point in enumerate(idx):
        if split[i] == 0:
            labels[point] = largest  # Keep old label
        else:
            labels[point] = next_label  # Assign new label

    # Save new clusters
    clusters[largest] = data_to_split[split == 0]
    clusters[next_label] = data_to_split[split == 1]
    next_label += 1

# Add labels to data
data['DivCluster'] = labels

# Plot result
# Define your custom colors (must match number of clusters)
from matplotlib.colors import ListedColormap
custom_colors = ListedColormap(['red', 'blue', 'green',  'purple'])
plt.figure(figsize=(8, 6))
plt.scatter(data['Age'], data['Annual Income (k$)'], c=data['DivCluster'], cmap=custom_colors, alpha=0.6)
plt.title("Simulated Divisive Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.show()
