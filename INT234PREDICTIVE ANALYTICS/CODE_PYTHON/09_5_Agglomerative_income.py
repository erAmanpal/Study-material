import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv("Mall_Customers.csv")
X = data[['Age', 'Annual Income (k$)']]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
# 1️⃣ Dendrogram to choose number of clusters
linked = linkage(X, method='ward')
dendrogram(linked, ax=axes[0], orientation='top', distance_sort='descending', show_leaf_counts=False)
axes[0].set_title("Dendrogram for Agglomerative Clustering")
axes[0].set_xlabel("Customers")
axes[0].set_ylabel("Euclidean Distance")

# 2️⃣ Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
data['AggCluster'] = agg.fit_predict(X)

# 3️⃣ Visualize Clusters
axes[1].scatter(data['Age'], data['Annual Income (k$)'], c=data['AggCluster'], cmap='Set1', alpha=0.6)
axes[1].set_title("Agglomerative Clustering (Age vs Income)")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Annual Income (k$)")

# Final layout
plt.tight_layout()
plt.show()
