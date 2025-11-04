import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("Mall_Customers.csv")
print(data.head())

#Plotting the scatter plot
# Create figure with 3 subplots
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

X=data[['Age']]
y=data['Annual Income (k$)']
axes[0].scatter(X,y);
axes[0].set_title("Age vs Annual Income")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Annual Income (k$)")


from sklearn.cluster import KMeans
# 1. (WCSS):WITHIN CLUSTERS SUM OF SQUARES
cols = ['Age', 'Annual Income (k$)']
X = data[cols]
wcss=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,random_state=42)
    km.fit(X)# 2D-single list of column names.
    wcss.append(km.inertia_)
axes[1].plot(range(1,10),wcss, marker='o')
axes[1].set_title("Elbow Method (WCSS)")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("WCSS")

# 2. Silhouette Score 
from sklearn.metrics import silhouette_score
sil_scores = []
for i in range(2, 10):  # silhouette_score needs at least 2 clusters
    km = KMeans(n_clusters=i, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)
axes[2].plot(range(2,10),sil_scores, marker='o')
axes[2].set_title("Silhouette Score Method")
axes[2].set_xlabel("Number of Clusters")
axes[2].set_ylabel("Silhouette Score")

#Use k predicted form WCSS
km=KMeans(n_clusters=5)
data['Cluster'] = km.fit_predict(X)
#Step 5: Visualization (2D for simplicity)
axes[3].scatter(data['Age'], data['Annual Income (k$)'], c=data['Cluster'], cmap='coolwarm', alpha=0.6)
axes[3].set_xlabel("Age")
axes[3].set_ylabel("Annual Income (k$)")
axes[3].set_title("K-Means Clustering on Titanic (Age vs Fare)")

# Final layout
plt.tight_layout()
plt.show()
