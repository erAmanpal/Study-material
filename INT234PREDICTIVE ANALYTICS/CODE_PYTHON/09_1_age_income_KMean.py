import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("Mall_Customers.csv")
print(data.head())

#Plotting the scatter plot
# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

X=data[['Age']]
y=data['Annual Income (k$)']
axes[0].scatter(X,y);
axes[0].set_title("Age vs Annual Income")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Annual Income (k$)")


from sklearn.cluster import KMeans
# WITHIN CLUSTERS SUM OF SQUARES(WCSS)
cols = ['Age', 'Annual Income (k$)']
X = data[cols]
wcss=[]
for i in range(1,10):
    km=KMeans(n_clusters=i)
    km.fit(X)# 2D-single list of column names.
    wcss.append(km.inertia_)
axes[1].plot(range(1,10),wcss, marker='o')
axes[1].set_title("Elbow Method (WCSS)")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("WCSS")

#use k predicted form WCSS
km=KMeans(n_clusters=5)
data['Cluster'] = km.fit_predict(X)
#Step 5: Visualization (2D for simplicity)
axes[2].scatter(data['Age'], data['Annual Income (k$)'], c=data['Cluster'], cmap='coolwarm', alpha=0.6)
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Annual Income (k$)")
axes[2].set_title("K-Means Clustering on Titanic (Age vs Fare)")

# Final layout
plt.tight_layout()
plt.show()
