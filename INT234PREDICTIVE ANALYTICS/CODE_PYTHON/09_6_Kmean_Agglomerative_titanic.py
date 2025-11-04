import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
url ='titanic.csv'
data = pd.read_csv(url)
# Step 2: Select & Preprocess Features
df = data[['Pclass', 'Age', 'Fare', 'Sex']].copy()
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
# Encode 'Sex' (male=0, female=1)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# Add labels back
df['Cluster'] = labels

# Scatter plot
# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2,figsize=(14,6))

import seaborn as sns
sns.scatterplot(x=df['Age'], y=df['Fare'], hue=df['Cluster'], palette='Set1', ax=axes[0])
axes[0].set_title("Agglomerative Clustering (Titanic: Age vs Fare)")

# Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode="lastp", p=20, ax=axes[1])
axes[1].set_title("Dendrogram (Agglomerative Clustering)")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Distance")
plt.tight_layout()
plt.show()
