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

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ = WCSS

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()
