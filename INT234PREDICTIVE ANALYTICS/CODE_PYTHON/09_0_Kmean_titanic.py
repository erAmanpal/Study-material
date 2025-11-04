'''
The KMeans algorithm groups passengers into 2 clusters based on patterns in:
 age, class, fare, and sex.
The crosstab shows how these clusters align with actual survival labels.
e.g., Cluster 0 might mostly be survivors, Cluster 1 mostly non-survivors.
The scatter plot shows cluster boundaries in Age vs Fare space.
'''
# Step 1: Import Libraries & Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load Titanic dataset
#url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
url ='titanic.csv'
data = pd.read_csv(url)
print(data.head())

# Step 2: Select & Preprocess Features
df = data[['Pclass', 'Age', 'Fare', 'Sex']].copy()
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
# Encode 'Sex' (male=0, female=1)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Step 3: Apply KMeans clustering
#kmeans = KMeans(n_clusters=2,init='k-means++',  random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
print('\n',df['Cluster'].value_counts())
print("\nCluster center:Two Cluster with 4 features:")
# This will give the average values of each feature for each cluster.
# 👉 From this, you can interpret which features separate the groups.
print(kmeans.cluster_centers_)

# Step 4: Compare clusters with Actual Survival
df['Survived'] = data['Survived']
ct = pd.crosstab(df['Cluster'], df['Survived'])
print('\n',ct)
                          
#Step 5: Visualization (2D for simplicity)
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['Fare'], c=df['Cluster'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("K-Means Clustering on Titanic (Age vs Fare)")
plt.show()

# Step 6: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(df['Survived'], df['Cluster'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Survived vs Clusters")
plt.show()

'''
KMeans doesn’t know anything about survival (it’s unsupervised).
It only looks at patterns in the features you provide (Age, Fare, Sex, Pclass).
When you set n_clusters=3, KMeans will try to split the passengers into
3 distinct groups.[3rd class,1st class,medium fare)
'''
