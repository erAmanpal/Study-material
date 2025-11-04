# 📚 Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 🧭 Step 1: Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 🧼 Step 2: Select relevant features
# We'll use numerical + encoded categorical features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[features + ['Survived']]

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Define feature matrix (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# ⚖️ Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🎯 Step 4: Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Survived'] = y.values

# 📊 Step 5: Visualize the PCA results
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Survived',
    palette=['red', 'green'],
    alpha=0.7
)
plt.title('PCA of Titanic Dataset')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
plt.show()

# 📈 Step 6: Check explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
