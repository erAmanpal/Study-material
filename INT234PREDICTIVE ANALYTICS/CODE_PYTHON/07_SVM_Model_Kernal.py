import pandas as pd
import numpy as np
# 📥 Step 1: Load and Preprocess Titanic Data
#url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
url="titanic.csv"
df = pd.read_csv(url)

# 🧼 Preprocess
from sklearn.preprocessing import LabelEncoder
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
# Step 2. Feature Matrix and Target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# 🔍Step 3. Standardize Features: Scale features
# This is crucial for SVM performance.
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# 🔪Step 4. Train-Test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 📉 Step 5. PCA to Reduce 4D to 2D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 🧠 Step 6. Train SVM Models with Different Kernels
kernels = ['linear', 'poly', 'rbf','sigmoid']
models = {}
predictions = {}
accuracies = {}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
for kernel in kernels:
    if kernel == 'poly':
        model = SVC(kernel=kernel, degree=3)
    elif kernel == 'sigmoid':
        model = SVC(kernel=kernel, gamma=0.1, coef0=0.5)
    else:
        model = SVC(kernel=kernel)

    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    #Stores the values wrt the dictionary keys
    models[kernel] = model
    predictions[kernel] = y_pred
    accuracies[kernel] = accuracy_score(y_test, y_pred)

# 🎨 Step 7. Plot Decision Boundaries
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
# grabs the first principal component, lowest and highest values
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

for i, kernel in enumerate(kernels):
    # SVM model to 'predict' the class (0 or 1) for each grid point.
    # a grid of predicted class labels over the entire feature space.
    Z = models[kernel].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Plot Decision Boundary
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    # Plot Test Points
    axes[i].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
    axes[i].set_title(f"SVM ({kernel} kernel)\nAccuracy: {accuracies[kernel]:.2f}")

plt.tight_layout()
plt.show()
