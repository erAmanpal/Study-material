import pandas as pd
import numpy as np
# 📥 Load Titanic dataset
#url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
url="titanic.csv"
df = pd.read_csv(url)

# 🧼 Preprocess
from sklearn.preprocessing import LabelEncoder
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# 🔍 Scale features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# 🔪 Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 📉 Reduce to 2D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 🧠 Train SVM models with different kernels
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

# 🎨 Plot decision boundaries
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
# grabs the first principal component, lowest and highest values
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

for i, kernel in enumerate(kernels):
    Z = models[kernel].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    axes[i].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
    axes[i].set_title(f"SVM ({kernel} kernel)\nAccuracy: {accuracies[kernel]:.2f}")

plt.tight_layout()
plt.show()
