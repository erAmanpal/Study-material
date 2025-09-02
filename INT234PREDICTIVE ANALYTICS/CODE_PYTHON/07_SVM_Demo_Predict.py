import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 🧼 Sample data: 2D points
X = np.array([[2, 3], [3, 4], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Blue, 1 = Red

# 🌟 Train SVM  # kernel='rbf' for curved boundaries
model = svm.SVC(kernel='linear')  # linear,Try 'rbf' or 'poly' for nonlinear
model.fit(X, y)

# 🔮 Predict a new point
new_point = np.array([[5, 4]])
pred = model.predict(new_point)
print("Prediction:", "Red" if pred[0] == 1 else "Blue")

# 🎨 Visualize
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.scatter(new_point[:, 0], new_point[:, 1], c='green', marker='x', s=100)
plt.title("SVM Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
