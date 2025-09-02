# SVM classifier
import numpy as np
# 🎲 Step 1: Create synthetic fruit data
# Feature 1: Size, Feature 2: Color (1 = red, 0 = orange)
X = np.array([[3, 1], [4, 1], [2, 0], [1, 0], [3.5, 1], [1.5, 0], [2,2]])
y_class = np.array([1, 1, -1, -1, 1, -1,-1])       # SVM labels: Apple (+1), Orange (-1)

# 🔍 Step 2: Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Transformed data with a mean of 0 and a standard deviation of 1.
X_scaled = scaler.fit_transform(X)  

# 🔪 Step 3: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_class_train, y_class_test = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)

# 🧠 Step 4: Train models
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_class_train)

# 🎨 Step 5: Plotting
import matplotlib.pyplot as plt
#fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SVM Plot
plt.suptitle("SVM Classification (Apple vs Orange)")
# first feature,second feature, color by class label
# The labels (y_class) are not coordinates,
# they’re used to color the points to show which class they belong to.
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_class)

# Decision boundary
# 𝑤0 𝑥 + 𝑤1 𝑦 + 𝑏=0
w = svm_model.coef_[0]      #w is the weight vector [w0, w1]
b = svm_model.intercept_[0] #b is the bias/intercept term
# This creates 100 evenly spaced x-values across the range of the first feature.
x_vals = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
# This is the key step: solving the decision boundary equation for y.
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, label='Decision Boundary')
plt.legend()

plt.show()
