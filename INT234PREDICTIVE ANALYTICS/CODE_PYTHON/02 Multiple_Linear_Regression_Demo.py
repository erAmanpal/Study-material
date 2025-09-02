import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data (5 students, 3 features)
X = np.array([
    [5, 90, 80],   # hours, attendance, previous score
    [3, 85, 70],
    [8, 95, 88],
    [2, 80, 65],
    [6, 92, 85]
])

y = np.array([82, 75, 90, 70, 88])  # Final scores

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict for a new student
new_student = np.array([[4, 88, 78]])
predicted_score = model.predict(new_student)
print("Predicted Final Score:", predicted_score[0])

# Plot --
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use only first two features for 3D plot
X_2D = X[:, :2] #select a subset of features for modeling/visualization
model.fit(X_2D, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_2D[:, 0], X_2D[:, 1], y, color='blue', label='Actual')

# Create meshgrid for surface
x_surf, y_surf = np.meshgrid(
    np.linspace(X_2D[:,0].min(), X_2D[:,0].max(), 10),
    np.linspace(X_2D[:,1].min(), X_2D[:,1].max(), 10)
)
z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
# Plot the prediction point
ax.scatter(new_student[0, 0], new_student[0, 1], predicted_score,
           color='green', s=100, marker='^', label='Predicted Point')

# Plot regression surface
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Attendance')
ax.set_zlabel('Final Score')
plt.title('Multiple Linear Regression (2 Features)')
ax.legend()
plt.show()



'''
marker='x'
Specifies the shape of the point.
'x' - Cross.
'o' - circle
'^' - triangle
's' - square
'D' - diamond
'''
