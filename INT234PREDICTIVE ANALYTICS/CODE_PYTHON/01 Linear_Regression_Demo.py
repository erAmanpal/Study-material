import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Sample data
x = np.array([1, 2, 3, 4, 5,20]).reshape(-1, 1)  # Independent variable
#y = np.array([2, 4, 5, 4, 5])                # Dependent variable
y = np.array([1, 2, 3, 4, 5,20])                # Dependent variable
# Create and fit the model
model = LinearRegression()
model.fit(x, y)
# Get slope and intercept
m,c = model.coef_[0],model.intercept_
print(f"Equation of line: y = {m:.2f}x + {c:.2f}")
# Predict values:against the original x values.
y_pred = model.predict(x)
xx=np.array([[10]])
yy=model.predict(xx)
# Plot
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Best-fit Line')
plt.plot(xx, yy, color='green', label='Prediction', marker='x')
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
