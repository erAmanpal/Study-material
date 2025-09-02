import numpy as np
import matplotlib.pyplot as plt
#Let randomly create some data’s for two variables with nonlinear trend and randomness.
# Normal Distribution:
# (Mean (loc): 0, Standard deviation (scale): 1, Size: 20 (number of samples))
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
x= []
y= []
print(x)
print(y)

fig = plt.figure(figsize=(12, 4)) # Create a figure, # Optional: set figure size in inches
# Create 3 subplots in 1 row, 3 columns
ax1 = fig.add_subplot(1, 3, 1)  # row=1, col=3, index=1
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
#Visualize the variables spreads for better understanding
ax1.scatter(x,y, s=10)
ax1.set_title('variables spreads')
#plt.show()
#==analyze random data using Regression Analysis
from sklearn.linear_model import LinearRegression
    # Reshapes x and y to 2D arrays (required by scikit-learn).
x = x[:, np.newaxis]
y = y[:, np.newaxis]
model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)
ax2.scatter(x, y, s=10)
ax2.plot(x, y_pred, color='r')
#plt.show()


import sklearn.metrics as metrics
mse = metrics.mean_squared_error(x,y)
rmse = np.sqrt(mse) 
r2 = metrics.r2_score(x,y)

print(f"RMSE value is {rmse:.2f}")   #93.47
print('R2 value: {r2:.2f}')          #-786.23

# Enhance the model’s complexity to overcome under-fitting degree=2
from sklearn.preprocessing import PolynomialFeatures
polynomial_features1 = PolynomialFeatures(degree=2)
x_poly1 = polynomial_features1.fit_transform(x)
model1 = LinearRegression()
model1.fit(x_poly1, y)
y_poly_pred1 = model1.predict(x_poly1)

from sklearn.metrics import mean_squared_error, r2_score
rmse1 = np.sqrt(mean_squared_error(y,y_poly_pred1))
r21 = r2_score(y,y_poly_pred1)
print(f"RMSE value = {rmse1:.2f}")    #49.66
print(f"R2 value   = {r21:.2f}")       #0.73

#plt.plot(x_poly1, y_poly_pred1, color='m')
# Sort x for smooth plotting
x_sorted = np.sort(x, axis=0)
x_poly_sorted = polynomial_features1.transform(x_sorted)
y_poly_pred_sorted = model1.predict(x_poly_sorted)

ax3.scatter(x, y, s=10)
ax3.plot(x_sorted, y_poly_pred_sorted, color='m')

plt.tight_layout()  # Avoid overlapping titles
plt.show()
