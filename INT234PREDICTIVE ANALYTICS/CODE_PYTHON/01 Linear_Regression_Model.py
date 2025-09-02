# Simple Linear Regression with Real Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load California Housing dataset
'''
:Attribute Information:
    - MedInc        median income in block group 
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude
'''
data = fetch_california_housing()
X = data.data[:, 0].reshape(-1, 1)  # Use only median income feature
#data[:,0] - > All the first column of the dataset
print(X)
# The target stored in data.target, not as a DataFrame column
# The sklearn dataset structure separates features (X) from target (y)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
# test_size=0.2: 20% of the data goes to the test set.
# random_state=42: ensures the same random split every time you run the code.
If not implemented:
Different model performance
Harder debugging
Inconsistent results across experiments
'''

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

# Original data and regression line
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, alpha=0.2, label='Actual')
plt.plot(X_test, y_pred, 'r.', alpha=0.2, label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.legend()

# Take user input for prediction
print("\nMake a prediction:")
income = float(input("Enter median income (0-15): "))
prediction = model.predict([[income]])
print(f"Predicted house price: ${prediction[0]:.2f} hundred thousand")
#plt.plot(income, prediction, 'rx', alpha=0.5, label='Predicted',s=100)
plt.scatter(income, prediction, 
            color='black',       # or use 'c='
            marker='s',        # same as 'rx' in plot()
            #alpha=0.5, 
            label='Predicted', 
            s=100)             # size of each dot


# Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# Draws a diagonal reference line. (bottom-left to the top-right)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()

'''
Common Marker Symbols:
Basic shapes:
'o' for circles
's' for squares
'^' for upward-pointing triangles
'v' for downward-pointing triangles
'<' for left-pointing triangles
'>' for right-pointing triangles
'd' for diamonds
'h' for hexagons
Geometric points/lines:
'.' for points
'+' for plus signs
'x' for 'x' marks
'''
