import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples between 0 and 10
y = 3 * X.squeeze() + 7 + np.random.randn(100) * 2  # squeeze:Linear relation + noise

# 2. Train model on full data (no split)
model_full = LinearRegression()
model_full.fit(X, y)
y_pred_full = model_full.predict(X)

# Compute RMSE manually (no 'squared=False' here)
rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_split = LinearRegression()
model_split.fit(X_train, y_train)
y_pred_split = model_split.predict(X_test)

# Compute RMSE for test predictions
rmse_split = np.sqrt(mean_squared_error(y_test, y_pred_split))

# 4. Plot both scenarios
plt.figure(figsize=(12, 5))

# Plot 1: No split
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.6, label='Data')
plt.plot(X, y_pred_full, color='red', label='Regression Line')
plt.title(f'Without Train/Test Split\nRMSE: {rmse_full:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot 2: With split
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Test Data')
plt.plot(X_test, y_pred_split, color='orange', label='Prediction')
plt.title(f'With Train/Test Split\nRMSE: {rmse_split:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Print RMSE values
print(f"✅ RMSE without split: {rmse_full:.2f}")
print(f"✅ RMSE with split: {rmse_split:.2f}")
