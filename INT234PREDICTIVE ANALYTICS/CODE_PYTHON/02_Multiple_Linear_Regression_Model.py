# ============================================================================
# MULTIPLE LINEAR REGRESSION - STUDENT TUTORIAL
# ============================================================================
# Multiple Linear Regression models the relationship between:
# - MULTIPLE input variables (X1, X2, X3, ..., Xn)
# - One output variable (y)
# The equation is: y = b0 + b1*X1 + b2*X2 + b3*X3 + ... + bn*Xn


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# STEP 1: Load and explore the data
from sklearn.datasets import fetch_california_housing
print("\nSTEP 1:Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # House prices

# STEP 2: Examine the data
print(f"\nSTEP 2: Data Overview...")
print("First 5 rows of features:")
print(X.head())
print(f"\nFirst 5 target values (house prices): {y[:5]}\n")


# STEP 3: Split the data
from sklearn.model_selection import train_test_split
print(f"\nSTEP 3: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples with {X_train.shape[1]} features")
print(f"Testing set:  {X_test.shape[0]} samples with {X_test.shape[1]} features")

# STEP 4: Create and train the Multiple Linear Regression model
print("\nSTEP 4: training model ...\n")
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Examine the coefficients (learned parameters)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"\nSTEP 5: Model trained successfully!")

# STEP 6: Make predictions
print("\nSTEP 6: Make predictions\n")
'''
	MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude
0	8.3252	41	        6.984126984	1.023809524	322	        2.555555556	37.88	        -122.23

'''
new_House = np.array([[8.3252,41,6.984126984,1.023809524,322,2.555555556,37.88,-122.23]])
new_House_df = pd.DataFrame(new_House, columns=housing.feature_names)
y_pred = model.predict(new_House_df)
print("\nPredicted Final Score:", y_pred[0])
# STEP 7: Evaluate model performance
print(f"\nSTEP 7: Evaluating Model Performance...")
from sklearn.metrics import mean_squared_error, r2_score
# compare the predicted values on the test set with the actual test labels
y_test_pred = model.predict(X_test)  # Predict on test set
# Compute the values
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print(f"Performance Metrics:")
print(f"  - Mean Squared Error (MSE): {mse:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  - R² Score: {r2:.4f} ({r2:.1%})")

# Interpret the results
if r2 > 0.8:
    performance = "Excellent"
elif r2 > 0.6:
    performance = "Good"
elif r2 > 0.4:
    performance = "Moderate"
else:
    performance = "Poor"

print(f"  - Model Performance: {performance}")
print(f"  - The model explains {r2:.1%} of the variance in house prices")

# STEP 8: Identify the most influential feature
print("\nSTEP 8: Identifying Most Influential Feature...")

# Create a DataFrame of features and their coefficients
coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
})

# Add absolute value for ranking
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()

# Sort by absolute coefficient
coef_df_sorted = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Display ranked features
print(coef_df_sorted)

# Most influential feature
most_influential = coef_df_sorted.iloc[0]
print(f"\n✅ Most Influential Feature: {most_influential['Feature']} (Coefficient: {most_influential['Coefficient']:.4f})")

'''
KEY CONCEPTS:

1. THE EQUATION:
   y = {intercept:.2f} + {coefficients[0]:.2f}*MedInc + {coefficients[1]:.2f}*HouseAge + ... 
   
2. MOST INFLUENTIAL FEATURE: {most_important['Feature']}
   - Coefficient: {most_important['Coefficient']:.4f}
   - Impact: Each unit change affects price by ${most_important['Coefficient']*100:.0f}k
   
3. LEAST INFLUENTIAL FEATURE: {least_important['Feature']}
   - Coefficient: {least_important['Coefficient']:.4f}
   - Impact: Each unit change affects price by ${least_important['Coefficient']*100:.0f}k

4. MODEL PERFORMANCE:
   - R² = {r2:.4f} means {r2:.1%} of price variation is explained
   - RMSE = ${rmse*100:.0f}k average prediction error

5. ADVANTAGES OF MULTIPLE LINEAR REGRESSION:
   - Can handle multiple input features
   - Provides interpretable coefficients
   - Fast training and prediction
   - Good baseline model

6. ASSUMPTIONS:
   - Linear relationship between features and target
   - Features are not highly correlated with each other
   - Residuals are normally distributed
   - Constant variance (homoscedasticity)
""")

print(f"✓ Multiple Linear Regression Tutorial Complete!")
print(f"✓ You successfully used {X.shape[1]} features to predict house prices!")
'''
