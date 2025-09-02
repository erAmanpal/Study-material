# ============================================================================
# POLYNOMIAL REGRESSION - STUDENT TUTORIAL
# ============================================================================
# Polynomial Regression captures NON-LINEAR relationships by using polynomial terms
# Instead of: y = mx + b (linear)
# We use: y = b0 + b1*x + b2*x² + b3*x³ + ... (polynomial)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# STEP 1: Load and prepare data
print("\nSTEP 1: Loading California Housing Dataset...")
housing = fetch_california_housing()
X_all = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# For polynomial regression, we'll use ONE feature to make it easier to visualize
# Let's use MedInc (Median Income) as it has good correlation with house prices
selected_feature = 'MedInc'
X = X_all[[selected_feature]]

print(f"Selected feature: {selected_feature}")
print(f"Why one feature? Polynomial regression is easier to understand and visualize with 1D input")

# STEP 2: Explore the relationship
print(f"\nSTEP 2: Exploring the relationship...")

# Calculate correlation
correlation = np.corrcoef(X[selected_feature], y)[0,1]
print(f"Linear correlation between {selected_feature} and house price: {correlation:.3f}")

# Check if relationship might be non-linear by looking at data
plt.figure(figsize=(10, 6))
plt.scatter(X[selected_feature], y, alpha=0.5, color='blue')
plt.xlabel(f'{selected_feature} (Median Income)')
plt.ylabel('House Price (hundreds of thousands $)')
plt.title('Data Distribution - Is the relationship linear?')
plt.grid(True, alpha=0.3)
plt.show()

print("Looking at the scatter plot above:")
print("- If points form a straight line → Linear relationship")
print("- If points form a curve → Non-linear relationship (good for polynomial)")

# STEP 3: Split the data
print(f"\nSTEP 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# STEP 4: Compare different polynomial degrees
print(f"\nSTEP 4: Testing different polynomial degrees...")
print("We'll compare linear (degree 1) vs polynomial (degree 2, 3) models")

degrees_to_test = [1, 2, 3]  # Linear, Quadratic, Cubic
results = {}

for degree in degrees_to_test:
    print(f"\n--- Testing Polynomial Degree {degree} ---")
    
    if degree == 1:
        print("Degree 1 = Linear Regression: y = b0 + b1*x")
    elif degree == 2:
        print("Degree 2 = Quadratic: y = b0 + b1*x + b2*x²")
    elif degree == 3:
        print("Degree 3 = Cubic: y = b0 + b1*x + b2*x² + b3*x³")
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Create a pipeline that first creates polynomial features, then applies linear regression
    model = Pipeline([
        ('polynomial_features', poly_features),
        ('linear_regression', LinearRegression())
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[degree] = {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': model.named_steps['linear_regression'].coef_,
        'intercept': model.named_steps['linear_regression'].intercept_
    }
    
    print(f"✓ MSE: {mse:.4f}")
    print(f"✓ R² Score: {r2:.4f} ({r2:.1%} variance explained)")
    
    # Show the equation
    coeffs = results[degree]['coefficients']
    intercept = results[degree]['intercept']
    
    if degree == 1:
        print(f"✓ Equation: y = {intercept:.4f} + {coeffs[0]:.4f}*x")
    elif degree == 2:
        print(f"✓ Equation: y = {intercept:.4f} + {coeffs[0]:.4f}*x + {coeffs[1]:.4f}*x²")
    elif degree == 3:
        print(f"✓ Equation: y = {intercept:.4f} + {coeffs[0]:.4f}*x + {coeffs[1]:.4f}*x² + {coeffs[2]:.4f}*x³")

# STEP 5: Compare performance
print(f"\nSTEP 5: Performance Comparison...")
print(f"{'Degree':<8} {'Type':<12} {'MSE':<10} {'R² Score':<10} {'Performance'}")
print("-" * 55)

best_r2 = 0
best_degree = 1

for degree in degrees_to_test:
    model_type = "Linear" if degree == 1 else f"Polynomial"
    mse = results[degree]['mse']
    r2 = results[degree]['r2']
    
    if r2 > 0.7:
        performance = "Excellent"
    elif r2 > 0.5:
        performance = "Good"
    elif r2 > 0.3:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f"{degree:<8} {model_type:<12} {mse:<10.4f} {r2:<10.4f} {performance}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree

print(f"\n✓ Best performing model: Degree {best_degree} (R² = {best_r2:.4f})")

# STEP 6: Visualize the results
print(f"\nSTEP 6: Creating visualizations...")

# Create subplot for each degree
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, degree in enumerate(degrees_to_test):
    ax = axes[i]
    
    # Plot original data points
    ax.scatter(X_test, y_test, alpha=0.5, color='blue', label='Actual Data', s=30)
    
    # Create smooth curve for the fitted line
    X_range = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
    X_range_df = pd.DataFrame(X_range, columns=[selected_feature])
    y_range_pred = results[degree]['model'].predict(X_range_df)
    
    # Plot the fitted curve
    ax.plot(X_range, y_range_pred, color='red', linewidth=2, 
            label=f'Degree {degree} Fit')
    
    # Formatting
    ax.set_xlabel(f'{selected_feature} (Median Income)')
    ax.set_ylabel('House Price')
    ax.set_title(f'Polynomial Degree {degree}\nR² = {results[degree]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STEP 7: Understanding overfitting vs underfitting
print(f"\nSTEP 7: Understanding Model Complexity...")

print(f"""
MODEL COMPLEXITY ANALYSIS:

1. DEGREE 1 (Linear): R² = {results[1]['r2']:.4f}
   - Simplest model
   - May be UNDERFITTING if relationship is actually curved
   - Good baseline, interpretable

2. DEGREE 2 (Quadratic): R² = {results[2]['r2']:.4f}
   - Can capture curved relationships
   - Good balance between complexity and performance
   - Still relatively interpretable

3. DEGREE 3 (Cubic): R² = {results[3]['r2']:.4f}
   - More complex, can fit more intricate curves
   - Risk of OVERFITTING if degree is too high
   - Less interpretable
""")

# Show the risk of overfitting with higher degrees
print(f"\nSTEP 8: Demonstrating overfitting risk...")
high_degrees = [1, 2, 5, 10]  # Test very high degrees
overfitting_results = {}

for degree in high_degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    model = Pipeline([
        ('polynomial_features', poly_features),
        ('linear_regression', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Calculate performance on BOTH training and testing sets
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    overfitting_results[degree] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': train_r2 - test_r2
    }

print(f"Overfitting Analysis (Training vs Test Performance):")
print(f"{'Degree':<8} {'Train R²':<10} {'Test R²':<10} {'Gap':<10} {'Status'}")
print("-" * 50)

for degree in high_degrees:
    train_r2 = overfitting_results[degree]['train_r2']
    test_r2 = overfitting_results[degree]['test_r2']
    gap = overfitting_results[degree]['gap']
    
    if gap > 0.1:
        status = "OVERFITTING!"
    elif gap > 0.05:
        status = "Slight overfit"
    else:
        status = "Good fit"
    print(f"{degree:<8} {train_r2:<10.4f} {test_r2:<10.4f} {gap:<10.4f} {status}")
