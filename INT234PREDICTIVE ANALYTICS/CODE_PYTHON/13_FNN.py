import pandas as pd
# Step 1: Load Titanic dataset
url = "titanic.csv"
df = pd.read_csv(url)

# Step 2: Select and preprocess features
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
df.dropna(inplace=True)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 3: Split data
from sklearn.model_selection import train_test_split
X = df.drop('Survived', axis=1) # separating the features
y = df['Survived']              # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   # mean = 0 and std = 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Feedforward Neural Network
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Query with New passenger data(must match feature order and encoding)
new_passenger = pd.DataFrame([[1, 1, 29, 100, 1]], columns=X.columns)

# Scale it using the same scaler
new_passenger_scaled = scaler.transform(new_passenger)


# Step 8: Predict probability of survival
survival_proba = model.predict_proba(new_passenger_scaled)[0][1]  # Probability of class 1 (survived)

# Predict binary outcome
survival_pred = model.predict(new_passenger_scaled)[0]

print(f"Survival Probability: {survival_proba:.2f}")
print("Prediction:", "Survived" if survival_pred == 1 else "Did Not Survive")

