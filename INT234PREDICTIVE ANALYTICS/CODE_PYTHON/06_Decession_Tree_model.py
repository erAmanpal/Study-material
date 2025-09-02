import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 📥 Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 🧼 Preprocess
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())

# 🎯 Features and target
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Train Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 📊 Evaluate
y_pred = tree.predict(X_test)
print(f"\nModel Accuracy on Test Set: {accuracy_score(y_test, y_pred):.2f}")

# 🎨 Plot tree
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=['Pclass', 'Sex', 'Age'], class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

# 🧠 Command-line prediction
print("\n🔍 Predict Survival")
try:
    pclass = int(input("Enter Pclass (1, 2, or 3): "))
    sex = input("Enter Sex (male/female): ").strip().lower()
    age = float(input("Enter Age: "))

    sex_encoded = 0 if sex == 'male' else 1
    sample = pd.DataFrame([[pclass, sex_encoded, age]], columns=['Pclass', 'Sex', 'Age'])

    prediction = tree.predict(sample)[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    print(f"\n🧾 Prediction: {result}")
except Exception as e:
    print(f"\n⚠️ Error: {e}")
