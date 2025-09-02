import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Load Titanic dataset
#url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
url="titanic.csv"
data = pd.read_csv(url)
# Select important features
# ensure that df is an independent copy of the selected columns, rather than a view
df = data[['Survived','Pclass','Sex','Age','Fare']].copy()

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
# Encode Sex: male=0, female=1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# df show top 10 records
print(df.head())
# Features and target
X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']
# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# --------USER INPUT-------------------
print("Enter passenger details:")
pclass = int(input("Pclass (1=Upper, 2=Middle, 3=Lower): "))
sex = int(input("Sex (0=Male, 1=Female): "))
age = float(input("Age: "))
fare = float(input("Fare: "))
new_passenger = [[pclass, sex, age, fare]]

# Keep feature names
new_passenger = pd.DataFrame([[pclass, sex, age, fare]], columns=['Pclass','Sex','Age','Fare'])
# Prediction
prediction = model.predict(new_passenger)[0]
probability = model.predict_proba(new_passenger)[0][1] # survival probability
# Output
if prediction == 1:
  print(f"Prediction: Survived (Probability: {probability:.2f})")
else:
  print(f"Prediction: Did NOT Survive (Probability of survival: {probability:.2f})")
# -------PLOT---------------
plt.figure(figsize=(10,6))
# Plot training data
plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap="coolwarm", edgecolor="k", alpha=0.6, label="Passengers")
# Plot new passenger as a RED STAR
plt.scatter(age, fare, marker='*', c='red', s=200, label="New Passenger (Input)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Titanic Survival with KNN (Age vs Fare)")
plt.legend()
plt.show()


