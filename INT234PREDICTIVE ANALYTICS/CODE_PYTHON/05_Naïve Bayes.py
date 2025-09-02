import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load Titanic dataset
#url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
url="titanic.csv"
data = pd.read_csv(url)
# Select features
df = data[['Survived','Pclass','Sex','Age','Fare']].copy()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test,y_pred))

# Try a new passenger
print("Prediction for 'Pclass','Sex','Age','Fare'")
new_passenger = pd.DataFrame([[1,1,25,100]], columns=['Pclass','Sex','Age','Fare'])
prediction = model.predict(new_passenger)[0]
# [0] caz. it returns the result inside an array: Output -> [1] 
prob = model.predict_proba(new_passenger)[0]
# Eg prob = [0.25, 0.75]
print(" model classes" )
print(model.classes_)

print("Prediction for [1,1,25,100]:", "Survived " if prediction==1 else "Not Survived ")
print("Probabilities -> Died:", round(prob[0],2), ", Survived:", round(prob[1],2))

# Confusion matrix
cm = confusion_matrix(y_test,y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Purples")
plt.title("Naïve Bayes Confusion Matrix (Titanic)")
plt.show()

