import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier# Tiny dataset
X = np.array([[2],[4],[6],[8],[10]])  # Hours studied
y = np.array([0,0,1,1,1])       # 0=Fail, 1=Pass# Train KNN
#Use the KNN classifier model with N=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)# Predict for new student
# Now use the new data for the prediction
new_student = np.array([[5]])
prediction = model.predict(new_student)

print("Prediction for 5 hours:", "Pass" if prediction[0]==1 else "Fail")
# Plot dataset + new point
plt.scatter(X, y, color="blue", label="Training Points")
plt.scatter(new_student, prediction, color="red", s=100, label="New Student (5 hrs)")
plt.yticks([0,1], ["Fail","Pass"])
plt.xlabel("Study Hours")
plt.ylabel("Result")
#plt.legend()
plt.title("KNN Example: Study Hours vs Result")
plt.show()
