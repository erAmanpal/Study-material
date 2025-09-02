import pandas as pd
#🧪 Step 1: Prepare the Data
# Sample dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

#🔄 Step 2: Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder
# Encode features
le_outlook = LabelEncoder()
le_wind = LabelEncoder()
le_target = LabelEncoder()

data['Outlook_enc'] = le_outlook.fit_transform(data['Outlook'])
data['Wind_enc'] = le_wind.fit_transform(data['Wind'])
data['Target'] = le_target.fit_transform(data['PlayTennis'])

# Features and target
X = data[['Outlook_enc', 'Wind_enc']]
y = data['Target']

# 🌳 Step 3: Train the Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Create and train model
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# 🎨 Step 4: Visualize the Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plot_tree(clf, 
          feature_names=['Outlook', 'Wind'], 
          class_names=le_target.classes_, 
          filled=True)
plt.show()

# 🔍 Step 5: Make Predictions
# Predict for a new sample: Outlook=Rain, Wind=Weak
sample = [[le_outlook.transform(['Rain'])[0], le_wind.transform(['Weak'])[0]]]
sample = pd.DataFrame(sample, columns=['Outlook_enc', 'Wind_enc'])
pred = clf.predict(sample)
print("Prediction:", le_target.inverse_transform(pred)[0])

