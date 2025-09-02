import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Dataset
data = pd.DataFrame({
    'Outlook': ['Rainy','Sunny','Overcast','Overcast','Sunny',
                'Rainy','Sunny','Overcast','Rainy','Sunny',
                'Sunny','Rainy','Overcast','Overcast'],
    'Play':    ['Yes','Yes','Yes','Yes','No',
                'Yes','Yes','Yes','No','No',
                'Yes','No','Yes','Yes']
})

# Encode categorical data
le_outlook = LabelEncoder()
le_play = LabelEncoder()

data['Outlook_enc'] = le_outlook.fit_transform(data['Outlook'])
data['Play_enc'] = le_play.fit_transform(data['Play'])

X = data[['Outlook_enc']]
y = data['Play_enc']

# Train Naive Bayes
model = CategoricalNB()
model.fit(X, y)

# Predict for Sunny
new_outlook = pd.DataFrame({'Outlook_enc':[le_outlook.transform(['Sunny'])]})
pred = model.predict(new_outlook)[0]

print("Prediction for Outlook=Sunny:", le_play.inverse_transform([pred]))
