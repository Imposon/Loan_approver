import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("data/loan_data.csv")

print("Dataset loaded successfully!")
print(data.head())
print("Shape:", data.shape)

print("\nMissing values before cleaning:\n", data.isnull().sum())

cat_cols = [
    'Gender', 'Married', 'Dependents',
    'Self_Employed', 'Property_Area'
]

for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

print("\nMissing values after cleaning:\n", data.isnull().sum())
print("Shape after cleaning:", data.shape)


data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop('Loan_Status_Y', axis=1)
y = data_encoded['Loan_Status_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Loan Approval Confusion Matrix")
plt.show()


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)