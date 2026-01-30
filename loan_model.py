import pandas as pd

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

