import pandas as pd

data = pd.read_csv("data/loan_data.csv")

print("Dataset loaded successfully!")
print(data.head())
print("Shape:", data.shape)
