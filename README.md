
=======
# Loan Approval Prediction using Logistic Regression

A Machine Learning project that predicts whether a loan application will be approved based on applicant demographic and financial features.

This project demonstrates a complete ML pipeline including:

- Data Cleaning
- Missing Value Handling
- Categorical Encoding
- Train-Test Splitting
- Logistic Regression Modeling
- Confusion Matrix Visualization
- Model Evaluation

---

## Problem Statement

Financial institutions need to evaluate loan applications efficiently and fairly.

This project builds a classification model that predicts loan approval status based on applicant attributes such as income, credit history, marital status, and property area.

---

## Dataset Features

The dataset includes:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target Variable)

### Target

- 1 → Loan Approved
- 0 → Loan Rejected

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Project Workflow

1. Load dataset
2. Handle missing values
3. Encode categorical variables using one-hot encoding
4. Split data into training and testing sets
5. Train Logistic Regression model
6. Generate predictions
7. Evaluate model using confusion matrix and accuracy score

---

## Model Evaluation

The model performance is evaluated using:

- Confusion Matrix
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)

### Confusion Matrix Interpretation

|                | Predicted Reject | Predicted Approve |
|----------------|------------------|-------------------|
| Actual Reject  | True Negative    | False Positive    |
| Actual Approve | False Negative   | True Positive     |

---

## Project Structure
loan_approval/
│
├── data/
│ └── loan_data.csv
│
├── loan_model.py
├── README.md
└── requirements.txt

---

## How to Run

### Clone the repository

### Create virtual environment
python3 -m venv venv
source venv/bin/activate


### Install dependencies

### Run the project
python loan_model.py

---

## Future Improvements

- Add ROC Curve
- Hyperparameter tuning
- Handle class imbalance
- Deploy using Streamlit
- Compare with Random Forest & XGBoost

---

## Author

Aditya Sinha  
BTech CSE (AI & ML)
>>>>>>> 6160d75 (advancements)
