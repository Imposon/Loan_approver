import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

class LoanPredictor:
    def __init__(self, data_path="data/loan_data.csv", model_path="loan_model.joblib"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.encoders = {}
        self.feature_names = None

    def load_and_preprocess(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Data Cleaning
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        
        # Encoding
        df_encoded = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        # Target encoding
        le_status = LabelEncoder()
        df_encoded['Loan_Status'] = le_status.fit_transform(df['Loan_Status'])
        self.encoders['Loan_Status'] = le_status
        
        X = df_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = df_encoded['Loan_Status']
        
        self.feature_names = X.columns.tolist()
        return X, y

    def train(self):
        X, y = self.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Save model and encoders
        joblib.dump({
            'model': self.model,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }, self.model_path)
        
        return self.model.score(X_test, y_test)

    def load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.encoders = data['encoders']
            self.feature_names = data['feature_names']
            return True
        return False

    def predict(self, input_data):
        if self.model is None:
            if not self.load_model():
                self.train()
        
        # Convert input dict to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Encode inputs
        for col, le in self.encoders.items():
            if col in df_input.columns:
                # Handle unseen labels by mapping to a default if necessary (simplification here)
                try:
                    df_input[col] = le.transform(df_input[col])
                except ValueError:
                    df_input[col] = 0 # Default fallback
        
        # Ensure correct column order
        df_input = df_input[self.feature_names]
        
        prob = self.model.predict_proba(df_input)[0][1]
        prediction = self.model.predict(df_input)[0]
        
        return prediction, prob

if __name__ == "__main__":
    predictor = LoanPredictor()
    accuracy = predictor.train()
    print(f"Model trained with accuracy: {accuracy:.4f}")