# src/preprocess_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Define column names for loading data
columns = [
    'Status_of_existing_checking_account', 'Duration_in_month', 'Credit_history',
    'Purpose', 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
    'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex',
    'Other_debtors_guarantors', 'Present_residence_since', 'Property',
    'Age_in_years', 'Other_installment_plans', 'Housing',
    'Number_of_existing_credits_at_this_bank', 'Job',
    'Number_of_people_being_liable_to_provide_maintenance_for',
    'Telephone', 'Foreign_worker', 'Credit_risk'
]

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None, names=columns)
    return df

# Preprocess data: handle categorical variables, scale numerical features, balance classes
def preprocess_data(df):
    # Separate features and target variable ('Credit_risk')
    X = df.drop('Credit_risk', axis=1)
    y = df['Credit_risk']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing pipeline for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply preprocessing pipeline to training and test data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Apply SMOTE to balance classes in training set (after encoding)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)

    return X_train_smote, X_test_preprocessed, y_train_smote, y_test

if __name__ == "__main__":
    # Path to the dataset
    file_path = r"C:\Users\shash\OneDrive\Desktop\Repos\german-credit-risk-analysis\data\german.data"
    
    # Load the dataset
    df = load_data(file_path)
    
    # Preprocess the data
    X_train_preprocessed, X_test_preprocessed, y_train_smote, y_test = preprocess_data(df)
    
    # Display shapes of preprocessed data
    print(f"Training Data Shape: {X_train_preprocessed.shape}")
    print(f"Test Data Shape: {X_test_preprocessed.shape}")
