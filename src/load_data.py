# src/load_data.py

import pandas as pd

# Define column names based on german.doc (documentation)
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

# Load the dataset from the space-separated file
def load_data(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None, names=columns)
    return df

if __name__ == "__main__":
    # Path to the dataset
    file_path = "C:/Users/shash/OneDrive/Desktop/Repos/german-credit-risk-analysis/data/german.data"
    
    # Load the dataset
    df = load_data(file_path)
    
    # Display basic information about the dataset
    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {df.columns}")
    print(df.head())
