# src/evaluate_model.py

from sklearn.metrics import accuracy_score, f1_score
import pickle
import numpy as np

# Load model from disk
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Evaluate model performance on test data
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Make predictions on new data
def make_prediction(model, new_data):
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    # Load preprocessed test data (use the same preprocessing pipeline from train script)
    from preprocess_data import preprocess_data, load_data
    
    # Path to dataset
    file_path = r"C:\Users\shash\OneDrive\Desktop\Repos\german-credit-risk-analysis\data\german.data"
    
    # Load and preprocess data (only test set is needed for evaluation)
    df = load_data(file_path)
    _, X_test_preprocessed, _, y_test = preprocess_data(df)

    # Evaluate Logistic Regression Model
    log_reg_model = load_model("models/logistic_regression.pkl")
    print("Evaluating Logistic Regression Model:")
    evaluate_model(log_reg_model, X_test_preprocessed, y_test)

    # Evaluate Random Forest Model
    rf_model = load_model("models/random_forest.pkl")
    print("\nEvaluating Random Forest Model:")
    evaluate_model(rf_model, X_test_preprocessed, y_test)

    # Evaluate Gradient Boosting Model
    gb_model = load_model("models/gradient_boosting.pkl")
    print("\nEvaluating Gradient Boosting Model:")
    evaluate_model(gb_model, X_test_preprocessed, y_test)

    # Example: Making a prediction on new data for each model
    # Assuming new_data is a single sample with 61 features (after preprocessing)
    new_data = np.array([X_test_preprocessed[0]])  # Just using an example from test set here
    
    print("\nMaking Predictions on New Data:")
    
    log_reg_prediction = make_prediction(log_reg_model, new_data)
    print(f"Logistic Regression Prediction: {log_reg_prediction}")
    
    rf_prediction = make_prediction(rf_model, new_data)
    print(f"Random Forest Prediction: {rf_prediction}")
    
    gb_prediction = make_prediction(gb_model, new_data)
    print(f"Gradient Boosting Prediction: {gb_prediction}")