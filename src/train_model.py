# src/train_model.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Train multiple models
def train_models(X_train, y_train):
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X_train, y_train)

    return log_reg, rf_clf, gb_clf

# Save trained models to disk
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # Load preprocessed data (you can load from preprocess_data.py or directly from saved files if needed)
    from preprocess_data import preprocess_data, load_data
    
    # Path to dataset
    file_path = r"C:\Users\shash\OneDrive\Desktop\Repos\german-credit-risk-analysis\data\german.data"
    
    # Load and preprocess data
    df = load_data(file_path)
    X_train_preprocessed, X_test_preprocessed, y_train_smote, y_test = preprocess_data(df)
    
    # Train models
    log_reg_model, rf_model, gb_model = train_models(X_train_preprocessed, y_train_smote)

    # Save models to disk
    save_model(log_reg_model, "models/logistic_regression.pkl")
    save_model(rf_model, "models/random_forest.pkl")
    save_model(gb_model, "models/gradient_boosting.pkl")

    print("Models trained and saved successfully.")