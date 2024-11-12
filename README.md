# german-credit-risk-analysis
A machine learning project to predict credit risk using German credit data.

# German Credit Risk Analysis and Predictive Modeling using Machine Learning Techniques

This project implements a machine learning pipeline to predict credit risk using the **German Credit Dataset**. The goal is to classify customers as either "good" or "bad" credit risks based on various features such as age, job status, loan amount, and credit history.

---

## Table of Contents:
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview <a id="project-overview"></a>

The **German Credit Risk Analysis** project aims to predict whether a customer is a good or bad credit risk based on their financial and personal information. The dataset contains 1000 samples with 20 features each. We used three machine learning models to predict credit risk:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

The project involves several key steps:
1. Data loading and exploration.
2. Data preprocessing (encoding categorical variables, scaling numerical features, balancing classes).
3. Model training using Logistic Regression, Random Forest, and Gradient Boosting.
4. Model evaluation using accuracy and F1 score.
5. Making predictions on new customer data.

---

## Technologies Used <a id="technologies-used"></a>

This project uses the following technologies and libraries:
- **Python 3.x**
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **Scikit-learn** (for machine learning models and preprocessing)
- **Imbalanced-learn** (for SMOTE class balancing)
- **Matplotlib/Seaborn** (for data visualization)

---

## Dataset <a id="dataset"></a>

The dataset used in this project is the **German Credit Dataset**, which contains information about individuals applying for loans. The dataset includes both categorical and numerical features such as:
- Age
- Job status
- Loan amount
- Duration of loan
- Credit history

The target variable (`Credit_risk`) is binary:
- `1`: Good credit risk 
- `2`: Bad credit risk

The dataset can be found in the `data/` directory in `.data` format.

---

## Data Preprocessing <a id="data-preprocessing"></a>

Data preprocessing is an essential step before feeding data into machine learning models. The following steps were performed:
1. **Handling Categorical Variables**: We used One-Hot Encoding to convert categorical variables into numerical form.
2. **Scaling Numerical Features**: We applied StandardScaler to scale numerical features such as `Credit_amount` and `Age_in_years`.
3. **Class Balancing**: Since the dataset was imbalanced (more good credit risks than bad ones), we applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in the training set.

---

## Modeling <a id="modeling"></a>

We compared three machine learning models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

Each model was trained on the preprocessed training data and evaluated on the test data.

### Model Evaluation Metrics:
We used two key metrics to evaluate model performance:
1. **Accuracy**: Measures overall correctness of predictions.
2. **F1 Score**: Harmonic mean of precision and recall (useful when dealing with imbalanced datasets).

---

## Results <a id="results"></a>

After training and evaluating the models, we obtained the following results:

| Model                 | Accuracy | F1 Score |
|-----------------------|----------|----------|
| Logistic Regression    | 0.7450   | 0.8061   |
| Random Forest          | 0.8000   | 0.8649   |
| Gradient Boosting      | 0.7800   | 0.8462   |

The best-performing model was **Random Forest**, which achieved an accuracy of 80% and an F1 score of 0.8649.

---

## Installation <a id="installation"></a>

To run this project locally, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/german-credit-risk-analysis.git
    ```

2. Navigate into the project directory:

    ```bash
    cd german-credit-risk-analysis
    ```

3. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

If you don't have a `requirements.txt` file, manually install these libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn