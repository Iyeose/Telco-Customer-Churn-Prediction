# Telco Customer Churn Prediction: A Segmented Machine Learning Analysis

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology Pipeline](#methodology-pipeline)
- [Key Findings & Results](#key-findings--results)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [How to Reproduce](#how-to-reproduce)
- [Conclusion](#conclusion)

---

## Introduction

Customer churn, the rate at which customers stop doing business with a company, is a critical metric, especially in the highly competitive telecommunications industry. Acquiring a new customer can be significantly more expensive than retaining an existing one. Therefore, accurately predicting which customers are at a high risk of churning is a valuable business capability.

This project develops a series of machine learning models to predict customer churn based on a public Telco dataset. The core of this analysis goes beyond a single predictive model by implementing a **segmented modeling approach**, creating specialized models for different customer groups based on their contract type. This allows for a deeper understanding of the unique factors that drive churn for distinct customer personas.

---

## Problem Statement

The objective of this project is to develop and evaluate machine learning models that can accurately predict customer churn. The key goals are:

1.  To identify the most significant drivers of customer churn through in-depth exploratory data analysis and feature importance analysis.
2.  To build, tune, and compare a range of classification models, from a simple baseline to advanced ensemble methods.
3.  To demonstrate the value of a segmented modeling strategy by analyzing how churn drivers and model performance differ across customer segments with Month-to-Month, One-Year, and Two-Year contracts.
4.  To provide actionable, data-driven insights that could inform targeted customer retention strategies.

---

## Dataset

This project utilizes the **"Telco Customer Churn"** dataset, which contains 7043 customer records and 21 features.

*   **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Features:** The dataset includes customer demographics (gender, senior citizen status), account information (tenure, contract type, payment method, monthly/total charges), and subscribed services (phone, internet, online security, etc.).
*   **Target Variable:** `Churn` - A binary indicator of whether the customer has left the company.

---

## Methodology Pipeline

The project follows a standard data science workflow:

1.  **Data Preprocessing & Cleaning:**
    *   Handled missing values in the `TotalCharges` column by converting it to a numeric type and imputing nulls with 0 for zero-tenure customers.
    *   Converted categorical features to appropriate data types and created a binary numerical target variable (`Churn_numeric`).
    *   Checked for and confirmed the absence of duplicate records.

2.  **Exploratory Data Analysis (EDA):**
    *   Performed a comprehensive EDA to visualize churn patterns. Key findings included a significantly higher churn rate for Month-to-Month customers, customers with Fiber Optic internet, and those without supportive add-ons like Online Security or Tech Support.

3.  **Feature Engineering:**
    *   **One-Hot Encoding:** Converted all categorical features into a numerical format suitable for machine learning using `pandas.get_dummies()`.
    *   **Feature Scaling:** Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure they were on a comparable scale for models like Logistic Regression.

4.  **Modeling Approach:**
    *   **Global Models:** Trained and tuned Logistic Regression, Random Forest, and XGBoost models on the entire dataset to establish performance benchmarks.
    *   **Segmented Models:** Segmented the dataset by `Contract` type (Month-to-Month, One-Year, Two-Year) and trained and tuned specialized Random Forest and XGBoost models for each segment independently.

5.  **Evaluation:**
    *   Models were evaluated using a range of metrics suitable for imbalanced classification: **Accuracy, Precision, Recall, F1-Score, ROC-AUC Score**, and the **Confusion Matrix**.

---

## Key Findings & Results

### Global Model Performance

The tuned **XGBoost** and **Random Forest** models were the top performers on the global dataset. The Tuned XGBoost achieved the highest **Recall (0.816)** and **ROC-AUC (0.846)**, making it excellent at identifying the maximum number of churners. The Tuned Random Forest achieved the best **F1-Score (0.641)**, indicating the best balance between precision and recall.

### Segmented Model Performance

The segmented analysis revealed that a "one-size-fits-all" model is suboptimal. The models tuned for each segment provided more targeted insights and performance characteristics.

| Contract Type | Best Model (by F1-Score) | F1-Score (Churn=1) | Recall (Churn=1) | Key Churn Drivers |
| :------------ | :----------------------- | :----------------- | :--------------- | :---------------- |
| **Month-to-Month** | Tuned Random Forest      | **0.637**          | 0.710            | Tenure, Total/Monthly Charges, Fiber Optic |
| **One-Year**       | Tuned XGBoost            | 0.376              | 0.758            | Streaming Services, Lack of Primary Service |
| **Two-Year**       | Tuned Random Forest      | 0.359              | 0.700            | Fiber Optic, Monthly Charges, Senior Citizen |

### Key Insights from Feature Importance

*   **Month-to-Month customers** are highly sensitive to tenure and cost. `InternetService_Fiber optic` is a major risk factor for this group.
*   **One-Year contract customers'** churn is more nuanced, driven by the perceived value of their specific service package (e.g., streaming services).
*   **Two-Year contract customers** are very loyal, but when they do churn, it's often linked to friction with premium services like Fiber Optic or high monthly charges.

![Feature Importance for Month-to-Month Segment](images/feature_importance_xgb_mtm.png "XGBoost Feature Importance - MTM Segment")
*(This is an example. Replace with the path to one of your key visualization images.)*

---

## Technologies Used

*   **Python 3.x**
*   **Jupyter Notebook** (via Google Colab)
*   **Pandas & NumPy:** Data manipulation and numerical operations.
*   **Matplotlib & Seaborn:** Data visualization.
*   **Scikit-learn:** Data preprocessing, model training (Logistic Regression, Random Forest), and evaluation.
*   **XGBoost:** High-performance gradient boosting library.

---

## File Structure
├── data/ # Data source information
├── notebooks/ # Main analysis Jupyter Notebook
│ └── Telco_Churn_Analysis.ipynb
├── images/ # Key visualizations used in the README
├── README.md # This file
└── requirements.txt # Python package dependencies

---

## How to Reproduce

1.  Clone this repository:
    ```bash
    git clone https://github.com/Iyeose/Telco-Customer-Churn-Prediction.git
    cd Telco-Customer-Churn-Prediction
    ```
2.  (Optional but recommended) Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the root directory of the project.
5.  Launch Jupyter Notebook or JupyterLab and open the `notebooks/Telco_Churn_Analysis.ipynb` file.

---

## Conclusion

This project successfully demonstrates the use of machine learning to predict customer churn. The key takeaway is the significant value of a **segmented modeling strategy**. By analyzing and building models for distinct customer groups based on contract type, we uncovered more specific and actionable insights into churn drivers than a global model could provide. The best models for each segment achieved high recall, successfully identifying the majority of at-risk customers, which is the first step in an effective customer retention strategy.
