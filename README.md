# Financial-Data-Analytics
# 📊 Assignment 1 – Linear Regression Analysis

This assignment focuses on applying various regression techniques using Python to explore and understand relationships between variables in real-world datasets. The assignment was executed on Google Colab across three Jupyter notebooks and provides a practical foundation in simple, multiple, and advanced linear regression.

## 🔧 Technologies Used

* **Platform:** Google Colab
* **Languages:** Python
* **Libraries:** `pandas`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`

---

## 📁 Assignment Structure

### 1. 📈 Simple Linear Regression – Salary vs Experience

**Notebook:** `Linear_Regression_Salary.ipynb`
**Objective:** Predict salary based on years of experience.

**Key Highlights:**

* Positive coefficients imply higher salary with more experience.
* Outliers can heavily skew regression results.
* Lack of test data leads to overfitting and biased evaluation.
* Skewed or missing variables (like certifications or education) introduce bias.
* Larger training sets improve generalizability and model stability.
* Implements **Ordinary Least Squares (OLS)** using `LinearRegression().fit()`.

---

### 2. 📊 Multiple Linear Regression – Startup Profit Prediction

**Notebook:** `MLR_Startup_Profit.ipynb`
**Objective:** Predict company profit based on R\&D, Administration, and Marketing spends.

**Key Insights:**

* **R\&D Spend** has the most significant positive impact on profit.
* Marketing spend has wide variability and potential outliers.
* **One-hot encoding** is required for categorical variables like Industry.
* Evaluates **feature importance**, **correlations**, and **multicollinearity**.
* Highlights issues of extrapolation and model transferability across countries.

---

### 3. 📚 Advanced Linear Regression – Student Performance Analysis

**Notebook:** `1_studentp_Regression_Applied.ipynb`
**Objective:** Predict a student’s performance index using multiple predictors.

**Predictors Used:**

* Hours Studied
* Previous Scores
* Extracurricular Activities
* Sleep Hours
* Sample Papers Practiced

**Findings:**

* All features were statistically significant (`p < 0.05`).
* **Hours Studied** and **Previous Scores** showed the strongest impact.
* Demonstrated risks of **overfitting** with too many or redundant variables.
* Emphasized importance of **cross-validation**, **regularization**, and **feature selection**.

---

## ✅ Learning Outcomes

* Developed practical skills in applying regression models.
* Understood the impact of data quality, feature relevance, and model assumptions.
* Gained experience interpreting model outputs such as coefficients, R², p-values, and F-statistics.
* Identified key challenges like bias, overfitting, multicollinearity, and generalizability.
  

# 🧾 Logistic Regression: Insurance Claim Fraud Detection

**Assignment 2**

---

## 📌 Objective

The goal of this assignment was to develop a **Logistic Regression model** that classifies insurance claims as **fraudulent (1)** or **non-fraudulent (0)** using the provided dataset. The project followed a complete machine learning pipeline including data preprocessing, handling class imbalance, model building, and evaluation.

---

## 📁 Project Structure

* `Assignment_2_26088_FDA.ipynb`: Jupyter Notebook containing full implementation
* `README.md`: Project summary and documentation
* Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

---

## ✅ Tasks & Marking Scheme Breakdown

### 1. 📊 Data Preprocessing & EDA

* Cleaned the dataset using `pandas`
* Summarized statistics and visualized class imbalance
* Detected significant skew in the target class (fraud cases were < 2%)

### 2. ⚖️ Handling Class Imbalance

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Compared with alternative methods like undersampling and class-weight adjustment
* Used **L2 regularization** to prevent overfitting

### 3. 🧮 Logistic Regression Model 

* Split data using an 80/20 **train-test split**
* Normalized features using **StandardScaler**
* Built logistic regression model with `sklearn.linear_model.LogisticRegression`

### 4. 📈 Model Evaluation 

* **Accuracy**: `98.6%`
* **AUC-ROC Score**: `0.62`
* **Classification Report**:

| Class         | Precision | Recall | F1-Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| 0 (Non-Fraud) | 0.99      | 1.00   | 0.99     | 1980    |
| 1 (Fraud)     | 0.00      | 0.00   | 0.00     | 20      |

* **Macro Avg F1**: `0.50`
* **Weighted Avg F1**: `0.98`
* Confusion Matrix showed high false negatives → model missed most fraud cases.

### 5. 💡 Insights & Recommendations

#### ✔ Strengths:

* Very high accuracy and recall for **non-fraud** class
* SMOTE improved learning on minority class (fraud)

#### ✘ Areas of Improvement:

* Logistic regression is too simplistic for extreme class imbalance
* Recommend using **ensemble models** (Random Forest, XGBoost)
* Incorporate **domain features** (claim type, customer history) to boost fraud signal
* Perform **threshold tuning** and **cost-sensitive learning**

---

## 📚 Key Learnings

* How to handle **imbalanced datasets** using oversampling techniques like SMOTE
* Practical use of **logistic regression** for binary classification
* Importance of **evaluation metrics** beyond accuracy in fraud detection
* Gained insight into **real-world fraud analytics challenges**

---

## 🔧 Technologies & Tools

* Python 3.x
* Jupyter Notebook
* Libraries:

  * `pandas`, `numpy` for data manipulation
  * `seaborn`, `matplotlib` for visualization
  * `scikit-learn` for modeling
  * `imblearn` for SMOTE

---

# 🧠 Assignment 3: HSBC Customer Segmentation using K-Means Clustering

This project applies **K-Means Clustering** on HSBC’s customer dataset (ages under 35) to segment customers based on their financial behaviors. The goal is to generate actionable insights for **Customer Life Cycle Management (CLCM)** by using both **top-down** and **bottom-up** segmentation strategies. Feature engineering, visualization, and clustering evaluation techniques are used to enhance model performance and real-world applicability.

---

## 📁 Files Included

* `26088-FDA assignment 3.ipynb`: Jupyter notebook with complete code including preprocessing, clustering, feature creation, and visualization.
* `26088-Assignment 3 analysis.pdf`: Written report answering the case study-related questions.
* `HSBC_ST138D-XLS-ENG.xlsx`: The original dataset (excluded from GitHub due to privacy compliance).

---

## 🎯 Assignment Objectives

* Apply **K-Means Clustering** to HSBC’s under-35 customer data.
* Segment customers using both **top-down** (macro) and **bottom-up** (behavior-driven) approaches.
* Engineer new financial features to improve cluster quality.
* Determine optimal cluster count using **Elbow Method** and **Silhouette Score**.
* Interpret and visualize clusters to drive real-world business insights for HSBC and local banks in Pakistan.

---

## 🔄 Process Overview

### 1. 🧼 Data Preprocessing

* Converted categorical fields like `AGE` into numerical format.
* Removed duplicates and reset index.
* Scaled financial features using **StandardScaler** to prepare for distance-based clustering.

### 2. 📊 Exploratory Data Analysis (EDA)

* Generated summary statistics and plotted distributions.
* Created a **correlation heatmap** to identify relationships between financial attributes and product adoption.

### 3. ✨ Feature Engineering

Created 3 additional features to enhance clustering performance:

* **Income-to-TRB Ratio**: Indicates financial stability.
* **Age-to-Income Ratio**: Captures generational earning differences.
* **TRB Squared**: Amplifies detection of outliers with very high balances.

✅ **Result**: Silhouette Score improved from **0.77 → 0.93**, showing tighter and more meaningful clusters.

---

### 4. ⚙️ K-Means Clustering

* Applied clustering on scaled features for various `k` values (from 1 to 10).
* Used **Elbow Method** to identify optimal `k`.
* Tested edge cases like `k = 1`, `k = n`, and `k > n` to explore clustering boundaries.

### 5. 🧠 Cluster Evaluation

* Evaluated using **Silhouette Score**.
* Added `Which_Cluster` column to the dataset.
* Cluster profiling revealed 3 distinct customer types:

| Cluster | Traits                                                       |
| ------- | ------------------------------------------------------------ |
| **0**   | Low-income, low-engagement → Risk of churn                   |
| **1**   | Middle-income, moderate product usage → Cross-sell potential |
| **2**   | High-income, digitally engaged → Upsell and loyalty focus    |

---

## 📌 Business Insights

* **Top-Down Segmentation**: Revealed macro patterns (e.g., income levels).
* **Bottom-Up Segmentation**: Exposed micro-patterns driven by actual behavior, yielding more actionable clusters.
* These segments help HSBC **target specific demographics**, **enhance customer experience**, and **optimize marketing efforts**.

---

## 🌍 Local Application: UBL (Pakistan)

If implemented at UBL, these clustering methods can:

* Identify financially underserved young segments.
* Personalize digital financial services for active users.
* Improve inclusion by offering tailored savings and lending products.

---

## 🛠 Technologies Used

* **Python**
* **Jupyter Notebook / Google Colab**
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📄 Report Summary (Answering Key Questions)

1.Three engineered features (Income/TRB, Age/Income, TRB²) significantly improved clustering, boosting silhouette score from 0.77 to 0.93.
2.Segmentation enables HSBC to tailor products based on real customer behaviors, increasing satisfaction and retention.
3.Top-down used predefined variables (e.g., income), while bottom-up discovered natural clusters without assumptions.
4.UBL could replicate this method to identify high-growth customers, personalize services, and drive inclusion with data-driven targeting.

---

## ✅ Conclusion

This assignment demonstrated the effectiveness of K-Means clustering in real-world financial segmentation. By incorporating feature engineering and evaluation, we generated meaningful customer profiles that can transform how banks like HSBC and UBL manage their customers across the life cycle.

# 🧠 Final Projet: Zest AI

# 💳 Credit or Regret? Tackling Economic Drift in Lending with Zest AI-Inspired ML

> A dynamic, explainable machine learning framework to fight economic model drift in credit risk — inspired by **Zest AI**

---

## 🎯 Objective

This project addresses a real-world challenge in the **fintech industry**:

> How can we make credit risk predictions that stay reliable during rapid economic shifts?

Using the case of **Zest AI**, we replicate their challenge of **economic model drift**, which happens when borrower behavior changes due to inflation, layoffs, or other volatile events — making traditional, static models inaccurate over time.

---

## 🧩 Problem Statement

> **“Credit or Regret? Let’s Predict That Debt.”**

Zest AI uses batch-updated credit scoring models. However, **when the economy changes fast**, these models may become **stale**, resulting in:

* ❌ Inaccurate lending decisions
* 📈 Higher default risk
* 🧊 Frozen model performance in dynamic conditions

### ✅ Our Solution:

A **drift-aware, online learning credit risk model** that:

* Detects when model performance degrades
* Retrains itself when borrower behavior shifts
* Keeps lenders ahead of economic turbulence

---

## 📊 Dataset & Preprocessing

* **Source**: Public credit risk dataset from Kaggle (synthetic/real blend)
* **Target Variable**: `loan_status` (approved vs defaulted)
* **Key Features**:

  * Income
  * Employment length
  * Loan intent
  * Credit score
  * ...and 12+ more

### 🔧 Preprocessing Steps:

* Handled missing values in target (`loan_status`)
* Label encoding for categorical variables
* Standard scaling for numerical features
* Train-Test Split: 70/30 stratified by `loan_status`
* **Drift Simulation**:

  * Train/test split mimics **time-based economic shifts**

---

## 🤖 Model Architecture

### 🧠 Final Model: **Stacked Ensemble with Drift Awareness**

#### 🔹 Base Learners:

* **Logistic Regression** – Interpretable & fast
* **Random Forest** – Handles non-linearities well
* **XGBoost (Optuna-tuned)** – High-performance gradient boosting

#### 🔹 Meta Learner:

* **Ridge Classifier** – Smooths ensemble output and avoids overfitting

### ⚙️ Optimization & Adaptability:

* **Optuna**: Hyperparameter tuning for all models
* **River**: Integrated for **online learning** (stream-based data)
* **Drift Detection**:

  * Monitors PSI, Y-drift, and adversarial validation
  * Automatically retrains when thresholds are exceeded

### 🔍 Explainability:

* **SHAP values**: For feature importance & transparency
* **Platt scaling**: For reliable probability outputs (trustworthy decisions)

---

## 📉 Drift Detection & Auto-Retraining

We simulate time-based drift and implement built-in model retraining:

* ✅ **Drift Detectors**:

  * PSI (Population Stability Index)
  * Adversarial Validation
  * Target Drift

* 🔁 **Auto-Retrain Triggers**:

  * When drift exceeds thresholds
  * Seamlessly re-updates using River’s online learning support

---

## 🧠 Learnings

> *"This model didn’t just learn — **we** did too."*

* 📉 We learned that 50% accuracy in credit decisions **hurts more than a heartbreak** 💔
* 🧹 Data cleaning is a battle, but one worth fighting
* 🧠 Model stacking isn't just for performance — it's for impact
* 🎯 Drift detection isn't optional — it's survival in volatile markets

---

## 🛠 Tech Stack

* **Python**, Jupyter Notebooks
* **scikit-learn**, **XGBoost**, **Optuna**, **River**
* **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
* **SHAP**, **Platt Scaling**

---

## 📌 Business Impact

* 🚀 **Increased Accuracy**: Improves loan approval and reduces default risk
* 🔄 **Real-time Learning**: Stays fresh even in chaotic economies
* 🤖 **Automation-Ready**: Drift triggers retraining with no human intervention
* 🔍 **Explainability**: Boosts regulatory compliance & trust

> “Every 1% gain in accuracy = a misdiagnosis avoided or a loan decision improved. That’s **impact at scale**.”

---

## 📄 Deliverables

* ✅ Final Jupyter Notebook (code + visualizations + model results)
* ✅ PowerPoint Slides (1-slide summary per model + project overview)
* ✅ Report (problem, pipeline, performance, learnings, business case)
