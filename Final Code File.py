# -*- coding: utf-8 -*-
"""
Created on Fri May 16 21:12:32 2025

@author: Aliyaan
"""

#Main Model

# STEP 1: INSTALL DEPENDENCIES FIRST


import pandas as pd
import numpy as np
import optuna
import shap
import xgboost as xgb
import joblib
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, brier_score_loss,
    precision_score, f1_score
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
# from river import linear_model, preprocessing as river_preprocessing, metrics as river_metrics

warnings.filterwarnings('ignore')

# STEP 2: LOAD DATA
df = pd.read_csv("credit_risk_dataset.csv")
categorical = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.dropna(subset=["loan_status"])
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

numeric = X.select_dtypes(include=["int64", "float64"]).columns
X[numeric] = StandardScaler().fit_transform(X[numeric])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
X_train = pd.DataFrame(X_train).fillna(0).astype(np.float64)
X_test = pd.DataFrame(X_test).fillna(0).astype(np.float64)

# STEP 3: PSI CALCULATION
def calculate_psi(expected, actual, buckets=10):
    def scale_range(data): return (data - data.min()) / (data.max() - data.min())
    psi_values = []
    for col in expected.columns:
        e, a = scale_range(expected[col]), scale_range(actual[col])
        breakpoints = np.linspace(0, 1, buckets + 1)
        expected_counts = np.histogram(e, bins=breakpoints)[0] / len(e)
        actual_counts = np.histogram(a, bins=breakpoints)[0] / len(a)
        psi = np.sum((expected_counts - actual_counts) * np.log((expected_counts + 1e-6) / (actual_counts + 1e-6)))
        psi_values.append((col, psi))
    return pd.DataFrame(psi_values, columns=["Feature", "PSI"]).sort_values(by="PSI", ascending=False)

psi_df = calculate_psi(X_train, X_test)
print("Top PSI drift features:\n", psi_df.head())

# STEP 4: Y-DRIFT
train_rate, test_rate = y_train.mean(), y_test.mean()
print(f"Train default rate: {train_rate:.4f}, Test default rate: {test_rate:.4f}, Diff: {abs(train_rate - test_rate):.4f}")

# STEP 5: ADVERSARIAL VALIDATION
X_adv = pd.concat([X_train, X_test])
y_adv = np.array([0]*len(X_train) + [1]*len(X_test))
adv_model = LogisticRegression(max_iter=1000)
adv_model.fit(X_adv, y_adv)
adv_auc = roc_auc_score(y_adv, adv_model.predict_proba(X_adv)[:, 1])
print(f"Adversarial AUC (train vs test): {adv_auc:.4f}")

# STEP 6: OPTUNA FOR XGBOOST
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "use_label_encoder": False
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
xgb_best_params = study.best_params
xgb_best_params["use_label_encoder"] = False

xgb_model = xgb.XGBClassifier(**xgb_best_params)
xgb_model.fit(X_train, y_train)

# STEP 7: SHAP FEATURE SELECTION (TOP 30)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train)
shap_mean = np.abs(shap_values.values).mean(axis=0)
top_indices = np.argsort(shap_mean)[-30:]
top_features = X_train.columns[top_indices]
X_train = X_train[top_features].astype(np.float64)
X_test = X_test[top_features].astype(np.float64)

# STEP 8: STACKING + CALIBRATION
base_models = [
    ('lr', LogisticRegression(max_iter=1000, multi_class="ovr")),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=10)),
    ('xgb', xgb_model)
]
meta = RidgeClassifier()
stacked = StackingClassifier(estimators=base_models, final_estimator=meta, passthrough=True, n_jobs=-1)
calibrated = CalibratedClassifierCV(estimator=stacked, method='sigmoid', cv=5)
calibrated.fit(X_train.values, y_train.values)

# STEP 9: METRICS + EXPORT
y_proba = calibrated.predict_proba(X_test.values)[:, 1]
y_pred = calibrated.predict(X_test.values)

print("Final AUC:", roc_auc_score(y_test, y_proba))
print("Brier Score:", brier_score_loss(y_test, y_proba))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

submission = X_test.copy()
submission["actual"] = y_test.values
submission["predicted"] = y_pred
submission["probability"] = y_proba
submission.to_csv(r"C:\Users\maazj\Downloads\final_submission_stacked.csv", index=False)

# STEP 10: VERSIONING & RETRAINING
if psi_df["PSI"].max() > 0.2 or adv_auc > 0.6 or abs(train_rate - test_rate) > 0.05:
    print("⚠ Drift detected — retraining model...")
    calibrated.fit(X_test.values, y_test.values)
    joblib.dump(calibrated, r"C:\Users\maazj\Downloads\model_v2_retrained.pkl")
else:
    joblib.dump(calibrated, r"‪C:\IBA\FDA\Final")

# STEP 11: RIVER ONLINE TRAINING
print("Running River Online Logistic Regression...")
river_pipeline = river_preprocessing.StandardScaler() | linear_model.LogisticRegression()
river_auc = river_metrics.ROCAUC()

for xi, yi in zip(X_test.to_dict(orient='records'), y_test):
    pred = river_pipeline.predict_one(xi)
    river_auc = river_auc.update(yi, pred)
    river_pipeline = river_pipeline.learn_one(xi, yi)

print("River Online AUC:", river_auc.get())

# STEP 12: OPTIONAL - PyCaret AutoML (Uncomment only if on Python 3.9–3.11)
# from pycaret.classification import *
# df_py = pd.DataFrame(X[top_features])
# df_py['loan_status'] = y.values
# setup(data=df_py, target='loan_status', silent=True, use_gpu=True)
# best_model = compare_models()
#%%
#Model 1
# Catboost (Updated and Final)
# STEP 1: Install libraries

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from google.colab import files

# STEP 3: Load Data
df = pd.read_csv("/content/drive/MyDrive/FDA Final Pres/credit_risk_dataset.csv")
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]
cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# STEP 4: Advanced Optuna Objective with Cross-Validation
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": 42,
        "cat_features": cat_features
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)
    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30, timeout=600)
print("Best AUC:", study.best_value)
print("Best Parameters:", study.best_params)

# STEP 5: Train Final Model
best_params = study.best_params
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": 100,
    "random_seed": 42,
    "cat_features": cat_features
})
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

# STEP 6: Evaluation
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]
print("\n✅ Final Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Classification Report:\n", classification_report(y_test, y_pred))

# STEP 7: SHAP
explainer = shap.Explainer(final_model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10)

# STEP 8: PSI
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[-1] += 1e-6
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    psi_values = []
    for e, a in zip(expected_percents, actual_percents):
        if e == 0 or a == 0:
            psi_values.append(0)
        else:
            psi_values.append((e - a) * np.log(e / a))
    return np.sum(psi_values)

# Simulate drift
X_drifted = X_test.copy()
X_drifted['loan_intent'] = np.random.choice(X['loan_intent'].unique(), size=len(X_drifted), p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
X_drifted['loan_int_rate'] += np.random.normal(loc=2.0, scale=1.0, size=len(X_drifted))
expected_scores = final_model.predict_proba(X_test)[:, 1]
drifted_scores = final_model.predict_proba(X_drifted)[:, 1]
psi_score = calculate_psi(expected_scores, drifted_scores, buckets=10)
print(f"\n📊 Population Stability Index (PSI): {psi_score:.4f}")

# STEP 9: Save & Download
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred
results_df["predicted_prob"] = y_prob
results_df.to_csv("zest_advanced_model_predictions.csv", index=False)
files.download("zest_advanced_model_predictions.csv")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC CURVE
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.grid(True)
plt.show()

# PRECISION-RECALL CURVE
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

# XGBOOST FEATURE IMPORTANCE (plotting top 30)
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=30, importance_type='gain', height=0.6)
plt.title("XGBoost Feature Importance (Gain)")
plt.show()

# SHAP SUMMARY PLOT
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=30)

#%%
#Model 2
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
file_path = "C:/Users/Dell/Desktop/credit_risk_dataset.csv"
data = pd.read_csv(file_path)

# Visualize target class distribution
sns.countplot(x='loan_status', data=data)
plt.title('Class Distribution of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# -------- Feature Engineering -------- #

# Base features
data['income_per_age'] = data['person_income'] / (data['person_age'] + 1)
data['log_person_income'] = np.log(data['person_income'] + 1)
data['log_loan_amnt'] = np.log(data['loan_amnt'] + 1)
data['income_loan_interaction'] = data['person_income'] * data['loan_amnt']
data['debt_income_ratio'] = data['person_income'] / (data['person_emp_length'] + 1)
data['loan_age_ratio'] = data['loan_amnt'] / (data['person_age'] + 1)
data['log_age'] = np.log(data['person_age'] + 1)

# Polynomial terms
data['loan_amnt_squared'] = data['loan_amnt'] ** 2
data['income_squared'] = data['person_income'] ** 2
data['age_squared'] = data['person_age'] ** 2

# Binned features
data['income_bin'] = pd.qcut(data['person_income'], q=5, labels=False)
data['age_bin'] = pd.cut(data['person_age'], bins=[17, 25, 35, 50, 65, 100], labels=False)
data['emp_length_bin'] = pd.qcut(data['person_emp_length'], q=4, labels=False)
data['loan_amnt_bin'] = pd.qcut(data['loan_amnt'], q=5, labels=False)

# Credit history indicators
data['has_short_credit'] = (data['person_emp_length'] < 2).astype(int)
data['has_long_credit'] = (data['person_emp_length'] > 10).astype(int)

# Credit behavior
data['loan_to_income'] = data['loan_amnt'] / (data['person_income'] + 1)
data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
data['credit_risk_score'] = data['cb_person_default_on_file'] * data['loan_to_income']

# Categorical feature combination
data['grade_intent'] = data['loan_grade'].astype(str) + "_" + data['loan_intent'].astype(str)

# One-hot encoding
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'grade_intent']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features and target
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Optional PCA (can comment out if not needed)
pca = PCA(n_components=10)
X_scaled = pca.fit_transform(X_scaled)

# Handle class imbalance with ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)

# Time-series cross-validation setup
tscv = TimeSeriesSplit(n_splits=5)

# Store metrics
accuracy_scores = []
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

print("Time-Series Split Evaluation:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_resampled)):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Adjust for class imbalance
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight
    }

    clf = xgb.train(params, dtrain, num_boost_round=100)

    preds = clf.predict(dtest)
    y_pred = (preds > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else np.nan

    # Store metrics
    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc)

    print(f"Fold {fold+1}: Accuracy = {acc:.3f}, AUC = {auc if not np.isnan(auc) else 'NA'}, "
          f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1-score = {f1:.3f}")

# Print average metrics
print("\nAverage Metrics Across All Folds:")
print(f"Accuracy: {np.mean(accuracy_scores):.3f}")
print(f"AUC: {np.nanmean(auc_scores):.3f}")
print(f"Precision: {np.mean(precision_scores):.3f}")
print(f"Recall: {np.mean(recall_scores):.3f}")
print(f"F1-score: {np.mean(f1_scores):.3f}")

# Plot performance
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.plot(range(1, 6), accuracy_scores, marker='o', label='Accuracy')
plt.plot(range(1, 6), f1_scores, marker='s', label='F1-score')
plt.title('Accuracy & F1 Across Folds')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, 6), precision_scores, marker='o', label='Precision')
plt.plot(range(1, 6), recall_scores, marker='s', label='Recall')
plt.title('Precision & Recall Across Folds')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, 6), auc_scores, marker='o', color='green', label='AUC')
plt.title('AUC Across Folds')
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(clf, max_num_features=10, importance_type='weight', title='Top 10 Important Features')
plt.tight_layout()
plt.show()

# Save model
model_filename = 'xgboost_credit_model_advanced.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)

print(f"Model saved as {model_filename}")

#%%
#Model 4
# Step 1: Install XGBoost if needed

!pip install xgboost --quiet

Step 2: Import libraries and load data

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

Load dataset
df = pd.read_csv("credit_risk_dataset.csv")

Encode categoricals
categorical = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical:
df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.dropna(subset=["loan_status"])
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

Scale numeric columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

Step 3: Train basic XGBoost model

model = xgb.XGBClassifier(
max_depth=5,
learning_rate=0.1,
n_estimators=100,
subsample=0.8,
colsample_bytree=0.8,
objective='binary:logistic',
use_label_encoder=False,
random_state=42
)

model.fit(X_train, y_train)

Step 4: Evaluate model

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
