"""
Corporate Credit Risk Model Training & Evaluation
-------------------------------------------------
This script trains and evaluates binary classification models to predict corporate default risk.
It compares a baseline Logistic Regression model against an XGBoost Ensemble Classifier.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Import custom data pipeline module
from data_pipeline import generate_corporate_financials, compute_financial_ratios


def run_model_pipeline():
    # --- Step 1: Load and Prepare Data ---
    print("1. Generating dataset and calculating financial ratios...")
    raw_df = generate_corporate_financials(n_companies=1000, seed=42)
    df = compute_financial_ratios(raw_df)
    
    # Define features (predictive metrics) and target (default label)
    feature_cols = ['leverage_ratio', 'liquidity_ratio', 'interest_coverage', 'roa']
    X = df[feature_cols]
    y = df['default_status']
    
    # --- Step 2: Split Data into Training and Testing Sets ---
    # 80% used to train models, 20% reserved for unbiased final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # --- Step 3: Feature Scaling (Required for Linear Models) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Step 4: Train Baseline Model (Logistic Regression) ---
    print("\n2. Training Baseline Model: Logistic Regression...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    print("--- Logistic Regression Performance ---")
    print(classification_report(y_test, lr_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_probs):.4f}")
    
    # --- Step 5: Train Advanced Model (XGBoost Classifier) ---
    print("\n3. Training Advanced Model: XGBoost Gradient Boosting...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print("--- XGBoost Performance ---")
    print(classification_report(y_test, xgb_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
    
    # --- Step 6: Feature Importance Analysis ---
    importances = xgb_model.feature_importances_
    print("\n4. Feature Importance Ranking (XGBoost):")
    for feature, importance in zip(feature_cols, importances):
        print(f" - {feature}: {importance:.4f}")


if __name__ == "__main__":
    run_model_pipeline()
