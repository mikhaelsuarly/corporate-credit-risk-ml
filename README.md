# Corporate Credit Risk & Default Prediction Model

An end-to-end Machine Learning pipeline evaluating corporate default risk using financial ratios and tree-based classification models. Designed to translate complex predictive modeling into actionable business insights for credit risk mitigation.

---

## Executive Summary

Credit risk modeling is critical for financial institutions to minimize non-performing loans and optimize capital allocation. This project constructs an end-to-end classification pipeline comparing baseline linear models against ensemble learning algorithms (XGBoost) to predict corporate default probability.

---

## Key Business Objectives

* **Predictive Accuracy:** Build and optimize a high-precision classifier for early detection of financial distress.
* **Model Explainability:** Extract feature importances to identify key drivers of default (e.g., leverage and liquidity ratios).
* **Risk Trade-off Analysis:** Minimize False Negatives (Type II error) to reduce unmitigated default exposure.

---

## Technical Architecture & Methodology

* **Data Ingestion & Preprocessing:** Synthetic accounting data generation
* **Feature Engineering:** Leverage, Liquidity, Interest Coverage, and ROA metrics
* **Model Training & Evaluation:** Logistic Regression (Baseline) vs. XGBoost Classifier
* **Risk Analytics:** ROC-AUC evaluation and feature ranking

---

## Model Performance & Comparison

Models were evaluated using an 80/20 Train-Test split with primary focus on **ROC-AUC** and **Recall** to handle default risk detection.

| Model | Classification Accuracy | ROC-AUC Score | Primary Driver |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | ~82% | ~0.84 | Linear coefficient weights |
| **XGBoost Classifier** | **~88%** | **~0.90** | Non-linear tree splits |

---

## Repository Structure

* `src/data_pipeline.py` - Data generation & ratio calculations
* `src/model_evaluator.py` - Machine learning training & evaluation
* `requirements.txt` - Project dependencies
* `README.md` - Documentation

---

## Author

**Mikhael**  
Data Technical Account Manager  
*Applied Machine Learning & Financial Data Science Project*  
[LinkedIn](www.linkedin.com/in/laurentiusmikhael)
