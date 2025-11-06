# 🌍 Risk Evaluation POC Project Plan

## 🔍 Overview

This document outlines the investigation, analysis, and proof-of-concept (POC) development plan for a startup project offering **risk evaluation as a service for retail banks**, based on the Home Credit dataset.

Since the most pressing business problems for our clients (banks) are not yet fully understood, the project explores **multiple predictive use cases** to provide a robust and diverse model offering.

---

## 📝 Assumptions

- Retail banks want to improve credit risk and customer management using data-driven tools.
- Machine learning models can provide value through predictive insights (e.g., default risk).
- The Home Credit dataset offers rich information to experiment with various use cases.
- Clients require accurate, interpretable models that align with regulatory expectations.

---

## 🎯 Overall Objectives

- Explore multiple business-relevant predictive tasks (e.g., default, early repayment).
- Build baseline and advanced ML models for each task.
- Identify promising models to include in the final demo offering.
- Package insights into a presentable, understandable format for clients.

---

## 🛋️ Step-by-Step Plan

### Step 1: Data Exploration & Setup

**Objective:** Understand the dataset and prepare it for modeling.

- Load and explore all relevant files.
- Merge tables (e.g., application, bureau, POS, installment).
- Handle missing values, outliers, and categorical variables.
- Conduct exploratory data analysis (EDA): distributions, correlations, target imbalance.
- Identify viable target variables for different predictive tasks.
- Apply statistical inference (e.g., t-tests, chi-square tests, ANOVA) to:

  - Compare default vs non-default groups on key variables
  - Identify significant associations (e.g., education level vs repayment behavior)
  - Support variable selection and hypothesis refinement



### Step 2: Define Prediction Targets

**Objective:** Identify several business problems to model.

Possible targets:

- **Loan Default Risk** (`TARGET` column) [realised]
 - **Interpretable Production Model**: Trafe-off between performance, limited number of features, interpretability
 - **Low-Dependency Model**: Model independent from external data (e.g. External evaluation scores), plus trafe-off between performance, limited number of features, interpretability
- **Early Repayment** (derived from payment schedule vs actual payments) [not realised]
- **Fraud/Inconsistencies** (based on conflicting inputs or rules)  [not realised]

Document rationale and business value of each.

### Step 3: Feature Engineering

**Objective:** Create a rich and informative feature set.

- Aggregate financial behavior (e.g., avg credit amount, payment delays).
- Behavior-based features (e.g., late payment count, change over time).
- Interaction terms (e.g., income/loan ratio, credit/debt ratio).
- Feature selection, PCA.

### Step 4: Modeling & Evaluation

**Objective:** Train and evaluate models per use case.

- Model type: CatBoost
- Train/test split or cross-validation.
- Evaluation metrics:
  - Classification: ROC-AUC, F1, PR-AUC, Accuracy
- Use SHAP values or other tools for interpretability.

### Step 5: Interpretation & Insights

**Objective:** Identify which models/tasks are promising.

- Compare performance across targets.
- Analyze and visualize feature importance.
- Assess business value of each use case.
- Choose 2-3 models to present in demo.

### Step 6: POC Packaging & Demo deployment

**Objective:** Prepare client-facing prototype.

- Wrap models for deployment via GoogleCloud Service
- Allow inputs and show predictions + explanations.

---

**Last updated:** August 2025