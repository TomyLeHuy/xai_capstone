# Capstone Project: Predicting Bank Term Deposit Subscriptions with Explainable AI

## Overview

This repository contains the work for the capstone project focused on the **Bank Marketing dataset**. The primary goal is to leverage machine learning to predict whether a client will subscribe to a term deposit and, critically, to use **Explainable AI (XAI)** to understand the drivers behind those predictions.

The project follows a comprehensive machine learning workflow, from data exploration and feature engineering to model building and evaluation. A key emphasis is placed on comparing interpretable "white-box" models with more complex "black-box" models, using a suite of XAI techniques to demystify the latter.

---

## Dataset

The analysis is based on the **Bank Marketing Data Set** from the UCI Machine Learning Repository. This dataset contains information on direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable `y`).

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Data File Used:** `bank-additional-full.csv`

## About the Dataset

The following description is provided by the data's authors at the UCI Machine Learning Repository.

> The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The dataset is provided in four distinct files, representing two different time periods and feature sets:

* **`bank-additional-full.csv`**: The primary dataset for this project. Contains all examples (**41,188 rows** and 20 input features), ordered by date from May 2008 to November 2010.
* **`bank-additional.csv`**: A 10% random sample of the above (**4,119 rows**), intended for testing more computationally demanding algorithms.
* **`bank-full.csv`**: An older version of the dataset with all examples from a previous period (**45,211 rows** and 17 input features).
* **`bank.csv`**: A 10% random sample of the older `bank-full.csv` dataset.

The central classification goal of this project is to build a model that can predict if a client will subscribe to a term deposit, which is indicated by the target **variable `y`**.

## Input variables:
   ### Bank client data:
   - age (numeric)
   - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
   - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
   - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
   - default: has credit in default? (categorical: "no","yes","unknown")
   - housing: has housing loan? (categorical: "no","yes","unknown")
   - loan: has personal loan? (categorical: "no","yes","unknown")
### related with the last contact of the current campaign:
   - contact: contact communication type (categorical: "cellular","telephone") 
   - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
   - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
   - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
  - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  - previous: number of contacts performed before this campaign and for this client (numeric)
  - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
### social and economic context attributes
  - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  - cons.price.idx: consumer price index - monthly indicator (numeric)     
  - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
  - euribor3m: euribor 3 month rate - daily indicator (numeric)
  - nr.employed: number of employees - quarterly indicator (numeric)

### Output variable (desired target):
  - y - has the client subscribed a term deposit? (binary: "yes","no")

> Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques. 

---

## Project Timeline

* **Submission Deadline:** Wednesday, 9 July 2025, 23:59 CEST

---

## Project Workflow

The project should be structured into three main stages:

### 1. Exploratory Data Analysis (EDA) 📊

* **Data Cleaning & Transformation:** Handling missing values, encoding categorical variables, and scaling numerical features to prepare the data for modeling.
* **Visualization:** Creating plots to uncover key patterns, distributions, and relationships between client attributes and the subscription outcome.

### 2. Machine Learning Workflow ⚙️

* **Feature Engineering:** Creating new, informative features from the existing data to improve model performance.
* **Model Building:** Implementing and tuning several models, including:
    * **Intrinsically Interpretable Models:** Such as Logistic Regression and Decision Trees.
    * **Black-Box Models:** Such as a Random Forest or Gradient Boosting Machine (e.g., XGBoost, LightGBM).
* **Evaluation:** Assessing model performance using a combination of single-threshold metrics (Precision, Recall, F1-Score) and threshold-independent metrics (ROC AUC).

### 3. Explainable AI (XAI) 🧠

The core of the analysis is to interpret model behavior at both a global and local level.

* **Global Methods:**
    * **Permutation Feature Importance** to identify the most influential features overall.
    * **Partial Dependence Plots (PDPs)** to visualize the marginal effect of features on predictions.
    * A **Global Surrogate Model** (e.g., a simple Decision Tree) to approximate the black-box model's behavior.
* **Local Methods:**
    * **SHAP (SHapley Additive exPlanations)** values to explain individual predictions.
    * **Counterfactual Explanations** to determine the smallest changes needed to flip a prediction outcome.
    * **Individual Conditional Expectation (ICE)** curves to show how a single instance's prediction changes as a feature is varied.

---

## Technology and Implementation

This project is currently implemented in **Python**. The core libraries used for analysis, modeling, and explainability include:

* `pandas` & `numpy` for data manipulation
* `matplotlib` & `seaborn` for visualization
* `scikit-learn` for data preprocessing and modeling
* `shap`, `dalex`, and `counterfactuals` for Explainable AI

### A Note on R

While the code in this repository is written in Python, the methodologies, analytical findings, and XAI principles are language-agnostic. The entire workflow—from EDA to generating SHAP values or partial dependence plots—can be replicated in **R** using popular packages like `tidymodels`, `DALEX`, and `iml`. The translation is primarily a matter of syntax, not of conceptual approach.

---
