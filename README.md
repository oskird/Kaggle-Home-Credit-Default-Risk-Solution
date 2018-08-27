# Home-Credit-Default-Risk
Solution for Kaggle competition - [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

**Public kernel on Kaggle** 

[EDA + baseline model using application](https://www.kaggle.com/sz8416/eda-baseline-model-using-application)

[6 ways for feature selection](https://www.kaggle.com/sz8416/6-ways-for-feature-selection)

[Simple Bayesian Optimization for LightGBM](https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm)

## Strategy
**Feature Extraction**: 1) generate features manually from each table; 2) create trend features with *featuretools* API; 3) use clustering techniques (Gussian Mixture Model and K-means) to extract features from bureau and previous_application table

**Baseline model**: use LightGBM to test different feature combinations

**Parameter Tuning**: human intuition to target the parameters at a 'narrow' range; Bayes Optimization to get a more accurate parameter combination

**Model Stacking**: Level 1: 3 LightGBM model, 2 XGBoost model, 1 Random Forest model and 1 Artificial Neural Network; Level 2: Artificial Neural Network 

## Files in this repository
### Feature
**feature - bureau & bureau balance.py**: feature extraction from bureau and bureau_balance table

**feature - credit_card_balance.py**: feature extraction from credit_card_balance table

**feature - POS_CASH_balance.py**: feature extraction from POS_CASH_balance table

**feature - previous_application.py**: feature extraction from previous_application table

**feature - installments_payments.py**: feature extraction from installments_payments table

**feature - trend.py**: trend feature extraction from all tables using *featuretools* API

**feature - clustering.py**: feature extraction using clustering (GMM)

### Modeling
**model - single model for stacking (LightGBM as example).py**: Single machine learning model demo; use LightGBM as an example

**model - stacking with ANN.py**: Stack single predicting model with ANN

**model - Bayes Optimization.py**: Bayes Optimization for tuning hyper parameters

### Notebook
**EDA.ipynb**: Exploratory data analysis

**Feature Selection.ipynb**: 6 methods for feature selection


