# Home-Credit-Default-Risk
Solution for Kaggle competition - [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
**Public kernel on Kaggle** 
[EDA + baseline model using application](https://www.kaggle.com/sz8416/eda-baseline-model-using-application)
[6 ways for feature selection](https://www.kaggle.com/sz8416/6-ways-for-feature-selection)
[Simple Bayesian Optimization for LightGBM](https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm)

## Introduction
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

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


