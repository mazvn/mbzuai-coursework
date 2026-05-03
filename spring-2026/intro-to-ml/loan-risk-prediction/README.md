# Loan Risk Prediction

**Task:** A Kaggle-style competition. From applicant and credit history data, predict `RiskTier` (5 classes) and `InterestRate` (regression) for each loan. The score combines accuracy and R².

**What I did:** A lot of feature engineering (delinquency scores, debt to income ratios, utilization, credit history and group-relative features), out-of-fold target encoding, then pruning by importance and correlation. XGBoost, LightGBM and CatBoost are tuned with Optuna and each one is trained as multiclass, ordinal and regression over 5 seeds and 5 folds. A second level stacking model and a weighted blend go on top.

```
pip install -r requirements.txt
python loan_risk.py
```

It needs `credit_train.csv` and `credit_test.csv` in the same folder. That's course data, so it's not included. The script is set up for a CUDA GPU and takes a long time to run.
