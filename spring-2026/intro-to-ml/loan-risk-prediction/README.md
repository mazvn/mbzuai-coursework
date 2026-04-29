# loan risk prediction

**task:** kaggle-style competition. from applicant + credit history data, predict `RiskTier` (5 classes) and `InterestRate` (regression) for each loan. score combines accuracy and R².

**what i did:** a lot of feature engineering (delinquency scores, debt/income ratios, utilization, credit history stuff, group-relative features), out-of-fold target encoding, then pruning by importance and correlation. xgboost, lightgbm and catboost tuned with optuna, each trained as multiclass, ordinal and regression, over 5 seeds x 5 folds. then a second level stacking model and a weighted blend on top.

```
pip install -r requirements.txt
python loan_risk.py
```
needs `credit_train.csv` and `credit_test.csv` in the same folder (course data, not included). it's set up for a CUDA GPU and takes a long time.
