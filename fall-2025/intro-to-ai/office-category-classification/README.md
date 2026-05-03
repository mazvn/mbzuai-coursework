# Office Category Classification

**Task:** Given a table of office buildings with numeric and categorical columns, predict their `OfficeCategory`. Graded on test accuracy.

**What I did:** Stacking. Around 8 different models (tree ensembles, boosting, logistic regression, naive Bayes, QDA) each predict class probabilities on a validation split, and a logistic regression learns how to combine them. Then everything is refit on the full training data to predict the test set. Only the linear and probabilistic models get scaled features, the trees use raw values.

```
pip install -r requirements.txt
python office_classifier.py --train office_train.csv --test office_test.csv
```

The data is from the course, so it's not included.
