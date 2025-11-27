# office category classification

**task:** given a table of office buildings (numeric + categorical columns), predict their `OfficeCategory`. graded on test accuracy.

**what i did:** stacking. around 8 different models (trees, boosting, logistic regression, naive bayes, QDA) each give class probabilities on a validation split, then a logistic regression learns how to combine them. then everything is refit on the full train data for the test predictions. only the linear/probabilistic models get scaled features, trees use raw values.

```
pip install -r requirements.txt
python office_classifier.py --train office_train.csv --test office_test.csv
```
data is from the course so it's not included.
