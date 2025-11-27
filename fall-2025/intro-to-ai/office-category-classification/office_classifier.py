import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

TARGET = "OfficeCategory"
SEED = 42
NEEDS_SCALING = {"logistic", "logistic_l1", "nb", "qda"}


def get_models(use_boosting=True):
    models = []
    # boosting libs are optional, just skip if not installed
    if use_boosting:
        try:
            from lightgbm import LGBMClassifier
            models.append(("lgbm", LGBMClassifier(random_state=SEED, n_estimators=400)))
        except ImportError:
            pass
        try:
            from xgboost import XGBClassifier
            models.append(("xgb", XGBClassifier(random_state=SEED, n_estimators=600,
                                                eval_metric="mlogloss", tree_method="hist")))
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier
            models.append(("cat", CatBoostClassifier(random_state=SEED, verbose=0)))
        except ImportError:
            pass

    models += [
        ("rf", RandomForestClassifier(random_state=SEED, n_estimators=400, n_jobs=-1)),
        ("et", ExtraTreesClassifier(random_state=SEED, n_estimators=400, n_jobs=-1)),
        ("histgb", HistGradientBoostingClassifier(random_state=SEED, max_iter=400)),
        ("gb", GradientBoostingClassifier(random_state=SEED, n_estimators=400)),
        ("logistic", LogisticRegression(random_state=SEED, max_iter=3000)),
        ("logistic_l1", LogisticRegression(random_state=SEED, penalty="l1", solver="saga", max_iter=3000)),
        ("nb", GaussianNB()),
        ("qda", QuadraticDiscriminantAnalysis()),
    ]
    return models


def find_id_col(df):
    for c in ["Id", "id", "ID", "ListingID", "PropertyID"]:
        if c in df.columns:
            return c
    return None


def encode(train, test):
    # fit encoders on train+test so test doesn't have unseen labels
    train, test = train.copy(), test.copy()
    cat_cols = [c for c in train.select_dtypes(include=["object", "category"]).columns if c != TARGET]
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]).astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    return train, test


def scale(fit_on, *dfs, cols):
    sc = StandardScaler().fit(fit_on[cols])
    out = []
    for d in dfs:
        d = d.copy()
        d[cols] = sc.transform(d[cols])
        out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="office_train.csv")
    p.add_argument("--test", default="office_test.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--no-boosting", action="store_true")
    args = p.parse_args()

    np.random.seed(SEED)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    id_col = find_id_col(test)
    train, test = encode(train, test)

    y = train[TARGET].values
    X = train.drop(columns=[TARGET])
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_tr_s, X_val_s = scale(X_tr, X_tr, X_val, cols=num_cols)

    models = get_models(not args.no_boosting)

    # level 1: each model predicts probabilities on the validation split
    val_probs = []
    for name, model in models:
        try:
            if name in NEEDS_SCALING:
                model.fit(X_tr_s, y_tr)
                probs = model.predict_proba(X_val_s)
            else:
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_val)
            print(f"{name:12s} {accuracy_score(y_val, probs.argmax(axis=1)):.4f}")
            val_probs.append(probs)
        except Exception as e:
            print(f"skipped {name}: {e}")

    # level 2: logistic regression on top of all the probabilities
    meta = LogisticRegression(random_state=SEED, max_iter=4000)
    meta.fit(np.hstack(val_probs), y_val)
    print(f"stacked      {accuracy_score(y_val, meta.predict(np.hstack(val_probs))):.4f}")

    # refit everything on full train and predict test
    X_s, test_s = scale(X, X, test, cols=num_cols)
    test_probs = []
    for name, model in models:
        try:
            if name in NEEDS_SCALING:
                model.fit(X_s, y)
                test_probs.append(model.predict_proba(test_s))
            else:
                model.fit(X, y)
                test_probs.append(model.predict_proba(test))
        except Exception as e:
            print(f"skipped {name} on full data: {e}")

    preds = meta.predict_proba(np.hstack(test_probs)).argmax(axis=1)
    ids = test[id_col].values if id_col else np.arange(len(test))
    pd.DataFrame({id_col or "Id": ids, TARGET: preds}).to_csv(args.output, index=False)
    print("saved", args.output)


if __name__ == "__main__":
    main()
