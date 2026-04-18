"""Train and persist models for the dashboard. Mirrors notebook 04."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "features.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

FEATURE_COLS_ORIG = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
ENGINEERED_COLS = ["dance_x_energy", "vocal_presence", "electronic_score", "rap_signal", "loudness_norm"]
FEATURE_COLS = FEATURE_COLS_ORIG + ENGINEERED_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dance_x_energy"]   = df["danceability"] * df["energy"]
    df["vocal_presence"]   = 1.0 - df["instrumentalness"]
    df["electronic_score"] = df["energy"] - df["acousticness"]
    df["rap_signal"]       = df["speechiness"] * df["danceability"]
    loud_min, loud_max     = df["loudness"].min(), df["loudness"].max()
    df["loudness_norm"]    = (df["loudness"] - loud_min) / (loud_max - loud_min)
    return df


def load_via_sql() -> tuple[np.ndarray, np.ndarray]:
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE features AS SELECT * FROM read_csv_auto('{DATA}');")
    df = con.execute("""
        SELECT danceability, energy, loudness, speechiness,
               acousticness, instrumentalness, liveness, valence, tempo, label
        FROM features
        WHERE tempo > 0 AND danceability IS NOT NULL AND loudness IS NOT NULL;
    """).fetchdf()
    df = engineer_features(df)
    return df[FEATURE_COLS].values, df["label"].values


def build_searches(cv):
    lr_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    lr_search = RandomizedSearchCV(
        lr_pipe,
        {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l2"]},
        n_iter=20, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )

    rf_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_search = RandomizedSearchCV(
        rf_pipe,
        {
            "clf__n_estimators":      randint(100, 500),
            "clf__max_depth":         [None, 5, 10, 15, 20],
            "clf__min_samples_split": randint(2, 20),
            "clf__min_samples_leaf":  randint(1, 10),
            "clf__max_features":      ["sqrt", "log2", None],
        },
        n_iter=25, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )

    xgb_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", xgb.XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
        )),
    ])
    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        {
            "clf__n_estimators":     randint(100, 600),
            "clf__max_depth":        randint(3, 10),
            "clf__learning_rate":    loguniform(1e-3, 0.3),
            "clf__subsample":        uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__gamma":            uniform(0, 5),
            "clf__reg_lambda":       loguniform(1e-2, 10),
        },
        n_iter=30, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return lr_search, rf_search, xgb_search


def evaluate(name, est, X_te, y_te):
    proba = est.predict_proba(X_te)[:, 1]
    pred = est.predict(X_te)
    return {
        "model": name,
        "roc_auc":   roc_auc_score(y_te, proba),
        "pr_auc":    average_precision_score(y_te, proba),
        "f1":        f1_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred),
    }


def main() -> int:
    X, y = load_via_sql()
    print(f"Dataset: {X.shape}, hits={int(y.sum())}, non-hits={int((y==0).sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr_search, rf_search, xgb_search = build_searches(cv)

    print("Tuning LR..."); lr_search.fit(X_tr, y_tr);  print(f"  best AUC {lr_search.best_score_:.3f}")
    print("Tuning RF..."); rf_search.fit(X_tr, y_tr);  print(f"  best AUC {rf_search.best_score_:.3f}")
    print("Tuning XGB..."); xgb_search.fit(X_tr, y_tr); print(f"  best AUC {xgb_search.best_score_:.3f}")

    results = pd.DataFrame([
        evaluate("Logistic Regression", lr_search.best_estimator_,  X_te, y_te),
        evaluate("Random Forest",       rf_search.best_estimator_,  X_te, y_te),
        evaluate("XGBoost",             xgb_search.best_estimator_, X_te, y_te),
    ]).round(3)
    print("\nHeld-out test metrics:")
    print(results.to_string(index=False))

    best_name = results.loc[results["roc_auc"].idxmax(), "model"]
    best_est = {
        "Logistic Regression": lr_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "XGBoost":             xgb_search.best_estimator_,
    }[best_name]

    rf_clf  = rf_search.best_estimator_.named_steps["clf"]
    xgb_clf = xgb_search.best_estimator_.named_steps["clf"]
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "rf_importance":  rf_clf.feature_importances_,
        "xgb_importance": xgb_clf.feature_importances_,
    })

    out = MODELS / "hit_predictor.joblib"
    joblib.dump({
        "best_model":         best_est,
        "best_name":          best_name,
        "feature_cols":       FEATURE_COLS,
        "feature_cols_orig":  FEATURE_COLS_ORIG,
        "engineered_cols":    ENGINEERED_COLS,
        "results":            results.to_dict(orient="records"),
        "feature_importance": importance_df.to_dict(orient="records"),
    }, out)
    print(f"\nSaved {out}  (best: {best_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
