"""Classifiers, hyperparameter tuning and out-of-fold predictions (Sections 4.1 and 4.5)."""

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import CATEGORICAL, NUMERICAL, SEED

MODEL_DIR = Path("results/models")


def make_preprocessing(scale):
    """Build the preprocessing step for one classifier (Section 3.3).

    Parameters
    ----------
    scale : bool
        If True, standardize the numerical predictors (logistic regression);
        if False, pass them through unchanged (tree ensembles).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted transformer: numerical columns scaled or passed through,
        categorical columns one-hot encoded (binary variables as one column).
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", numeric, NUMERICAL),
        ("cat", OneHotEncoder(drop="if_binary"), CATEGORICAL),
    ])


def make_models():
    """Define the four classifiers and their hyperparameter grids (Section 4.1, Table 4).

    Returns
    -------
    dict
        Maps each model name to a dict with keys "estimator" (unfitted
        classifier), "grid" (hyperparameter values to try) and "scale"
        (whether numerical predictors are standardized).
    """
    return {
        "Logistic regression": {
            "estimator": LogisticRegression(
                solver="saga",  # supports both L1 and L2 penalties
                max_iter=2000,
                random_state=SEED),
            "grid": {"model__l1_ratio": [0, 1],  # 0 = ridge, 1 = lasso (selects the penalty)
                     "model__C": [0.01, 0.1, 1, 10]},  # inverse penalty strength (alpha = 1/C)
            "scale": True,
        },

        "Random forest": {
            "estimator": RandomForestClassifier(random_state=SEED, n_jobs=1),
            "grid": {"model__n_estimators": [100, 200],  # number of trees B
                     "model__min_samples_leaf": [5, 20],  # minimum number of loans per leaf
                     "model__max_features": ["sqrt", 0.8]},  # m: predictors considered per split
            "scale": False,
        },

        "XGBoost": {
            "estimator": XGBClassifier(
                tree_method="hist",  # histogram-based split search (fast)
                random_state=SEED,
                n_jobs=1),  # one core per model; the grid search runs in parallel
            "grid": {"model__n_estimators": [150, 300],  # number of trees K
                     "model__max_depth": [3, 5],  # maximum tree depth
                     "model__learning_rate": [0.05, 0.1]},  # learning rate eta
            "scale": False,
        },

        "LightGBM": {
            "estimator": LGBMClassifier(
                random_state=SEED,
                n_jobs=1,
                verbosity=-1),  # print only fatal errors
            "grid": {"model__n_estimators": [150, 300],  # number of trees K
                     "model__num_leaves": [15, 31],  # maximum leaves per tree J_max
                     "model__learning_rate": [0.05, 0.1]},  # learning rate eta
            "scale": False,
        },
    }


def tune(spec, X_fit, y_fit):
    """Tune one classifier by grid search and compute out-of-fold probabilities (Section 4.5).

    Hyperparameters are chosen by 5-fold stratified cross-validation on
    log-loss; the best configuration is then refitted on the whole fit set.

    Parameters
    ----------
    spec : dict
        One entry of make_models(), with keys "estimator", "grid" and "scale".
    X_fit : pandas.DataFrame
        Predictors of the fit set.
    y_fit : pandas.Series
        Default indicator of the fit set.

    Returns
    -------
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search; best_estimator_ is the final model.
    oof : numpy.ndarray
        Out-of-fold default probabilities for the fit set (used for the F1 threshold).
    """
    pipe = Pipeline([("prep", make_preprocessing(spec["scale"])),
                     ("model", spec["estimator"])])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = GridSearchCV(pipe,
                          spec["grid"],
                          scoring="neg_log_loss",  # judge each combination by log-loss
                          cv=cv,
                          n_jobs=-1,  # use all CPU cores
                          verbose=1)  # print progress
    search.fit(X_fit, y_fit)  # CV over all combinations, then refit the best on the whole fit set

    # Each fit loan is predicted by a model trained on the other four folds
    oof = cross_val_predict(search.best_estimator_,  # winning hyperparameters
                            X_fit,
                            y_fit,
                            cv=cv,  # same folds as the grid search
                            method="predict_proba",  # probabilities, not labels
                            n_jobs=-1)[:, 1]  # column 1 = P(default)
    return search, oof


def _key(name):
    """Convert a model name into a file name, e.g. "Random forest" -> "random_forest"."""
    return name.lower().replace(" ", "_")


def save(name, search, oof):
    """Store a fitted grid search and its out-of-fold probabilities.

    Files are written to MODEL_DIR so the models only need to be trained once.

    Parameters
    ----------
    name : str
        Model name as used in make_models(), e.g. "Random forest".
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search returned by tune().
    oof : numpy.ndarray
        Out-of-fold default probabilities returned by tune().

    Returns
    -------
    None
        Writes <name>.pkl (grid search) and <name>_oof.npy (probabilities).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if missing
    joblib.dump(search, MODEL_DIR / f"{_key(name)}.pkl") #write search as file
    np.save(MODEL_DIR / f"{_key(name)}_oof.npy", oof)    #write off as file


def load(name):
    """Load a stored grid search and its out-of-fold probabilities.

    Parameters
    ----------
    name : str
        Model name as used in make_models(), e.g. "Random forest".

    Returns
    -------
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search saved by save().
    oof : numpy.ndarray
        Out-of-fold default probabilities saved by save().
    """
    return (joblib.load(MODEL_DIR / f"{_key(name)}.pkl"),
            np.load(MODEL_DIR / f"{_key(name)}_oof.npy"))