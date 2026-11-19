"""Feature importance and failure analysis by borrower segment (Sections 5.1 and 5.5).

Table 5 shows which variables drive the predictions; Tables 15-17 show where
the remaining losses of the individualized rule are concentrated.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import SEED
from src.data import costs
from src.decision_rules import ConventionalRule, IndividualizedRule
from src.evaluate import loss_per_loan


def permutation_table(search, X_test, y_test, n_repeats=5):
    """Permutation importance: mean drop in test ROC-AUC when one variable is shuffled.

    Parameters
    ----------
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search from models.load().
    X_test : pandas.DataFrame
        Test loans (raw columns).
    y_test : array-like
        True default indicator of the test set.
    n_repeats : int
        Number of shuffles per variable.

    Returns
    -------
    pandas.DataFrame
        Mean and standard deviation of the AUC drop per variable, largest first.
    """
    # the pipeline takes raw columns, so each original variable is shuffled as a whole
    result = permutation_importance(search.best_estimator_, X_test, y_test,
                                    scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=SEED, n_jobs=-1)
    table = pd.DataFrame({"Importance": result.importances_mean,
                          "Std. dev.": result.importances_std},
                         index=X_test.columns)
    return table.sort_values("Importance", ascending=False)


def gain_table(search):
    """Total split gain per original variable for a boosted model (XGBoost or LightGBM).

    Gains of the one-hot columns of a categorical variable are summed, so the
    result has one value per original variable.

    Parameters
    ----------
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search of XGBoost or LightGBM from models.load().

    Returns
    -------
    pandas.Series
        Total gain per variable, largest first (scale is model-specific).
    """
    pipe = search.best_estimator_
    names = pipe.named_steps["prep"].get_feature_names_out()  # e.g. "num__Age", "cat__HasCoSigner_Yes"
    variables = [n.split("__", 1)[1].split("_", 1)[0] for n in names]  # -> "Age", "HasCoSigner"
    model = pipe.named_steps["model"]

    if hasattr(model, "get_booster"):  # XGBoost: gains keyed by "f0", "f1", ...
        scores = model.get_booster().get_score(importance_type="total_gain")
        gain = [scores.get(f"f{i}", scores.get(name, 0.0)) for i, name in enumerate(names)]
    else:  # LightGBM: one gain per column, in column order
        gain = model.booster_.feature_importance(importance_type="gain")

    return pd.Series(gain, index=variables).groupby(level=0).sum().sort_values(ascending=False)


def table5(perm, gain_xgb, gain_lgbm, top=5):
    """Top predictors by permutation importance, with total gain of both boosted models (Table 5).

    Parameters
    ----------
    perm : pandas.DataFrame
        Output of permutation_table() for XGBoost.
    gain_xgb, gain_lgbm : pandas.Series
        Output of gain_table() for XGBoost and LightGBM.
    top : int
        Number of variables to show.

    Returns
    -------
    pandas.DataFrame
        Importance, its standard deviation, and total gain per model.
    """
    table = perm.head(top).copy()
    table["Gain XGBoost"] = gain_xgb.reindex(table.index)  # align by variable name
    table["Gain LightGBM"] = gain_lgbm.reindex(table.index)
    return table


def segment_table(groups, y_test, loss_conv, loss_ind, mean_of=None, mean_name=None):
    """Loans, default rate, losses and savings per borrower segment.

    Parameters
    ----------
    groups : array-like
        Segment label of each test loan.
    y_test : array-like
        True default indicator of the test set.
    loss_conv, loss_ind : numpy.ndarray
        Per-loan loss under the conventional and individualized rule.
    mean_of : array-like, optional
        Variable whose mean per segment is reported (e.g. loan amount).
    mean_name : str, optional
        Column name for that mean.

    Returns
    -------
    pandas.DataFrame
        One row per segment.
    """
    df = pd.DataFrame({"group": np.asarray(groups),
                       "default": np.asarray(y_test),
                       "conv": loss_conv,
                       "ind": loss_ind})
    if mean_of is not None:
        df["mean"] = np.asarray(mean_of)

    g = df.groupby("group")  # one group per segment
    table = pd.DataFrame({"Loans": g.size(),
                          "Default rate (%)": 100 * g["default"].mean(),
                          "Conventional (M)": g["conv"].sum() / 1e6,
                          "Individualized (M)": g["ind"].sum() / 1e6})
    if mean_of is not None:
        table.insert(1, mean_name, g["mean"].mean())  # as second column
    table["Savings (%)"] = 100 * (1 - table["Individualized (M)"] / table["Conventional (M)"])
    return table


def _xgb_losses(prepared, X_test, y_test):
    """Per-loan losses of XGBoost under the conventional and individualized rule."""
    C_FN, C_FP = costs(X_test)
    p = prepared["XGBoost"]["p"]
    return (loss_per_loan(y_test, ConventionalRule().decide(p, X_test), C_FN, C_FP),
            loss_per_loan(y_test, IndividualizedRule().decide(p, X_test), C_FN, C_FP))


def table15(prepared, X_test, y_test):
    """XGBoost losses by employment type (Table 15)."""
    conv, ind = _xgb_losses(prepared, X_test, y_test)
    return segment_table(X_test["EmploymentType"], y_test, conv, ind)


def table16(prepared, X_test, y_test):
    """XGBoost losses by loan-amount quartile (Table 16)."""
    conv, ind = _xgb_losses(prepared, X_test, y_test)
    quartiles = pd.qcut(X_test["LoanAmount"], 4,
                        labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])
    return segment_table(quartiles, y_test, conv, ind, X_test["LoanAmount"], "Mean loan")


def table17(prepared, X_test, y_test):
    """XGBoost losses by interest-rate quartile (Table 17)."""
    conv, ind = _xgb_losses(prepared, X_test, y_test)
    quartiles = pd.qcut(X_test["InterestRate"], 4,
                        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    return segment_table(quartiles, y_test, conv, ind, X_test["InterestRate"], "Mean rate (%)")