"""Predictive and monetary evaluation of classifiers and decision rules (Sections 5.1 and 5.3).

For each classifier, test probabilities are calibrated and turned into decisions by
four rules (conventional, F1-optimal, global, individualized); the results are
summarized in Tables 3, 7, 8 and 9.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import models
from src.calibration import Calibrator
from src.data import costs
from src.decision_rules import ConventionalRule, F1Rule, GlobalRule, IndividualizedRule

MODEL_ORDER = list(models.make_models())  # list of model names
RULE_ORDER = [ConventionalRule.name, F1Rule.name, GlobalRule.name, IndividualizedRule.name]


def prepare(s):
    """Calibrated test probabilities and fitted F1 rule for each classifier.

    For each stored classifier: calibrate on C1/C2, apply the chosen
    calibration to the test probabilities, and fit the F1-optimal threshold
    on the calibrated out-of-fold probabilities of the fit set.

    Parameters
    ----------
    s : dict
        Output of split_data(), with keys "fit", "c1", "c2" and "test".

    Returns
    -------
    dict
        Model name -> {"p": calibrated test probabilities (numpy.ndarray),
        "f1_rule": fitted F1Rule}.
    """
    X_fit, y_fit = s["fit"]
    X_c1, y_c1 = s["c1"]
    X_c2, y_c2 = s["c2"]
    X_test, _ = s["test"]

    prepared = {}
    for name in MODEL_ORDER:
        search, oof = models.load(name)
        model = search.best_estimator_
        cal = Calibrator().fit(model.predict_proba(X_c1)[:, 1], y_c1)
        cal.select(model.predict_proba(X_c2)[:, 1], y_c2)
        prepared[name] = {
            "p": cal.transform(model.predict_proba(X_test)[:, 1]),
            "f1_rule": F1Rule().fit(cal.transform(oof), y_fit),
            "calibrator": cal,
        }
    return prepared


def classification_metrics(y, pred):
    """Classification metrics of 0/1 decisions (1 = reject, i.e. predicted default).

    Parameters
    ----------
    y : array-like
        True default indicator (1 = defaulted, 0 = repaid).
    pred : array-like
        Decisions (1 = reject, 0 = approve).

    Returns
    -------
    dict
        Accuracy, Precision, Recall, F1 and G-mean (Section 4.5).
    """
    recall_1 = recall_score(y, pred)  # TP / (TP + FN): share of defaulters that the rule caught
    recall_0 = recall_score(y, pred, pos_label=0)  # TN / (TN + FP): share of good borrowers approved
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_1,
            "F1": f1_score(y, pred, zero_division=0),
            "G-mean": np.sqrt(recall_1 * recall_0)}


def loss_per_loan(y, reject, C_FN, C_FP):
    """Monetary cost of each decision (eq. 17).

    Parameters
    ----------
    y : array-like
        True default indicator.
    reject : numpy.ndarray of bool
        Decisions (True = reject).
    C_FN : numpy.ndarray
        Cost of approving each loan if it defaults (lambda * A).
    C_FP : numpy.ndarray
        Cost of rejecting each loan if it would have been repaid (r * A * T).

    Returns
    -------
    numpy.ndarray
        Per-loan cost: C_FN for approved defaulters, C_FP for rejected
        loans that would have been repaid, 0 for correct decisions.
    """
    y = np.asarray(y)
    return np.where(~reject & (y == 1), C_FN, 0.0) + np.where(reject & (y == 0), C_FP, 0.0)


def table3(prepared, y_test):
    """Predictive performance of each classifier at the 0.5 threshold (Table 3).

    Parameters
    ----------
    prepared : dict
        Output of prepare().
    y_test : array-like
        True default indicator of the test set.

    Returns
    -------
    pandas.DataFrame
        One row per classifier: accuracy, precision, recall, F1, G-mean,
        ROC-AUC, PR-AUC and Brier score.
    """
    rows = []
    for name, d in prepared.items():
        p = d["p"]
        rows.append({"Model": name,
                     **classification_metrics(y_test, (p >= 0.5).astype(int)),  # accuracy, precision, recall, F1, G-mean
                     "ROC-AUC": roc_auc_score(y_test, p),
                     "PR-AUC": average_precision_score(y_test, p),
                     "Brier": brier_score_loss(y_test, p)})
    return pd.DataFrame(rows).set_index("Model")


def evaluate_rules(prepared, X_train, X_test, y_test, lam=1.0):
    """Monetary loss and accuracy of each classifier under each decision rule.

    Parameters
    ----------
    prepared : dict
        Output of prepare().
    X_train : pandas.DataFrame
        Training loans, used for the global threshold.
    X_test : pandas.DataFrame
        Test loans.
    y_test : array-like
        True default indicator of the test set.
    lam : float
        Share of principal not recovered on default.

    Returns
    -------
    pandas.DataFrame
        One row per classifier and rule (16 rows): Model, Rule,
        Loss (M) in millions, and Accuracy (%).
    """
    C_FN, C_FP = costs(X_test, lam)
    global_rule = GlobalRule(lam).fit(X_train)
    rows = []
    for name, d in prepared.items():
        rules = [ConventionalRule(), d["f1_rule"], global_rule, IndividualizedRule(lam)]
        for rule in rules:
            reject = rule.decide(d["p"], X_test)
            rows.append({"Model": name,
                         "Rule": rule.name,
                         "Loss (M)": loss_per_loan(y_test, reject, C_FN, C_FP).sum() / 1e6,
                         "Accuracy (%)": 100 * accuracy_score(y_test, reject.astype(int))})
    return pd.DataFrame(rows)


def table7(results):
    """Estimated loss in millions by classifier and decision rule (Table 7).

    Parameters
    ----------
    results : pandas.DataFrame
        Output of evaluate_rules().

    Returns
    -------
    pandas.DataFrame
        Classifiers as rows, rules as columns, in the paper's order.
    """
    return results.pivot(index="Model", columns="Rule", values="Loss (M)").loc[MODEL_ORDER, RULE_ORDER]


def table8(results):
    """Accuracy-cost trade-off of the individualized vs conventional rule (Table 8).

    Parameters
    ----------
    results : pandas.DataFrame
        Output of evaluate_rules().

    Returns
    -------
    pandas.DataFrame
        Per classifier: accuracy under both rules, change in percentage
        points, and loss savings in % relative to the conventional rule.
    """
    acc = results.pivot(index="Model", columns="Rule", values="Accuracy (%)").loc[MODEL_ORDER]
    loss = table7(results)
    conv, ind = ConventionalRule.name, IndividualizedRule.name
    return pd.DataFrame({"Conv. acc. (%)": acc[conv],
                         "Indiv. acc. (%)": acc[ind],
                         "Delta (pp)": acc[ind] - acc[conv],
                         "Savings (%)": 100 * (1 - loss[ind] / loss[conv])})


def table9(results):
    """Global vs individualized cost-sensitive rule (Table 9).

    Parameters
    ----------
    results : pandas.DataFrame
        Output of evaluate_rules().

    Returns
    -------
    pandas.DataFrame
        Per classifier: loss under both rules and the additional savings in %
        of the individualized rule relative to the global rule.
    """
    loss = table7(results)
    glob, ind = GlobalRule.name, IndividualizedRule.name
    return pd.DataFrame({"Global loss (M)": loss[glob],
                         "Individualized loss (M)": loss[ind],
                         "Additional savings (%)": 100 * (1 - loss[ind] / loss[glob])})