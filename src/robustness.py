"""Robustness checks: bootstrap, recovery rate, DeLong, McNemar, class weighting (Section 5.4).

Checks whether the savings of the individualized rule are statistically robust
(Tables 10, 13), how they depend on the assumed recovery rate (Table 11), whether
classifiers differ significantly in AUC (Table 12), and whether cost-sensitive
training adds to cost-sensitive thresholds (Table 14).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import models
from src.calibration import Calibrator
from src.config import LAMBDAS, SEED
from src.data import costs
from src.decision_rules import ConventionalRule, IndividualizedRule
from src.evaluate import MODEL_ORDER, loss_per_loan


def bootstrap_savings(loss_conv, loss_ind, n_boot=1000, seed=SEED):
    """Paired bootstrap 95% percentile CI of the savings of one rule vs another.

    Parameters
    ----------
    loss_conv : numpy.ndarray
        Per-loan loss under the baseline (conventional) rule.
    loss_ind : numpy.ndarray
        Per-loan loss under the compared (individualized) rule, same loans and order.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Lower and upper bound (2.5th and 97.5th percentile) of the savings in %.
    """
    rng = np.random.default_rng(seed)  # random number generator
    n = len(loss_conv)
    savings = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # n positions from 0 to n-1, with repetition
        # same idx for both rules: paired resample of the test loans
        savings[b] = 100 * (1 - loss_ind[idx].sum() / loss_conv[idx].sum())
    return np.percentile(savings, [2.5, 97.5])


def table10(prepared, X_test, y_test):
    """Savings of the individualized vs conventional rule with bootstrap 95% CIs (Table 10).

    Parameters
    ----------
    prepared : dict
        Output of evaluate.prepare().
    X_test : pandas.DataFrame
        Test loans.
    y_test : array-like
        True default indicator of the test set.

    Returns
    -------
    pandas.DataFrame
        Per classifier: savings in % and the bounds of its 95% confidence interval.
    """
    C_FN, C_FP = costs(X_test)
    rows = []
    for name, d in prepared.items():
        loss_conv = loss_per_loan(y_test, ConventionalRule().decide(d["p"], X_test), C_FN, C_FP)
        loss_ind = loss_per_loan(y_test, IndividualizedRule().decide(d["p"], X_test), C_FN, C_FP)
        low, high = bootstrap_savings(loss_conv, loss_ind)
        rows.append({"Model": name,
                     "Savings (%)": 100 * (1 - loss_ind.sum() / loss_conv.sum()),
                     "CI low (%)": low,
                     "CI high (%)": high})
    return pd.DataFrame(rows).set_index("Model")


def table11(prepared, X_test, y_test, lambdas=LAMBDAS):
    """Savings of the individualized vs conventional rule for each lambda (Table 11).

    Lambda (share of principal not recovered) changes the cost of approving a
    defaulter and the individualized thresholds; the conventional decisions
    stay the same but their loss is priced with the same lambda.

    Parameters
    ----------
    prepared : dict
        Output of evaluate.prepare().
    X_test : pandas.DataFrame
        Test loans.
    y_test : array-like
        True default indicator of the test set.
    lambdas : list of float
        Recovery assumptions to evaluate.

    Returns
    -------
    pandas.DataFrame
        Savings in %, classifiers as rows and lambda values as columns.
    """
    rows = []
    for lam in lambdas:
        C_FN, C_FP = costs(X_test, lam)  # C_FN = lam * A changes with lambda
        for name, d in prepared.items():
            loss_conv = loss_per_loan(y_test, ConventionalRule().decide(d["p"], X_test), C_FN, C_FP)
            loss_ind = loss_per_loan(y_test, IndividualizedRule(lam).decide(d["p"], X_test), C_FN, C_FP)
            rows.append({"Model": name,
                         "lambda": lam,
                         "Savings (%)": 100 * (1 - loss_ind.sum() / loss_conv.sum())})
    table = pd.DataFrame(rows).pivot(index="Model", columns="lambda", values="Savings (%)")
    return table.loc[MODEL_ORDER]


def delong_test(y, p1, p2):
    """DeLong test for the difference in ROC-AUC of two classifiers on the same loans.

    Parameters
    ----------
    y : array-like
        True default indicator.
    p1, p2 : numpy.ndarray
        Predicted default probabilities of the two classifiers.

    Returns
    -------
    diff : float
        AUC of classifier 1 minus AUC of classifier 2.
    p_value : float
        Two-sided p-value of H0: equal AUCs.
    """
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()  # m defaulters, n non-defaulters
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        r_all = rankdata(p)  # rank among all loans
        r_pos = rankdata(p[pos])  # rank among defaulters
        r_neg = rankdata(p[neg])  # rank among non-defaulters
        # AUC = wins / pairs; wins = defaulters' rank sum minus 1 + ... + m = m(m+1)/2
        aucs.append((r_all[pos].sum() - m * (m + 1) / 2) / (m * n))
        # per defaulter: share of non-defaulters ranked below it
        v10.append((r_all[pos] - r_pos) / n)
        # per non-defaulter: share of defaulters ranked above it
        v01.append(1 - (r_all[neg] - r_neg) / m)
    s10, s01 = np.cov(v10), np.cov(v01)
    # Var(AUC1 - AUC2) = Var1 + Var2 - 2 Cov, over defaulters and non-defaulters
    var = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m
           + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n)
    diff = aucs[0] - aucs[1]
    return diff, 2 * norm.sf(abs(diff / np.sqrt(var)))


def table12(prepared, y_test):
    """Pairwise DeLong tests of ROC-AUC differences between classifiers (Table 12).

    Parameters
    ----------
    prepared : dict
        Output of evaluate.prepare().
    y_test : array-like
        True default indicator of the test set.

    Returns
    -------
    pandas.DataFrame
        One row per pair of classifiers: AUC difference and p-value.
    """
    rows = []
    for a, b in combinations(MODEL_ORDER, 2):  # all 6 pairs of models
        diff, p_value = delong_test(y_test, prepared[a]["p"], prepared[b]["p"])
        rows.append({"Comparison": f"{a} vs {b}",
                     "AUC difference": diff,
                     "p-value": p_value})
    return pd.DataFrame(rows).set_index("Comparison")


def mcnemar_test(y, pred_a, pred_b):
    """Exact McNemar test: do two sets of decisions differ in which loans they get right?

    Only loans classified correctly by exactly one rule (discordant loans) matter;
    under H0 each of them is equally likely to favour either rule.

    Parameters
    ----------
    y : array-like
        True default indicator.
    pred_a, pred_b : numpy.ndarray
        Decisions of the two rules (True/1 = reject).

    Returns
    -------
    only_a : int
        Loans correct under rule a only.
    only_b : int
        Loans correct under rule b only.
    p_value : float
        Two-sided exact binomial p-value.
    """
    y = np.asarray(y)
    correct_a = pred_a == y  # True where rule a decided correctly
    correct_b = pred_b == y
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a + only_b
    # under H0 the discordant loans split 50/50 between the two rules
    p_value = binomtest(only_a, n_discordant, 0.5).pvalue if n_discordant > 0 else 1.0
    return only_a, only_b, p_value


def table13(prepared, X_test, y_test):
    """McNemar tests: conventional vs individualized decisions per classifier (Table 13).

    Parameters
    ----------
    prepared : dict
        Output of evaluate.prepare().
    X_test : pandas.DataFrame
        Test loans.
    y_test : array-like
        True default indicator of the test set.

    Returns
    -------
    pandas.DataFrame
        Per classifier: loans correct only under each rule, and the p-value.
    """
    rows = []
    for name, d in prepared.items():
        conv = ConventionalRule().decide(d["p"], X_test)
        ind = IndividualizedRule().decide(d["p"], X_test)
        only_conv, only_ind, p_value = mcnemar_test(y_test, conv, ind)
        rows.append({"Model": name,
                     "Conventional only correct": only_conv,
                     "Individualized only correct": only_ind,
                     "p-value": p_value})
    return pd.DataFrame(rows).set_index("Model")


def weighted_xgboost(s):
    """Retrain XGBoost with a larger weight on defaulters and calibrate it (Section 4.3).

    Uses the tuned XGBoost hyperparameters, sets the defaulter weight to
    (number of non-defaulters) / (number of defaulters) in the fit set, and
    calibrates the probabilities the same way as the other classifiers.

    Parameters
    ----------
    s : dict
        Output of split_data().

    Returns
    -------
    p_test : numpy.ndarray
        Calibrated test probabilities of the weighted model.
    method : str
        Calibration method selected on C2.
    """
    X_fit, y_fit = s["fit"]
    X_c1, y_c1 = s["c1"]
    X_c2, y_c2 = s["c2"]
    X_test, _ = s["test"]

    search, _ = models.load("XGBoost")
    weight = (y_fit == 0).sum() / (y_fit == 1).sum()  # about 7.6
    # clone = untrained copy with the same tuned hyperparameters
    model = clone(search.best_estimator_).set_params(model__scale_pos_weight=weight)
    model.fit(X_fit, y_fit)

    # weighting inflates probabilities, so calibration is essential here
    cal = Calibrator().fit(model.predict_proba(X_c1)[:, 1], y_c1)
    cal.select(model.predict_proba(X_c2)[:, 1], y_c2)
    return cal.transform(model.predict_proba(X_test)[:, 1]), cal.method


def table14(prepared, s):
    """Class-weighted vs original XGBoost under the individualized rule (Table 14).

    Parameters
    ----------
    prepared : dict
        Output of evaluate.prepare().
    s : dict
        Output of split_data().

    Returns
    -------
    pandas.DataFrame
        ROC-AUC, Brier score, total loss and savings vs the original
        XGBoost with the conventional threshold.
    """
    X_test, y_test = s["test"]
    C_FN, C_FP = costs(X_test)
    p_orig = prepared["XGBoost"]["p"]
    p_weighted, method = weighted_xgboost(s)

    # baseline: original XGBoost with the 0.5 threshold
    base_loss = loss_per_loan(y_test, ConventionalRule().decide(p_orig, X_test), C_FN, C_FP).sum()

    configs = {
        "Original, conventional": (p_orig, ConventionalRule()),
        "Original, individualized": (p_orig, IndividualizedRule()),
        f"Weighted ({method}), individualized": (p_weighted, IndividualizedRule()),
    }
    rows = []
    for label, (p, rule) in configs.items():
        loss = loss_per_loan(y_test, rule.decide(p, X_test), C_FN, C_FP).sum()
        rows.append({"Configuration": label,
                     "ROC-AUC": roc_auc_score(y_test, p),
                     "Brier": brier_score_loss(y_test, p),
                     "Total loss (M)": loss / 1e6,
                     "Savings vs conventional (%)": 100 * (1 - loss / base_loss)})
    return pd.DataFrame(rows).set_index("Configuration")