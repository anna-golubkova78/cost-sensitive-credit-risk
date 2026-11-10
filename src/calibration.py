"""Probability calibration: Platt scaling vs isotonic regression (Section 4.4).

For each classifier, both calibration maps are fitted on C1 and the method
with the lowest Brier score on C2 (original, Platt or isotonic) is selected.
"""
import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-10


def _log_odds(p):
    """Convert probabilities to log-odds for Platt scaling.

    Parameters
    ----------
    p : numpy.ndarray
        Predicted default probabilities.

    Returns
    -------
    numpy.ndarray
        Column vector of log(p / (1 - p)), with p clipped to [EPS, 1 - EPS].
    """
    return logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)  # map p in interval, apply logit and reshape in table


class Calibrator:
    """Choose and apply a calibration map for one classifier (Section 4.4).

    Attributes
    ----------
    platt : sklearn.linear_model.LogisticRegression
        Unpenalized logit of the outcome on the classifier's log-odds (eq. 15).
    isotonic : sklearn.isotonic.IsotonicRegression
        Non-decreasing step function of the classifier's probability.
    method : str or None
        Selected method: "Original", "Platt" or "Isotonic" (None before select).
    brier : dict
        Brier score on C2 for each method (eq. 16).
    """

    def __init__(self):
        """Create unfitted Platt and isotonic calibrators."""
        self.platt = LogisticRegression(C=np.inf)  # plain max likelihood: regression with penalty power 0 = 1/inf
        self.isotonic = IsotonicRegression(out_of_bounds="clip") # outside seen range: use nearest end
        self.method = None
        self.brier = {}  # empty dictionary for scores of calibrators

    def fit(self, p_c1, y_c1):
        """Fit Platt scaling and isotonic regression on C1.

        Parameters
        ----------
        p_c1 : numpy.ndarray
            Classifier's predicted default probabilities for C1.
        y_c1 : pandas.Series
            Default indicator for C1.

        Returns
        -------
        Calibrator
            The fitted calibrator (self), so calls can be chained.
        """
        self.platt.fit(_log_odds(p_c1), y_c1)
        self.isotonic.fit(p_c1, y_c1)
        return self

    def _apply(self, p, method):
        """Map probabilities with a given calibration method.

        Parameters
        ----------
        p : numpy.ndarray
            Predicted default probabilities.
        method : str
            "Platt", "Isotonic", or "Original" (no calibration).

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities.
        """
        if method == "Platt":
            return self.platt.predict_proba(_log_odds(p))[:, 1]
        if method == "Isotonic":
            return self.isotonic.predict(p)
        return p

    def select(self, p_c2, y_c2):
        """Select the calibration method with the lowest Brier score on C2.

        Parameters
        ----------
        p_c2 : numpy.ndarray
            Classifier's predicted default probabilities for C2.
        y_c2 : pandas.Series
            Default indicator for C2.

        Returns
        -------
        Calibrator
            The calibrator (self) with method and brier set.
        """
        for method in ["Original", "Platt", "Isotonic"]:
            self.brier[method] = brier_score_loss(y_c2, self._apply(p_c2, method))
        self.method = min(self.brier, key=self.brier.get)  # single method with the lowest Brier score selected
        return self

    def transform(self, p):
        """Apply the selected calibration to new probabilities.

        Parameters
        ----------
        p : numpy.ndarray
            Predicted default probabilities (e.g. for the test set).

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities.
        """
        return self._apply(p, self.method)