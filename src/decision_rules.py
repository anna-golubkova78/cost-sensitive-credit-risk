"""Decision rules turning default probabilities into reject/approve decisions (Section 4.2)."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.data import costs

class DecisionRule(ABC):
    """Base class: reject a loan if its default probability is at least its threshold."""

    name = "Rule"

    @abstractmethod
    def threshold(self, X):
        """Return the threshold: one number for all loans, or one per loan in X."""

    def decide(self, p, X):
        """Return True (reject) where p >= threshold, False (approve) otherwise."""
        return p >= self.threshold(X)


class ConventionalRule(DecisionRule):
    """Rule 1: threshold 0.5 for every loan."""

    name = "Conventional (0.5)"

    def threshold(self, X):
        return 0.5


class F1Rule(DecisionRule):
    """Rule 2: one threshold maximizing F1 on out-of-fold probabilities (eq. 13)."""

    name = "F1-optimal"

    def __init__(self):
        self.t = None

    def fit(self, p, y):
        """Choose the threshold that maximizes F1 for probabilities p and outcomes y."""
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        self.t = thresholds[np.argmax(f1[:-1])]
        return self

    def threshold(self, X):
        return self.t

class GlobalRule(DecisionRule):
    """Rule 3: one threshold from average training costs (eq. 14)."""

    name = "Global"

    def __init__(self, lam=1.0):
        self.lam = lam
        self.t = None

    def fit(self, X_train):
        """Compute the threshold from the mean misclassification costs in X_train."""
        C_FN, C_FP = costs(X_train, self.lam)
        self.t = C_FP.mean() / (C_FN.mean() + C_FP.mean())
        return self

    def threshold(self, X):
        return self.t

class IndividualizedRule(DecisionRule):
    """Rule 4: per-loan threshold from that loan's own costs (eq. 12)."""

    name = "Individualized"

    def __init__(self, lam=1.0):
        self.lam = lam

    def threshold(self, X):
        C_FN, C_FP = costs(X, self.lam)
        return C_FP / (C_FN + C_FP)