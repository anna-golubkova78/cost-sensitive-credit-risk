"""Data loading, train/calibration/test splits and per-loan costs (Sections 3 and 4.2)."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SEED, DATA_PATH, TEST_SIZE, CAL_SIZE

def load_data(path=DATA_PATH):
    """Load the loan data and separate predictors from the target.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    X : pandas.DataFrame
        The 16 predictors (LoanID and Default removed).
    y : pandas.Series
        Default indicator (1 = defaulted, 0 = repaid).
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["LoanID", "Default"])
    y = df["Default"]
    return X, y

def split_data(X, y):
    """Split the data into fit, calibration (C1, C2) and test sets.

    70/30 train/test split; 20% of train becomes the calibration set C,
    which is halved into C1 and C2. All splits are stratified on y.

    Returns
    -------
    dict
        Keys "train", "fit", "c1", "c2", "test"; each value is an (X, y) pair.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=CAL_SIZE, stratify=y_train, random_state=SEED)
    X_c1, X_c2, y_c1, y_c2 = train_test_split(
        X_cal, y_cal, test_size=0.5, stratify=y_cal, random_state=SEED)
    return {"train": (X_train, y_train), "fit": (X_fit, y_fit), "c1": (X_c1, y_c1), "c2": (X_c2, y_c2), "test": (X_test, y_test)}

def costs(X, lam=1.0):
    """Per-loan misclassification costs (Section 4.2).

    Parameters
    ----------
    X : pandas.DataFrame
        Loans with LoanAmount, InterestRate (%) and LoanTerm (months).
    lam : float
        Share of principal not recovered on default.

    Returns
    -------
    C_FN : numpy.ndarray
        Cost of approving a defaulter: lam * A.
    C_FP : numpy.ndarray
        Cost of rejecting a good borrower: r * A * T.
    """
    A = X["LoanAmount"].to_numpy()
    r = X["InterestRate"].to_numpy() / 100
    T = X["LoanTerm"].to_numpy() / 12
    return lam * A, r * A * T    #C_FN, C_FP





