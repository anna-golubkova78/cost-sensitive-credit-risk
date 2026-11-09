"""Exploratory data checks reproducing Section 3 (Tables 1 and 2)."""

import numpy as np
import pandas as pd
from src.config import NUMERICAL, CATEGORICAL

def quality_checks(X, y):
    """Count missing values and duplicate rows."""
    df = X.assign(Default=y)
    return {"missing values": int(df.isna().sum().sum()),
            "duplicate values": int(df.duplicated().sum())}

def table1(X,y):
    """Mean, minimum and maximum of each numerical variable (Table 1)."""
    t = X[NUMERICAL].agg(["mean", "min", "max"]).T
    t.loc["Default"] = [y.mean(), y.min(), y.max()]
    return t

def synthetic_checks(X):
    """Evidence of synthetic data: correlations, mean vs median, category sizes."""
    corr = X[NUMERICAL].corr().abs()
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    skew = (X[NUMERICAL].mean() - X[NUMERICAL].median()) / X[NUMERICAL].std()
    sizes = {c: (int(X[c].value_counts().min()), int(X[c].value_counts().max()))
             for c in CATEGORICAL}
    return {"max |corr| between predictors": float(off_diag.max().max()),
            "corr(InterestRate, LoanTerm)": float(X["InterestRate"].corr(X["LoanTerm"])),
            "(mean - median) / sd": skew.round(3).to_dict(),
            "category sizes (min, max)": sizes}

def table2(X,y):
    """Means by default status, standardized mean difference and correlation with Default (Table 2)."""
    rows = []
    for col in NUMERICAL:
        x0 = X.loc[y==0, col]
        x1 = X.loc[y==1, col]
        n0, n1 = len(x0), len(x1)
        sp = np.sqrt(((n0 - 1) * x0.var() + (n1 - 1) * x1.var()) / (n0 + n1 - 2))
        rows.append({"Variable": col,
                     "No default": x0.mean(),
                     "Default": x1.mean(),
                     "SMD": (x1.mean() - x0.mean()) / sp,
                     "rho_D": X[col].corr(y)})
    return pd.DataFrame(rows).set_index("Variable")

def default_rates(X, y):
    """Default rate per category of each categorical variable."""
    return{col: y.groupby(X[col]).mean().sort_values() for col in CATEGORICAL}