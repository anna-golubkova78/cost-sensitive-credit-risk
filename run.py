import pandas as pd

from src.data import load_data, split_data
from src import evaluate, robustness

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

X, y = load_data()
s = split_data(X, y)
X_test, y_test = s["test"]
prepared = evaluate.prepare(s)

print(robustness.table10(prepared, X_test, y_test).round(2), "\n")
print(robustness.table11(prepared, X_test, y_test).round(2), "\n")
print(robustness.table12(prepared, y_test).round(4), "\n")
print(robustness.table13(prepared, X_test, y_test), "\n")
print(robustness.table14(prepared, s).round(4))