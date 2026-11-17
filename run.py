import pandas as pd

from src.data import load_data, split_data
from src import evaluate

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

X, y = load_data()
s = split_data(X, y)
X_train, _ = s["train"]
X_test, y_test = s["test"]

prepared = evaluate.prepare(s)
print(evaluate.table3(prepared, y_test).round(4), "\n")

results = evaluate.evaluate_rules(prepared, X_train, X_test, y_test)
print(evaluate.table7(results).round(2), "\n")
print(evaluate.table8(results).round(2), "\n")
print(evaluate.table9(results).round(2))