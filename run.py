from src.data import load_data, split_data, costs

from src.data import load_data
from src import eda

X, y = load_data()

print(X.shape)

s = split_data(X, y)
for name, (Xk, yk) in s.items():
    print(name, len(yk), round(yk.mean(), 4))

C_FN, C_FP = costs(s["train"][0])

print("interest / principal:", round((C_FP / C_FN).mean(), 3))
print(eda.quality_checks(X, y), "\n")
print(eda.table1(X, y).round(2), "\n")
print(eda.synthetic_checks(X), "\n")
print(eda.table2(X, y).round(3), "\n")
for col, rates in eda.default_rates(X, y).items():
    print(rates.round(4), "\n")