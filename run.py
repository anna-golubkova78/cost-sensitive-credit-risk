from src.data import load_data, split_data
from src import models

X, y = load_data()
X_fit, y_fit = split_data(X, y)["fit"]

for name, spec in models.make_models().items():
    print(f"\nTuning {name}")
    search, oof = models.tune(spec, X_fit, y_fit)
    models.save(name, search, oof)
    print(name, search.best_params_, round(-search.best_score_, 6))