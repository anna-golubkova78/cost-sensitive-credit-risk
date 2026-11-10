from src.data import load_data, split_data
from src import models
from src.calibration import Calibrator
from src.decision_rules import F1Rule, GlobalRule, IndividualizedRule

X, y = load_data()
s = split_data(X, y)
X_train, _ = s["train"]
X_fit, y_fit = s["fit"]
X_c1, y_c1 = s["c1"]
X_c2, y_c2 = s["c2"]
X_test, y_test = s["test"]

glob = GlobalRule().fit(X_train)
t_ind = IndividualizedRule().threshold(X_test)
print("Global threshold:", round(glob.t, 4))
print("Individualized: min", round(t_ind.min(), 4),
      "mean", round(t_ind.mean(), 4), "max", round(t_ind.max(), 4))

for name in models.make_models():
    search, oof = models.load(name)
    model = search.best_estimator_
    cal = Calibrator().fit(model.predict_proba(X_c1)[:, 1], y_c1)
    cal.select(model.predict_proba(X_c2)[:, 1], y_c2)
    f1 = F1Rule().fit(cal.transform(oof), y_fit)
    print(name, "F1 threshold:", round(f1.t, 4))