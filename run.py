from src.data import load_data, split_data
from src import models
from src.calibration import Calibrator

X, y = load_data()
s = split_data(X, y)
X_c1, y_c1 = s["c1"]
X_c2, y_c2 = s["c2"]

for name in models.make_models():
    search, _ = models.load(name)
    model = search.best_estimator_
    p_c1 = model.predict_proba(X_c1)[:, 1]
    p_c2 = model.predict_proba(X_c2)[:, 1]
    cal = Calibrator().fit(p_c1, y_c1).select(p_c2, y_c2)
    print(name, {k: round(v, 6) for k, v in cal.brier.items()}, "->", cal.method)