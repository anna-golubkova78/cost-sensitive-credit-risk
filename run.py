"""Reproduce all tables of the paper and save them to results/tables.

Usage: python run.py
Models are trained on the first run and loaded from results/models afterwards.
"""

from pathlib import Path

import pandas as pd

from src import analysis, eda, evaluate, models, robustness
from src.data import load_data, split_data

TABLE_DIR = Path("results/tables")


def save(table, name):
    """Print a table and save it as CSV in results/tables."""
    print(f"\n{name}\n{table.round(4)}")
    table.to_csv(TABLE_DIR / f"{name}.csv")


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    s = split_data(X, y)
    X_train, _ = s["train"]
    X_fit, y_fit = s["fit"]
    X_test, y_test = s["test"]

    # Section 3: data
    save(eda.table1(X, y), "table1_descriptives")
    save(eda.table2(X, y), "table2_by_default")

    # Section 4.1: train each model once, then reuse the saved version
    for name, spec in models.make_models().items():
        try:
            models.load(name)
        except FileNotFoundError:
            print(f"Training {name} (first run only) ...")
            search, oof = models.tune(spec, X_fit, y_fit)
            models.save(name, search, oof)

    table4 = pd.DataFrame(
        {name: {"Selected": models.load(name)[0].best_params_,
                "CV log-loss": -models.load(name)[0].best_score_}
         for name in evaluate.MODEL_ORDER}).T
    save(table4, "table4_hyperparameters")

    # Sections 5.1-5.3: calibration, predictive performance, losses
    prepared = evaluate.prepare(s)
    table6 = pd.DataFrame(
        {name: {**d["calibrator"].brier, "Selected": d["calibrator"].method}
         for name, d in prepared.items()}).T
    save(table6, "table6_calibration")

    results = evaluate.evaluate_rules(prepared, X_train, X_test, y_test)
    save(evaluate.table3(prepared, y_test), "table3_performance")
    save(evaluate.table7(results), "table7_loss")
    save(evaluate.table8(results), "table8_accuracy_vs_savings")
    save(evaluate.table9(results), "table9_global_vs_individualized")

    # Section 5.4: robustness
    save(robustness.table10(prepared, X_test, y_test), "table10_bootstrap")
    save(robustness.table11(prepared, X_test, y_test), "table11_lambda")
    save(robustness.table12(prepared, y_test), "table12_delong")
    save(robustness.table13(prepared, X_test, y_test), "table13_mcnemar")
    save(robustness.table14(prepared, s), "table14_weighted_xgboost")

    # Sections 5.1 and 5.5: feature importance and segments
    xgb, _ = models.load("XGBoost")
    lgbm, _ = models.load("LightGBM")
    perm = analysis.permutation_table(xgb, X_test, y_test)
    save(analysis.table5(perm, analysis.gain_table(xgb), analysis.gain_table(lgbm)), "table5_importance")
    save(analysis.table15(prepared, X_test, y_test), "table15_employment")
    save(analysis.table16(prepared, X_test, y_test), "table16_loan_size")
    save(analysis.table17(prepared, X_test, y_test), "table17_interest_rate")


if __name__ == "__main__":
    main()