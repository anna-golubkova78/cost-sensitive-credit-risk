# cost-sensitive-credit-risk
# Minimizing Expected Loss, Not Error: Cost-Sensitive Credit Risk Modeling

Loan-default classifiers are usually tuned for accuracy, treating a wrongly approved  defaulter and a wrongly rejected good borrower as equally costly. For a lender they are
not: the first loses the principal, the second only the interest. This project keeps four classifiers fixed and changes only the **decision rule**, comparing a conventional 0.5
threshold, an F1-optimal threshold, a global cost threshold and an **individualized expected-cost threshold** based on each loan's own interest rate and term.

## Key results (test set, 76,605 loans)

| | Logistic regression | Random forest | XGBoost | LightGBM |
|---|---|---|---|---|
| Loss, 0.5 threshold (millions) | 1257.9 | 1214.2 | 1233.4 | 1217.0 |
| Loss, individualized rule (millions) | 1096.9 | 1080.6 | 1074.8 | 1074.8 |
| Savings | 12.8% | 11.0% | 12.9% | 11.7% |

- The individualized rule lowers estimated loss by **11.0–12.9%** vs the 0.5 threshold,
  at a cost of **7.2–7.9 pp** accuracy, and by a further **8.6–10.4%** vs a single global cost threshold.
- Optimizing F1 instead *increases* losses for every classifier.
- Savings are significant (bootstrap 95% CIs above zero) and positive for all tested recovery rates.

## Method
1. **Data:** 255,347 loans, 11.6% default rate; 70/30 stratified train/test split.
2. **Classifiers:** penalized logistic regression, random forest, XGBoost, LightGBM, tuned by 5-fold stratified CV on log-loss.
3. **Calibration:** Platt scaling vs isotonic regression, fitted on C1 and selected by Brier score on C2.
4. **Decision rules:** reject loan *i* if its calibrated default probability is at least its threshold;
   the individualized threshold is r·T / (λ + r·T).
5. **Evaluation:** monetary loss, bootstrap CIs, recovery-rate sensitivity, DeLong and McNemar tests, segment analysis.

## Repository structure
```
src/
  config.py           settings: seed, splits, column lists
  data.py             loading, splits, per-loan costs
  eda.py              descriptive statistics (Tables 1–2)
  models.py           classifiers and CV tuning (Table 4)
  calibration.py      Calibrator class (Table 6)
  decision_rules.py   abstract DecisionRule + four rules
  evaluate.py         metrics and losses (Tables 3, 7–9)
  robustness.py       bootstrap, λ, DeLong, McNemar, class weighting (Tables 10–14)
  analysis.py         feature importance and segments (Tables 5, 15–17)
run.py                reproduces all tables into results/tables/
```

## How to run
1. Download the [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and save it as `data/Loan_default.csv`.
2. `pip install -r requirements.txt` (macOS: XGBoost also needs `brew install libomp`).
3. `python run.py` (first run trains the models, ~15 minutes; later runs take ~3 minutes).

## Notes
- The dataset appears to be synthetically generated, so the savings illustrate the
  framework rather than estimate what a real lender would achieve.
- This is a revised version of a group paper for *Topics in Business Intelligence*
  (Erasmus School of Economics). The code was rebuilt independently after the course with
  some methodological refinements, so some numbers differ slightly from the course version.

## Authors
Paper: Ollie Newell, Anna Golubkova, Vladimir Abdullin, Angelina Darbinyan.
Code: rebuilt by Anna Golubkova with AI assistance (Claude).

## License
MIT