"""Global settings: seed, paths, split sizes, recovery rates and column lists."""

SEED = 42
DATA_PATH = "data/Loan_default.csv"
TEST_SIZE = 0.30   # share of all loans in the test set
CAL_SIZE = 0.20     # share of the training set used for calibration (C)
LAMBDAS = [0.25, 0.5, 0.75, 1.0]

NUMERICAL = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
             "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
CATEGORICAL = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage",
               "HasDependents", "LoanPurpose", "HasCoSigner"]