import pandas as pd
df = pd.read_csv("data/Loan_default.csv")
print(df.shape, df["Default"].mean())
