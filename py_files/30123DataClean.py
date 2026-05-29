import pandas as pd

# READ ONLY FIRST 5 ROWS
df = pd.read_csv(
    "2023_combined_mlar.txt",
    sep="|",
    nrows=5,
    low_memory=False
)

# PRINT COLUMN NAMES
print("\nCOLUMN NAMES:\n")
print(df.columns.tolist())

# PRINT SAMPLE DATA
print("\nFIRST 5 ROWS:\n")
print(df.head())