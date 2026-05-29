import pandas as pd

#test with one year to examine data
tdf = pd.read_csv(
    "/Users/varshininarayanan/Downloads/hmda_data/2025_filtered.csv"
)

print(tdf.head())

print("\nColumns:")
print(tdf.columns)

print("\nShape:")
print(tdf.shape)

#find out what is missing
print("\nMissing Values:")
print(tdf.isnull().sum())


print(tdf["action_taken"].unique())
print(tdf["approved"].value_counts())
print(tdf["applicant_ethnicity_1"].value_counts()) #drop this fully, not needed
print(tdf["applicant_race_1"].value_counts())
print(tdf["interest_rate"].describe())

#found extreme incomes to be removed
print(tdf["income"].describe())

input_files = [
    "/Users/varshininarayanan/Downloads/hmda_data/2022_filtered.csv",
    "/Users/varshininarayanan/Downloads/hmda_data/2023_filtered.csv",
    "/Users/varshininarayanan/Downloads/hmda_data/2024_filtered.csv",
    "/Users/varshininarayanan/Downloads/hmda_data/2025_filtered.csv"
]

cleaned_dfs = []

for file in input_files:

    print(f"PROCESSING: {file}")

    df = pd.read_csv(
        file,
        low_memory=False
    )

    print(f"Original Shape: {df.shape}")

    if "applicant_ethnicity_1" in df.columns:

        df = df.drop(
            columns=["applicant_ethnicity_1"]
        )

    df = df.dropna(subset=[
        "county_code",
        "applicant_race_1",
        "income"
    ])

    numeric_cols = [
        "loan_amount",
        "income",
        "interest_rate",
        "property_value"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    #take care of negative income 
    df = df[
    (df["income"] >= 0) |
    (df["income"].isna())
    ]       

    # interest rates
    df = df[
    (df["interest_rate"] >= 0) |
    (df["interest_rate"].isna())
    ]

    # loans
    df = df[
    (df["loan_amount"] >= 0) |
    (df["loan_amount"].isna())
    ]


    print(f"Cleaned Shape: {df.shape}")

    cleaned_filename = file.replace(
        "_filtered.csv",
        "_cleaned.csv"
    )

    df.to_csv(
        cleaned_filename,
        index=False
    )

    print(f"Saved: {cleaned_filename}")

    cleaned_dfs.append(df)

#combine

combined_df = pd.concat(
    cleaned_dfs,
    ignore_index=True
)

combined_df.to_csv(
    "hmda_final.csv",
    index=False
)

print("FINAL DATASET CREATED")

print(f"Final Shape: {combined_df.shape}")

print("\nSaved:")
print("hmda_final.csv")
