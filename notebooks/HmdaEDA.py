from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("HMDAProject") \
    .getOrCreate()


file_path = "/home/varshi/hdma/data/hmda_ca_tx_combined_cleaned.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("ROWS:", df.count())

print("SCHEMA:")
df.printSchema()

print("SAMPLE:")
df.show(5)

# feature engineering

#get loan to income ratio
df = df.withColumn(
    "loan_to_income",
    col("loan_amount") / col("income")
)

#fix race categories
df = df.withColumn(
    "race_group",

    when(
        col("applicant_race_1") == 1,
        "American Indian / Alaska Native"
    )

    .when(
        col("applicant_race_1").isin(
            2, 21, 22, 23, 24, 25, 26, 27
        ),
        "Asian"
    )

    .when(
        col("applicant_race_1") == 3,
        "Black"
    )

    .when(
        col("applicant_race_1").isin(
            4, 41, 42, 43, 44
        ),
        "Native Hawaiian / Pacific Islander"
    )

    .when(
        col("applicant_race_1") == 5,
        "White"
    )

    .when(
        col("applicant_race_1") == 6,
        "Not Provided"
    )

    .otherwise("Other")
)

# check to make sure approval is valid

print("Action taken count")

df.groupBy("action_taken") \
    .count() \
    .orderBy("action_taken") \
    .show()

print("approval count")

df.groupBy("approved") \
    .count() \
    .show()

#Plot approval rate by race

approval_by_race = (
    df.groupBy("race_group")
      .agg(
          avg("approved").alias("approval_rate"),
          count("*").alias("applications")
      )
      .filter(col("applications") > 1000)
      .orderBy(desc("approval_rate"))
)

approval_by_race.show()

approval_pd = approval_by_race.toPandas()

approval_pd.to_csv(
    "/home/varshi/hdma/outputs/approval_by_race.csv",
    index=False
)

plt.figure(figsize=(12, 7))

plt.barh(
    approval_pd["race_group"],
    approval_pd["approval_rate"]
)

plt.xlabel("Approval Rate")
plt.ylabel("Race Group")
plt.title("Mortgage Approval Rate by Race")

plt.xlim(0, 1)

for i, v in enumerate(approval_pd["approval_rate"]):
    plt.text(v + 0.01, i, f"{v:.2f}")

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/approval_by_race.png",
    dpi=300
)

plt.show()

# plot interest rate by race

interest_df = df.filter(
    col("interest_rate").isNotNull()
)

interest_by_race = (
    interest_df.groupBy("race_group")
               .agg(
                   avg("interest_rate")
                   .alias("avg_interest_rate"),

                   count("*").alias("loans")
               )
               .filter(col("loans") > 1000)
               .orderBy(desc("avg_interest_rate"))
)

interest_by_race.show()

interest_pd = interest_by_race.toPandas()

interest_pd.to_csv(
    "/home/varshi/hdma/outputs/interest_by_race.csv",
    index=False
)

plt.figure(figsize=(12, 7))

plt.barh(
    interest_pd["race_group"],
    interest_pd["avg_interest_rate"]
)

plt.xlabel("Average Interest Rate (%)")
plt.ylabel("Race Group")
plt.title("Average Mortgage Interest Rate by Race")

for i, v in enumerate(interest_pd["avg_interest_rate"]):
    plt.text(v + 0.02, i, f"{v:.2f}%")

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/interest_by_race.png",
    dpi=300
)

plt.show()

# plot approval rate over time

time_df = (
    df.groupBy("activity_year")
      .agg(
          avg("approved").alias("approval_rate")
      )
      .orderBy("activity_year")
)

time_pd = time_df.toPandas()

time_pd.to_csv(
    "/home/varshi/hdma/outputs/approval_over_time.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.plot(
    time_pd["activity_year"],
    time_pd["approval_rate"],
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Approval Rate")
plt.title("Mortgage Approval Rate Over Time")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/approval_over_time.png",
    dpi=300
)

plt.show()

# plot loan amount distribution

loan_pd = (
    df.select("loan_amount")
      .dropna()
      .toPandas()
)

plt.figure(figsize=(10, 6))

plt.hist(
    loan_pd["loan_amount"],
    bins=50
)

plt.xlabel("Loan Amount ($1,000s)")
plt.ylabel("Frequency")
plt.title("Distribution of Loan Amounts")

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/loan_amount_distribution.png",
    dpi=300
)

plt.show()

#plot income distribution

income_pd = (
    df.select("income")
      .dropna()
      .toPandas()
)

plt.figure(figsize=(10, 6))

plt.hist(
    income_pd["income"],
    bins=50
)

plt.xlabel("Income ($1,000s)")
plt.ylabel("Frequency")
plt.title("Distribution of Applicant Income")

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/income_distribution.png",
    dpi=300
)

plt.show()

# plot approval rate by loan

loan_type_df = (
    df.groupBy("loan_type")
      .agg(
          avg("approved").alias("approval_rate"),
          count("*").alias("applications")
      )
      .orderBy("loan_type")
)

loan_type_df.show()

loan_type_pd = loan_type_df.toPandas()

loan_type_pd.to_csv(
    "/home/varshi/hdma/outputs/loan_type_approval.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    loan_type_pd["loan_type"].astype(str),
    loan_type_pd["approval_rate"]
)

plt.xlabel("Loan Type")
plt.ylabel("Approval Rate")
plt.title("Approval Rate by Loan Type")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "/home/varshi/hdma/figures/loan_type_approval.png",
    dpi=300
)

plt.show()

# summary

summary_df = df.select(
    "loan_amount",
    "income",
    "interest_rate",
    "loan_to_income"
).summary()

summary_df.show()

summary_pd = summary_df.toPandas()

summary_pd.to_csv(
    "/home/varshi/hdma/outputs/summary_statistics.csv",
    index=False
)

# parq

output_path = "/home/varshi/hdma/outputs/final_hmda_dataset"

(
    df.write
      .mode("overwrite")
      .parquet(output_path)
)

print("Final dataset saved")



spark.stop()