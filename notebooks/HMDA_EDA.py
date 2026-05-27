from pyspark.sql import SparkSession
from pyspark.sql.functions import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("HMDAProject") \
    .getOrCreate()

df = spark.read.csv(
    "/home/varshi/hdma/data/hmda_ca_tx_combined_cleaned.csv",
    header=True,
    inferSchema=True
)

print("ROW COUNT:")
print(df.count())

print("\nCOLUMN COUNT:")
print(len(df.columns))

df.printSchema()

df.show(5)

#test sample
test_df = df.sample(
    withReplacement=False,
    fraction=0.02,
    seed=42
)

print("TEST SAMPLE ROW COUNT:")
print(test_df.count())

df.write.mode("overwrite").parquet(
    "/home/varshi/hdma/data/hdma_parquet"
)

df = spark.read.parquet(
    "/home/varshi/hdma/data/hdma_parquet"
)

#approval rates by race 
approval_rates = df.groupBy(
    "applicant_race_1"
).agg(
    avg("approved").alias("approval_rate")
)

approval_pd = approval_rates.toPandas()

approval_pd



#approval rate visualization
plt.figure(figsize=(8,5))

sns.barplot(
    data=approval_pd,
    x="applicant_race_1",
    y="approval_rate"
)

plt.title("Mortgage Approval Rates by Race")
plt.xlabel("Race Category")
plt.ylabel("Approval Rate")

plt.savefig("/home/varshi/hdma/figures/approval_by_race.png")

plt.show()

#interest rate by race 
interest_rates = df.groupBy(
    "applicant_race_1"
).agg(
    avg("interest_rate").alias("avg_interest_rate")
)

interest_pd = interest_rates.toPandas()

#plot interest rate by race
plt.figure(figsize=(8,5))

sns.barplot(
    data=interest_pd,
    x="applicant_race_1",
    y="avg_interest_rate"
)

plt.title("Average Interest Rates by Race")
plt.xlabel("Race Category")
plt.ylabel("Average Interest Rate")

plt.savefig("/home/varshi/hdma/figures/interest_by_race.png")

plt.show()

#income distribution and visualization
income_sample = test_df.select(
    "income"
).toPandas()


plt.figure(figsize=(10,6))

sns.histplot(
    income_sample["income"],
    bins=50
)

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")

plt.savefig("/home/varshi/hdma/figures/income_distribution.png")

plt.show()

#approval rates over time
year_approval = df.groupBy(
    "activity_year"
).agg(
    avg("approved").alias("approval_rate")
)

year_pd = year_approval.toPandas()

#plot app rates
plt.figure(figsize=(8,5))

sns.lineplot(
    data=year_pd,
    x="activity_year",
    y="approval_rate",
    marker="o"
)

plt.title("Approval Rates Over Time")
plt.xlabel("Year")
plt.ylabel("Approval Rate")

plt.savefig("/home/varshi/hdma/figures/approval_over_time.png")

plt.show()

#load amount distribution
loan_sample = test_df.select(
    "loan_amount"
).toPandas()

plt.figure(figsize=(10,6))

sns.histplot(
    loan_sample["loan_amount"],
    bins=50
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.savefig("/home/varshi/hdma/figures/loan_amount_distribution.png")

plt.show()

#loan types and approvals
loan_type_rates = df.groupBy(
    "loan_type"
).agg(
    avg("approved").alias("approval_rate")
)

loan_type_pd = loan_type_rates.toPandas()

#visualize 
plt.figure(figsize=(8,5))

sns.barplot(
    data=loan_type_pd,
    x="loan_type",
    y="approval_rate"
)

plt.title("Approval Rates by Loan Type")

plt.xlabel("Loan Type")

plt.ylabel("Approval Rate")

plt.savefig(
    "/home/varshi/hdma/figures/loan_type_approval.png"
)

plt.show()

#loan to income ratio
df = df.withColumn(
    "loan_to_income",
    col("loan_amount") / col("income")
)

df.select(
    "loan_to_income"
).describe().show()

df.write.mode("overwrite").parquet(
    "/home/varshi/hdma/outputs/final_hmda_dataset"
)
