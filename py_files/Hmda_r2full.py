from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os



spark = SparkSession.builder \
    .appName("HMDA_County_Racial_Disparities_TX_CA") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


OUTPUT_DIR = "/home/varshi/finalHMDA/outputs"

FIG_DIR = "/home/varshi/finalHMDA/figures"

os.makedirs(f"{FIG_DIR}/maps", exist_ok=True)


df = spark.read.csv(
    "/home/varshi/hmda_final.csv",
    header=True,
    inferSchema=True
)


print("Rows Loaded:", df.count())

# cleaning

df = df.filter(col("approved").isin([0, 1]))

df = df.filter(
    col("county_code").isNotNull()
)

#racial groups

df = df.withColumn(
    "race_group",
    when(col("applicant_race_1") == 1, "Native")
    .when(col("applicant_race_1") == 3, "Black")
    .when(col("applicant_race_1") == 5, "White")
    .when(
        col("applicant_race_1").isin(
            2,21,22,23,24,25,26,27
        ),
        "Asian"
    )
    .otherwise("Other")
)

# target groups based on first regression results

df = df.withColumn(
    "comparison_group",
    when(
        col("race_group").isin(
            "White",
            "Asian"
        ),
        "Advantaged"
    )
    .when(
        col("race_group").isin(
            "Black",
            "Native"
        ),
        "Disadvantaged"
    )
    .otherwise(None)
)

df = df.filter(
    col("comparison_group").isNotNull()
)

# county

county_stats = df.groupBy(
    "state_code",
    "county_code"
).agg(

    F.count("*").alias(
        "total_applications"
    ),

    F.sum(
        when(
            col("comparison_group")
            == "Advantaged",
            1
        ).otherwise(0)
    ).alias(
        "advantaged_n"
    ),

    F.sum(
        when(
            col("comparison_group")
            == "Disadvantaged",
            1
        ).otherwise(0)
    ).alias(
        "disadvantaged_n"
    ),

    F.avg(
        when(
            col("comparison_group")
            == "Advantaged",
            col("approved")
        )
    ).alias(
        "advantaged_approval_rate"
    ),

    F.avg(
        when(
            col("comparison_group")
            == "Disadvantaged",
            col("approved")
        )
    ).alias(
        "disadvantaged_approval_rate"
    )

)

# filter out counties with low sample

county_stats = county_stats.filter(
    (col("advantaged_n") >= 50) &
    (col("disadvantaged_n") >= 50)
)

# aproval disparity calculation

county_stats = county_stats.withColumn(
    "approval_disparity_gap",
    col("advantaged_approval_rate")
    -
    col("disadvantaged_approval_rate")
)



county_stats = county_stats.orderBy(
    F.desc("approval_disparity_gap")
)



print("\nTop 25 most disparate counties\n")

county_stats.select(
    "state_code",
    "county_code",
    "advantaged_n",
    "disadvantaged_n",
    "advantaged_approval_rate",
    "disadvantaged_approval_rate",
    "approval_disparity_gap"
).show(
    25,
    truncate=False
)


county_pd = county_stats.toPandas()

county_pd.to_csv(
    f"{OUTPUT_DIR}/county_disparities_full.csv",
    index=False
)

county_pd.sort_values(
    "approval_disparity_gap",
    ascending=False
).head(25).to_csv(
    f"{OUTPUT_DIR}/top25_disparities_full.csv",
    index=False
)


# Bar plot viz

top25 = county_pd.sort_values(
    "approval_disparity_gap",
    ascending=False
).head(25)

top25["county_label"] = (
    top25["county_code"].astype(int).astype(str)
)

plt.figure(
    figsize=(12,8)
)

plt.barh(
    top25["county_label"],
    top25["approval_disparity_gap"]
)

plt.gca().invert_yaxis()

plt.title(
    "Top 25 Counties by Approval Gap"
)

plt.xlabel(
    "Approval Gap"
)

plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/maps/top25_disparities_full.png",
    dpi=300
)

plt.close()

print("DONE")

spark.stop()
