from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, DoubleType

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

import math


spark = SparkSession.builder \
    .appName("HMDA_R3_Interest_Rate_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


file_path = "/home/varshi/hmda_final.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Original Rows:", df.count())

#only approved loans

df = df.filter(
    col("approved") == 1 
)
df = df.filter(
    col("interest_rate").isNotNull()
)

df = df.filter(
    (col("interest_rate") > 0) &
    (col("interest_rate") < 25)
)



df = df.filter(
    col("county_code").isNotNull()
)

df = df.na.fill({
    "income": 0.0,
    "loan_amount": 0.0,
    "loan_type": -1
})

# loan to income

df = df.withColumn(
    "loan_to_income",
    when(
        (col("income") > 0) &
        (col("loan_amount") > 0),
        col("loan_amount") / col("income")
    ).otherwise(0.0)
)

# Race categories

df = df.withColumn(
    "race_group",
    when(col("applicant_race_1") == 1,
         "Native American")
    .when(col("applicant_race_1") == 3,
          "Black")
    .when(col("applicant_race_1") == 5,
          "White")
    .when(
        col("applicant_race_1").isin(
            2,21,22,23,24,25,26,27
        ),
        "Asian"
    )
    .otherwise("Other")
)


#sample
df = df.sample(
    withReplacement=False,
    fraction=0.10,
    seed=42
)

print("Test Sample Rows:", df.count())


categorical_cols = [
    "race_group",
    "loan_type",
    "county_code",
    "activity_year"
]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_vec"
    )
    for c in categorical_cols
]


feature_cols = [
    "income",
    "loan_amount",
    "loan_to_income"
] + [
    c + "_vec"
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)



lr = LinearRegression(
    featuresCol="features",
    labelCol="interest_rate",
    maxIter=20,
    regParam=0.01
)

pipeline = Pipeline(
    stages=indexers +
           encoders +
           [assembler, lr]
)

model = pipeline.fit(df)

lr_model = model.stages[-1]



print("Intercept:")
print(lr_model.intercept)

print("R2:")
print(lr_model.summary.r2)

print("RMSE:")
print(lr_model.summary.rootMeanSquaredError)



transformed_df = model.transform(df)

meta = transformed_df.schema[
    "features"
].metadata

attrs_summary = meta["ml_attr"]["attrs"]

flat_feature_names = [
    "income",
    "loan_amount",
    "loan_to_income"
]

for attr_type in [
    "numeric",
    "binary",
    "nominal"
]:
    if attr_type in attrs_summary:
        for attr in attrs_summary[attr_type]:
            if attr["name"] not in flat_feature_names:
                flat_feature_names.append(
                    attr["name"]
                )

coef_values = [
    float(x)
    for x in lr_model.coefficients
]

if len(flat_feature_names) < len(coef_values):
    flat_feature_names += [
        f"unknown_{i}"
        for i in range(
            len(coef_values)
            -
            len(flat_feature_names)
        )
    ]



coef_rows = []

for name, coef in zip(
    flat_feature_names,
    coef_values
):
    coef_rows.append(
        (
            name,
            float(coef)
        )
    )

coef_schema = StructType([
    StructField(
        "feature",
        StringType(),
        True
    ),
    StructField(
        "coefficient",
        DoubleType(),
        True
    )
])

coef_df = spark.createDataFrame(
    coef_rows,
    coef_schema
)

coef_df.show(
    100,
    truncate=False
)



coef_df.toPandas().to_csv(
    "/home/varshi/finalHMDA/outputs/r3_interest_coefficients.csv",
    index=False
)

# save predictions

transformed_df.select(
    "interest_rate",
    "prediction",
    "race_group",
    "income",
    "loan_amount",
    "county_code"
).toPandas().to_csv(
    "/home/varshi/finalHMDA/outputs/r3_predictions.csv",
    index=False
)

print("DONE")

spark.stop()
