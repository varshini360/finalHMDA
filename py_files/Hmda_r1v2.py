from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("HMDA_R1_Approval_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = "/home/varshi/hmda_final.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# scaling
df = df.sample(
    withReplacement=False,
    fraction=0.10,
    seed=42
)

print("Sample Rows:", df.count())

print("Original Rows:", df.count())

#deal with target and structural tracking nulls
df = df.filter(col("approved").isin([0, 1]))

df = df.filter(
    (col("activity_year").isNotNull()) &
    (col("county_code").isNotNull())
)

# Missing loan amounts filled with 0.0
df = df.na.fill({
    "income": 0.0,
    "loan_amount": 0.0,
    "loan_type": -1,        #placeholder num for missing types
})

df = df.filter(
    (col("income") >= 0) &
    (col("loan_amount") >= 0)
)

# Calculate loan_to_income
df = df.withColumn(
    "loan_to_income",
    when(
        (col("income") > 0) & (col("loan_amount") > 0),
        col("loan_amount") / col("income")
    )
    .otherwise(0.0) 
)

# Simplify racial groups
df = df.withColumn(
    "race_group",
    when(col("applicant_race_1") == 1, "Native American")
    .when(col("applicant_race_1") == 3, "Black")
    .when(col("applicant_race_1") == 5, "White")
    .when(col("applicant_race_1").isin(2, 21, 22, 23, 24, 25, 26, 27), "Asian")
    .otherwise("Other")
)

df = df.cache()
print("Final Regression Rows:", df.count())

# Categorical columns
categorical_cols = [
    "race_group",
    "loan_type",
    "county_code",
    "activity_year"
] 

# handleInvalid="keep" maps any missing categorical variables to their own specific index
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
] + [c + "_vec" for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# log reg
lr = LogisticRegression(
    featuresCol="features",
    labelCol="approved",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0
) 

#Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
model = pipeline.fit(df)

lr_model = model.stages[-1]

print("Intercept:")
print(lr_model.intercept)

print("Coefficients Summary:")
print(lr_model.coefficients)

# Transform the df to  get the feature schema metadata
transformed_df = model.transform(df)
meta = transformed_df.schema["features"].metadata
attrs_summary = meta["ml_attr"]["attrs"]

flat_feature_names = ["income", "loan_amount", "loan_to_income"]

# Pull dynamically generated names from OneHotEncoding
for attr_type in ["numeric", "binary", "nominal"]:
    if attr_type in attrs_summary:
        for attr in attrs_summary[attr_type]:
            # Avoid duplicating base numeric columns
            if attr["name"] not in flat_feature_names:
                flat_feature_names.append(attr["name"])

# fixed.. convert numpy.float64 values to native Python floats
coef_values = [float(x) for x in lr_model.coefficients]

if len(flat_feature_names) < len(coef_values):
    flat_feature_names += [f"unknown_feat_{i}" for i in range(len(coef_values) - len(flat_feature_names))]

flat_feature_names.append("Intercept")
coef_values.append(float(lr_model.intercept))

# Create results Df
coef_df = spark.createDataFrame(
    zip(flat_feature_names[:len(coef_values)], coef_values),
    ["feature", "coefficient"]
)

coef_df.show(100, truncate=False)

coef_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
    "/home/varshi/hmda/outputs/r1_coefficients"
)

# Model Summary Evaluators
summary = lr_model.summary
print("Accuracy:", summary.accuracy)
print("Area Under ROC:", summary.areaUnderROC)

# Save Predictions Matrix
predictions = transformed_df # Using the already transformed dataframe to optimize performance
predictions.select(
    "approved",
    "probability",
    "prediction"
).write.mode("overwrite").csv(
    "/home/varshi/hmda/outputs/r1_predictions",
    header=True
)

print("DONE")
spark.stop()
