# Large-Scale Analysis of Racial Disparities in Mortgage Lending Using HMDA Data

## Project Overview

This project investigates racial disparities in mortgage lending outcomes using data from the Home Mortgage Disclosure Act (HMDA). Specifically, the analysis focuses on mortgage applications from California and Texas between 2022 and 2025 and examines whether race is associated with differences in mortgage approval rates and loan pricing after controlling for observable financial characteristics.

The project addresses three related research questions:

### R1

Are applicants of different racial groups denied mortgages at different rates after accounting for financial characteristics and geography?

### R2

Which counties exhibit the largest racial disparities in mortgage approval outcomes?

### R3

Among approved borrowers, do racial groups receive different interest rates after controlling for observable financial characteristics?

The project combines social science research questions with scalable computing techniques to analyze over 5.6 million mortgage applications, making it necessary to perform the complete analysis using large scale computing techniques.

---

## Social Science Motivation

Homeownership is one of the primary pathways to wealth accumulation in the United States. Because mortgage lending determines who can access homeownership and under what terms, disparities in mortgage outcomes can contribute to broader patterns of economic inequality and wealth gaps. Despite legal protections against discrimination, prior research has continued to identify racial disparities in mortgage and credit markets. Studies using HMDA data have found that race remains associated with mortgage approval outcomes even after controlling for applicant characteristics (Munnell et al., 1996). More recent research has also documented evidence of racial discrimination in modern lending markets, suggesting that unequal access to credit remains an important policy concern (Bartlett et al., 2022).

Mortgage lending is particularly important to study because it affects both access to credit and the cost of credit. Lower approval rates can restrict opportunities for homeownership, while higher interest rates can increase borrowing costs and reduce long-term wealth accumulation. Historical housing discrimination may also continue to influence present-day economic outcomes. Santucci (2025) argues that racially restrictive housing policies contributed to wealth disparities that continue to affect racial groups today.

By analyzing mortgage applications from California and Texas, I examine whether racial disparities persist in mortgage approvals and interest rates post Covid after controlling for observable financial characteristics. The project also investigates how these disparities vary geographically across counties.

### References

* Bartlett, R., Morse, A., Stanton, R., & Wallace, N. (2022). *Consumer-Lending Discrimination in the FinTech Era*. Journal of Financial Economics, 143(1), 30–56.
* Munnell, A. H., Tootell, G. M. B., Browne, L. E., & McEneaney, J. (1996). *Mortgage Lending in Boston: Interpreting HMDA Data*. American Economic Review, 86(1), 25–53.
* Santucci, L. (2025). *The Racial Wealth Gap and the Legacy of Racially Restrictive Housing Covenants*. In *The Oxford Handbook of the Economics of Housing* (pp. 135–152). Oxford University Press. https://doi.org/10.1093/oso/9780198939030.003.0006

---

## Why Scalable Computing Was Necessary

The HMDA dataset used in this project contained approximately 5.6 million mortgage applications from California and Texas. The size of the dataset created several computational challenges:

* Reading multiple years of mortgage records
* Cleaning millions of observations
* Creating derived variables
* Encoding categorical variables with hundreds of counties
* Running machine learning regression models
* Producing county-level disparity calculations
* Generating large prediction outputs

Traditional desktop analysis tools are not sufficient to analyze a dataset of this size. The machine learning procedures require repeated passes through the data, and county-level fixed effects introduce hundreds of additional predictors.

To address these challenges, the project relied heavily on Apache Spark running on the Midway3 computing cluster.

Most computation was executed through Spark jobs submitted using SBATCH scripts, while initial data cleaning and preprocessing were performed locally before transferring the final dataset to Midway3 for large-scale analysis. Additionally, state map visualizations were produced locally, after the analysis was run on Midway3.

---

## Data Sources

The project uses publicly available data from the Home Mortgage Disclosure Act (HMDA) database.

The dataset contains information on:

* Loan applications
* Approval outcomes
* Loan amounts
* Applicant income
* Applicant race
* Interest rates
* County location
* Loan type
* Application year

Only records from California (CA) and Texas (TX) were retained for analysis, chosen for their high population, large number of counties, and opposing general social/political environments.

Data from four years (2022–2025) were combined into a single dataset. Initial filtering retained only approved and denied applications and removed records with missing key variables. Data cleaning included removing invalid income, loan amount, and interest rate values and constructing a loan-to-income measure. The cleaning and combination procedures were performed locally using Python prior to large-scale Spark analysis.

---

## Computing Infrastructure

The majority of computation was performed on the Midway3 cluster using Apache Spark.

### Submitted Jobs

    | Analysis | Dataset Size | Job ID   |
    | -------- | ------------ | -------- |
    | R1       | 10%          | 50211204 |
    | R1       | 100%         | 50211856 |
    | R2       | 1%           | 50242217 |
    | R2       | 100%         | 50243681 |
    | R3       | 10%          | 50243839 |
    | R3       | 100%         | 50244413 |

### Development Strategy

#### R1 and R3

* Initially tested on a 10% sample of the dataset
* Logic and outputs were verified
* After validation, models were run on the full dataset

#### R2

* Initially tested on a 1% sample because the objective was to validate aggregation logic and disparity calculations
* After verification, the complete dataset was processed

Final production runs used an SBATCH configuration with expanded memory allocation.

---

## Research Question 1 (R1)

### Are Mortgage Approval Outcomes Associated with Race?

R1 estimated a large-scale logistic regression model predicting mortgage approval.

### Dependent Variable

* Mortgage approval (1 = approved, 0 = denied)

### Independent Variables

* Income
* Loan amount
* Loan-to-income ratio
* Race group
* Loan type
* County
* Application year

Categorical variables were converted into machine-readable features using:

* StringIndexer
* OneHotEncoder
* VectorAssembler

before fitting a Spark MLlib logistic regression model.

### Full Dataset Results

The final model used 5,626,294 observations.

**Key model output:**

* Intercept = 0.940
* Sample size = 5.63 million applications

### Race Coefficients
    
    | Race Group      | Coefficient |
    | --------------- | ----------- |
    | White           | +0.174      |
    | Other           | -0.207      |
    | Asian           | +0.181      |
    | Black           | -0.377      |
    | Native American | -0.336      |

### Interpretation

The racial coefficients indicate differences in approval likelihood relative to the omitted reference category while holding income, loan amount, county, loan type, and year constant.

The strongest negative effects appeared for:

* Black applicants (-0.377)
* Native American applicants (-0.336)

These negative coefficients suggest lower odds of approval even after controlling for observable financial characteristics.

Conversely:

* White applicants (+0.174)
* Asian applicants (+0.181)

showed positive coefficients, indicating higher approval odds relative to the baseline category.

These findings suggest that racial disparities in mortgage approval cannot be fully explained by income, loan size, geography, or loan type alone.

---

## Research Question 2 (R2)

### Which Counties Exhibit the Largest Approval Gaps?

R2 was developed directly from the findings of R1.

Because R1 revealed a clear divide between racial groups with positive approval effects (White and Asian borrowers) and groups with negative approval effects (Black and Native American borrowers), R2 used these observed racial dynamics to create two comparison groups.

### Advantaged Group

* White
* Asian

### Disadvantaged Group

* Black
* Native American

County-level approval rates were then calculated for both groups, and an approval disparity measure was constructed:

**Approval Gap = Advantaged Approval Rate − Disadvantaged Approval Rate**

This analysis was performed using Spark aggregations across all Texas and California counties.

### Findings

The largest disparities were concentrated in Texas counties.
    
    | County FIPS | Approval Gap |
    | ----------- | ------------ |
    | 48477       | 0.430        |
    | 48051       | 0.378        |
    | 48149       | 0.371        |
    | 48063       | 0.369        |
    | 48089       | 0.366        |

The largest observed county gap was approximately 43%, indicating that approval outcomes differed substantially across racial groups in certain local lending environments.

These findings suggest that geographic context plays an important role in shaping mortgage outcomes and that disparities are not evenly distributed across space.

---

## Research Question 3 (R3)

### Do Approved Borrowers Receive Different Interest Rates?

R3 focused on borrowers whose mortgages were approved.

A Spark MLlib linear regression model was estimated using:

### Dependent Variable

* Interest rate

### Independent Variables

* Income
* Loan amount
* Loan-to-income ratio
* Race group
* Loan type
* County
* Year

Only approved loans with valid interest rates were retained.

### Full Dataset Results

The final model included approximately 4.08 million approved loans.

**Model Performance**

* R² = 0.393
* RMSE = 1.422
* Intercept = 6.290

### Race Coefficients
    
    | Race Group      | Coefficient |
    | --------------- | ----------- |
    | White           | +0.036      |
    | Other           | +0.056      |
    | Asian           | -0.211      |
    | Black           | +0.132      |
    | Native American | +0.039      |

### Interpretation

The R² value of approximately 0.39 indicates that the model explains a substantial portion of variation in mortgage interest rates.

The race coefficients suggest important differences in loan pricing.

Most notably:

* Black borrowers received interest rates approximately 0.132 percentage points higher than the baseline category after controlling for financial characteristics.
* Asian borrowers received interest rates approximately 0.211 percentage points lower than the baseline category.
* White borrowers experienced a small positive coefficient (+0.036).
* Native American borrowers showed a modest positive coefficient (+0.039).

These results indicate that disparities may extend beyond approval decisions and affect the financial terms under which mortgages are issued.

---

## Scaling the Research

Future avenues for increasing scalability include expanding the analysis beyond California and Texas to incorporate HMDA data from all U.S. states and territories. Such an expansion would substantially increase both the number of observations and the geographic complexity of the dataset.

To support this larger workload, the project could transition from local Spark execution to a fully distributed cluster environment with multiple worker nodes, allowing data and computations to be partitioned across several machines. Additional optimizations could include storing the HMDA data in columnar formats such as Parquet, leveraging Spark partitioning strategies based on state or county identifiers, increasing executor memory and core allocations, and utilizing distributed file systems for more efficient storage and retrieval.
