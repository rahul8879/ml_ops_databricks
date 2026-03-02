# src/data/bronze_to_silver.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Bronze to Silver Layer
# Cleans raw drug data + data quality checks
# ═══════════════════════════════════════════════════════

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from dotenv import load_dotenv
import os

load_dotenv(".env")

# ── Spark Session ────────────────────────────────────────
spark = DatabricksSession.builder \
    .host(os.getenv("DATABRICKS_HOST")) \
    .token(os.getenv("DATABRICKS_TOKEN")) \
    .clusterId(os.getenv("DATABRICKS_CLUSTER_ID")) \
    .getOrCreate()

print("✅ Spark connected!")

# ── Config ───────────────────────────────────────────────
CATALOG = "astrazeneca_dev"
SOURCE  = f"{CATALOG}.bronze.drug_raw"
TARGET  = f"{CATALOG}.silver.drug_clean"

# ════════════════════════════════════════════════════════
# STEP 1 — Load Bronze Data
# ════════════════════════════════════════════════════════
def load_bronze():
    print(f"\n[1/5] Loading bronze data from {SOURCE}...")
    df = spark.table(SOURCE)
    print(f"      Rows loaded: {df.count()}")
    print(f"      Columns:     {len(df.columns)}")
    return df


# ════════════════════════════════════════════════════════
# STEP 2 — Data Quality Checks
# ════════════════════════════════════════════════════════
def data_quality_checks(df):
    print("\n[2/5] Running data quality checks...")

    total = df.count()
    issues = []

    # Check nulls
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    for col, cnt in null_counts.items():
        if cnt > 0:
            issues.append(f"  ⚠️  Nulls in {col}: {cnt}")

    # Check duplicates
    dupes = total - df.dropDuplicates().count()
    if dupes > 0:
        issues.append(f"  ⚠️  Duplicates found: {dupes}")

    # Check value ranges
    stats = df.select(
        F.min("molecular_weight").alias("mw_min"),
        F.max("molecular_weight").alias("mw_max"),
        F.min("patient_age").alias("age_min"),
        F.max("patient_age").alias("age_max"),
        F.min("dosage_mg").alias("dose_min"),
        F.max("dosage_mg").alias("dose_max"),
        F.min("biomarker_pdl1").alias("pdl1_min"),
        F.max("biomarker_pdl1").alias("pdl1_max"),
    ).collect()[0]

    print(f"      Molecular weight range: {stats.mw_min:.1f} - {stats.mw_max:.1f} Da")
    print(f"      Patient age range:      {stats.age_min} - {stats.age_max} yrs")
    print(f"      Dosage range:           {stats.dose_min:.1f} - {stats.dose_max:.1f} mg")
    print(f"      PDL1 range:             {stats.pdl1_min:.1f} - {stats.pdl1_max:.1f}%")

    if issues:
        print("\n      Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("      ✅ No data quality issues found!")

    return df


# ════════════════════════════════════════════════════════
# STEP 3 — Clean Data
# ════════════════════════════════════════════════════════
def clean_data(df):
    print("\n[3/5] Cleaning data...")

    original_count = df.count()

    df = df \
        .dropDuplicates() \
        .dropna(subset=["efficacy_label", "molecular_weight",
                        "patient_age", "dosage_mg"]) \
        .filter(
            (F.col("molecular_weight").between(100, 800)) &
            (F.col("patient_age").between(18, 90)) &
            (F.col("dosage_mg").between(10, 400)) &
            (F.col("biomarker_pdl1").between(0, 100)) &
            (F.col("logP").between(-5, 10)) &
            (F.col("ecog_performance_score").between(0, 4))
        ) \
        .withColumn("patient_gender",
            F.col("patient_gender").cast(IntegerType())) \
        .withColumn("prior_chemotherapy",
            F.col("prior_chemotherapy").cast(IntegerType())) \
        .withColumn("efficacy_label",
            F.col("efficacy_label").cast(IntegerType())) \
        .withColumn("cell_line_type",
            F.upper(F.trim(F.col("cell_line_type")))) \
        .withColumn("silver_processed", F.lit(True)) \
        .withColumn("silver_version", F.lit("v1.0"))

    final_count = df.count()
    removed     = original_count - final_count

    print(f"      Original rows: {original_count}")
    print(f"      Removed rows:  {removed}")
    print(f"      Clean rows:    {final_count}")

    return df


# ════════════════════════════════════════════════════════
# STEP 4 — Class Imbalance Report
# ════════════════════════════════════════════════════════
def imbalance_report(df):
    print("\n[4/5] Class balance report...")

    df.groupBy("efficacy_label") \
      .agg(
          F.count("*").alias("count"),
          F.round(
              F.count("*") * 100.0 / df.count(), 2
          ).alias("percentage")
      ) \
      .orderBy("efficacy_label") \
      .show()

    print("      ℹ️  Imbalance noted — will handle with")
    print("         class_weight in XGBoost training")

    return df


# ════════════════════════════════════════════════════════
# STEP 5 — Save to Silver Layer
# ════════════════════════════════════════════════════════
def save_to_silver(df):
    print(f"\n[5/5] Saving to {TARGET}...")

    df.write \
      .format("delta") \
      .mode("overwrite") \
      .option("overwriteSchema", "true") \
      .partitionBy("data_split") \
      .saveAsTable(TARGET)

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TARGET}") \
                 .collect()[0]["cnt"]

    print(f"      ✅ Saved {count} rows to {TARGET}")
    print(f"      ✅ Partitioned by: data_split")

    # Show sample
    print("\n      Sample rows:")
    spark.table(TARGET).limit(3).show(truncate=False)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*55)
    print(" AstraZeneca — Bronze to Silver")
    print("="*55)

    df = load_bronze()
    df = data_quality_checks(df)
    df = clean_data(df)
    df = imbalance_report(df)
    save_to_silver(df)

    print("\n" + "="*55)
    print(" ✅ Silver Layer Ready!")
    print(" Next: Gold layer — feature engineering")
    print("="*55)