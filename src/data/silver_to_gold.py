# src/data/silver_to_gold.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Silver to Gold Layer
# Feature Engineering — model ready data
# ═══════════════════════════════════════════════════════

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
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
SOURCE  = f"{CATALOG}.silver.drug_clean"
TARGET  = f"{CATALOG}.gold.drug_features"
REF     = f"{CATALOG}.gold.reference_data"

# ════════════════════════════════════════════════════════
# STEP 1 — Load Silver
# ════════════════════════════════════════════════════════
def load_silver():
    print(f"\n[1/6] Loading silver data...")
    df = spark.table(SOURCE)
    print(f"      Rows: {df.count()}")
    return df


# ════════════════════════════════════════════════════════
# STEP 2 — Derived Features
# New features banao existing se
# ════════════════════════════════════════════════════════
def add_derived_features(df):
    print("\n[2/6] Adding derived features...")

    df = df.withColumn(
        # Lipinski Rule of 5 — drug-likeness score
        # MW<500, logP<5, HBD<5, HBA<10
        "lipinski_score",
        (
            (F.col("molecular_weight") < 500).cast("int") +
            (F.col("logP") < 5).cast("int") +
            (F.col("hbd_count") < 5).cast("int") +
            (F.col("hba_count") < 10).cast("int")
        ).cast(DoubleType())
    ).withColumn(
        # Patient risk score
        # Higher = more at risk (older + worse ecog + prior chemo)
        "patient_risk_score",
        F.round(
            (F.col("patient_age") / 80.0) * 0.4 +
            (F.col("ecog_performance_score") / 4.0) * 0.4 +
            F.col("prior_chemotherapy") * 0.2,
            4
        )
    ).withColumn(
        # Drug efficiency ratio — efficacy per unit dose
        "dose_normalized_tpsa",
        F.round(F.col("tpsa") / F.col("dosage_mg"), 4)
    ).withColumn(
        # High PDL1 flag — important for immunotherapy
        "high_pdl1_flag",
        (F.col("biomarker_pdl1") > 50).cast("int")
    ).withColumn(
        # Age group bucketing
        "age_group",
        F.when(F.col("patient_age") < 40, "young")
         .when(F.col("patient_age") < 60, "middle")
         .otherwise("senior")
    )

    print("      ✅ Added: lipinski_score")
    print("      ✅ Added: patient_risk_score")
    print("      ✅ Added: dose_normalized_tpsa")
    print("      ✅ Added: high_pdl1_flag")
    print("      ✅ Added: age_group")

    return df


# ════════════════════════════════════════════════════════
# STEP 3 — Normalize Numerical Features
# ════════════════════════════════════════════════════════
def normalize_features(df):
    print("\n[3/6] Normalizing numerical features...")

    # Min-max normalization
    # (value - min) / (max - min)
    num_features = {
        "molecular_weight": (100.0, 800.0),
        "logP":             (-5.0,  10.0),
        "tpsa":             (0.0,   200.0),
        "biomarker_pdl1":   (0.0,   100.0),
        "dosage_mg":        (10.0,  400.0),
        "patient_age":      (18.0,  90.0),
    }

    for feat, (min_val, max_val) in num_features.items():
        norm_col = f"{feat}_norm"
        df = df.withColumn(
            norm_col,
            F.round(
                (F.col(feat) - min_val) / (max_val - min_val),
                4
            ).cast(DoubleType())
        )
        print(f"      ✅ Normalized: {feat} → {norm_col}")

    return df


# ════════════════════════════════════════════════════════
# STEP 4 — Encode Categorical Features
# ════════════════════════════════════════════════════════
# encode_categoricals function replace karo — line ~120
def encode_categoricals(df):
    print("\n[4/6] Encoding categorical features...")

    # Pandas mein convert karo — local encoding
    import pandas as pd
    pdf = df.toPandas()

    # cell_line_type → integer
    cell_line_map = {
        "NSCLC": 0, "BREAST": 1, "CRC": 2,
        "LEUKEMIA": 3, "LYMPHOMA": 4
    }
    pdf["cell_line_index"] = pdf["cell_line_type"] \
        .str.upper().map(cell_line_map).fillna(0).astype(int)

    # age_group → integer
    age_group_map = {"young": 0, "middle": 1, "senior": 2}
    pdf["age_group_index"] = pdf["age_group"] \
        .map(age_group_map).fillna(1).astype(int)

    print("      ✅ Encoded: cell_line_type → cell_line_index")
    print("      ✅ Encoded: age_group → age_group_index")

    # Wapas Spark DataFrame banao
    df = spark.createDataFrame(pdf)

    return df


# ════════════════════════════════════════════════════════
# STEP 5 — Select Final Feature Set
# ════════════════════════════════════════════════════════
def select_final_features(df):
    print("\n[5/6] Selecting final feature set...")

    final_cols = [
        # Raw molecular features
        "molecular_weight", "logP", "hbd_count",
        "hba_count", "rotatable_bonds", "tpsa",

        # Raw patient features
        "patient_age", "patient_gender",
        "ecog_performance_score", "prior_chemotherapy",
        "biomarker_pdl1", "dosage_mg", "treatment_cycles",

        # Encoded categoricals
        "cell_line_index", "age_group_index",

        # Derived features
        "lipinski_score", "patient_risk_score",
        "dose_normalized_tpsa", "high_pdl1_flag",

        # Normalized features
        "molecular_weight_norm", "logP_norm",
        "tpsa_norm", "biomarker_pdl1_norm",
        "dosage_mg_norm", "patient_age_norm",

        # Metadata
        "data_split", "data_version",
        "is_drift", "efficacy_label"
    ]

    df = df.select(final_cols)

    print(f"      ✅ Final features: {len(final_cols) - 4} features")
    print(f"         + 4 metadata columns")

    return df


# ════════════════════════════════════════════════════════
# STEP 6 — Save Gold + Reference Data
# ════════════════════════════════════════════════════════
def save_to_gold(df):
    print(f"\n[6/6] Saving to Gold layer...")

    # Save full gold table
    df.write \
      .format("delta") \
      .mode("overwrite") \
      .option("overwriteSchema", "true") \
      .partitionBy("data_split") \
      .saveAsTable(TARGET)

    count = spark.sql(
        f"SELECT COUNT(*) as cnt FROM {TARGET}"
    ).collect()[0]["cnt"]
    print(f"      ✅ Gold table saved: {count} rows → {TARGET}")

    # Save reference data separately
    # Reference = training data only
    # Evidently AI drift detection ke liye use hoga
    df_ref = df.filter(F.col("data_split") == "train")

    df_ref.write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .saveAsTable(REF)

    ref_count = spark.sql(
        f"SELECT COUNT(*) as cnt FROM {REF}"
    ).collect()[0]["cnt"]
    print(f"      ✅ Reference data saved: {ref_count} rows → {REF}")
    print(f"         (Used for drift detection baseline)")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*55)
    print(" AstraZeneca — Silver to Gold")
    print("="*55)

    df = load_silver()
    df = add_derived_features(df)
    df = normalize_features(df)
    df = encode_categoricals(df)
    df = select_final_features(df)
    save_to_gold(df)

    # Final summary
    print("\n" + "="*55)
    print(" MEDALLION ARCHITECTURE STATUS")
    print("="*55)
    for table in [
        "astrazeneca_dev.bronze.drug_raw",
        "astrazeneca_dev.silver.drug_clean",
        "astrazeneca_dev.gold.drug_features",
        "astrazeneca_dev.gold.reference_data",
    ]:
        cnt = spark.sql(
            f"SELECT COUNT(*) as c FROM {table}"
        ).collect()[0]["c"]
        layer = table.split(".")[1].upper()
        print(f"  {layer:10} {table:45} {cnt:,} rows")

    print("\n ✅ Gold Layer Ready!")
    print(" Next: Drift data simulation")
    print("="*55)