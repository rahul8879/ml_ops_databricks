# src/data/feature_store_register.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Feature Store Registration
# Gold table ko Feature Store mein register karo
# ═══════════════════════════════════════════════════════

from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureEngineeringClient
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
CATALOG      = "astrazeneca_dev"
GOLD_TABLE   = f"{CATALOG}.gold.drug_features"
FEATURE_TABLE = f"{CATALOG}.ml.drug_efficacy_features"

# ════════════════════════════════════════════════════════
# STEP 1 — Add Primary Key
# Feature Store ko primary key chahiye
# ════════════════════════════════════════════════════════
def add_primary_key():
    print("\n[1/3] Adding primary key to gold table...")

    from pyspark.sql import functions as F

    df = spark.table(GOLD_TABLE)

    # Unique ID add karo
    df = df.withColumn(
        "feature_id",
        F.monotonically_increasing_id()
    )

    # Save back
    df.write \
      .format("delta") \
      .mode("overwrite") \
      .option("overwriteSchema", "true") \
      .saveAsTable(GOLD_TABLE)

    print(f"      ✅ Primary key added: feature_id")
    return df


# ════════════════════════════════════════════════════════
# STEP 2 — Register Feature Table
# ════════════════════════════════════════════════════════
def register_feature_table(df):
    print("\n[2/3] Registering Feature Table...")

    fe = FeatureEngineeringClient()

    # Feature columns — target aur metadata chhod do
    feature_cols = [
        # Molecular
        "molecular_weight", "logP", "hbd_count",
        "hba_count", "rotatable_bonds", "tpsa",
        # Patient
        "patient_age", "patient_gender",
        "ecog_performance_score", "prior_chemotherapy",
        "biomarker_pdl1", "dosage_mg", "treatment_cycles",
        # Encoded
        "cell_line_index", "age_group_index",
        # Derived
        "lipinski_score", "patient_risk_score",
        "dose_normalized_tpsa", "high_pdl1_flag",
        # Normalized
        "molecular_weight_norm", "logP_norm",
        "tpsa_norm", "biomarker_pdl1_norm",
        "dosage_mg_norm", "patient_age_norm",
    ]

    df_features = df.select(["feature_id"] + feature_cols)

    try:
        # Create feature table
        fe.create_table(
            name=FEATURE_TABLE,
            primary_keys=["feature_id"],
            df=df_features,
            description="""
                AstraZeneca Drug Efficacy Features
                ───────────────────────────────────
                Molecular features: MW, logP, HBD, HBA, TPSA
                Patient features: age, ECOG, PDL1, dosage
                Derived: lipinski_score, patient_risk_score
                Normalized: all numerical features 0-1 scaled

                Used for: Drug Efficacy Prediction Model
                Owner: ml-platform@astrazeneca.com
                Version: v1.0
            """
        )
        print(f"      ✅ Feature table created: {FEATURE_TABLE}")

    except Exception as e:
        if "already exists" in str(e).lower():
            # Update existing table
            fe.write_table(
                name=FEATURE_TABLE,
                df=df_features,
                mode="overwrite"
            )
            print(f"      ✅ Feature table updated: {FEATURE_TABLE}")
        else:
            raise e

    return fe


# ════════════════════════════════════════════════════════
# STEP 3 — Verify Registration
# ════════════════════════════════════════════════════════
def verify_registration(fe):
    print("\n[3/3] Verifying Feature Store registration...")

    ft = fe.get_table(name=FEATURE_TABLE)

    print(f"      Name:        {ft.name}")
    print(f"      Primary Key: {ft.primary_keys}")
    print(f"      Description: registered ✅")

    # Count features
    count = spark.table(FEATURE_TABLE).count()
    cols  = len(spark.table(FEATURE_TABLE).columns) - 1

    print(f"      Rows:        {count:,}")
    print(f"      Features:    {cols}")

    print("\n      Visible in:")
    print(f"      Catalog → {FEATURE_TABLE}")
    print(f"      Databricks → Feature Store tab")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*55)
    print(" AstraZeneca — Feature Store Registration")
    print("="*55)

    df  = add_primary_key()
    fe  = register_feature_table(df)
    verify_registration(fe)

    print("\n" + "="*55)
    print(" WEEK 2 — DATA LAYER COMPLETE!")
    print("="*55)
    print("""
  What we built:
  ✅ Bronze  — Raw drug + patient data (12,000 rows)
  ✅ Silver  — Cleaned + validated data
  ✅ Gold    — Feature engineered data (26 features)
  ✅ Feature Store — Registered in Unity Catalog

  Next Week:
  📌 MLflow experiment setup
  📌 XGBoost model training
  📌 Model Registry
  📌 Model deployment
    """)
    print("="*55)