# src/training/log_lineage.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Model Lineage Tracker
# Data → Features → Model → Deployment
# Complete audit trail
# ═══════════════════════════════════════════════════════

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="astrazeneca_dev")
parser.add_argument("--env",     default="dev")
args = parser.parse_args()

CATALOG = args.catalog
ENV     = args.env

LINEAGE_TABLE  = f"{CATALOG}.monitoring.model_lineage"
TRAINING_TABLE = f"{CATALOG}.monitoring.training_runs"

print("\n" + "="*55)
print(" AstraZeneca — Model Lineage Logger")
print("="*55)


def log_lineage():
    print("\n[1/2] Loading latest training run...")

    # Latest registered run lo
    df = spark.table(TRAINING_TABLE) \
          .filter(F.col("environment") == ENV) \
          .filter(F.col("status") == "trained") \
          .orderBy(F.col("created_at").desc()) \
          .limit(1)

    if df.count() == 0:
        print("⚠️  No registered runs found")
        return

    row = df.collect()[0]

    print("\n[2/2] Saving lineage record...")

    lineage = spark.createDataFrame([{
    "lineage_id":       f"{row['run_id'][:8]}_{ENV}",
    "environment":      ENV,
    "created_at":       pd.Timestamp.now().isoformat(),

    # Data Layer
    "gold_table":       str(row["gold_table"]),
    "delta_version":    str(row["delta_version"]),
    "delta_timestamp":  str(row["delta_timestamp"]),
    "train_rows":       "N/A",                          # ← hardcode for now

    # Model Layer
    "run_id":           str(row["run_id"]),
    "model_version":    str(row["model_version"]),
    "model_name":       f"{CATALOG}.ml.drug_efficacy_model",
    "model_alias":      "champion",

    # Performance
    "precision":        float(row["precision"]),
    "roc_auc":          float(row["roc_auc"]),
    "f1_score":         float(row["f1_score"]),
    "accuracy":         float(row["accuracy"]),

    # Environment
    "xgboost_version":  str(row["xgboost_version"]),
    "catalog":          CATALOG,
    "status":           "active",
   }])

    lineage.write \
           .format("delta") \
           .mode("append") \
           .option("mergeSchema", "true") \
           .saveAsTable(LINEAGE_TABLE)

    print(f"\n✅ Lineage saved: {LINEAGE_TABLE}")

    # Display
    spark.sql(f"""
        SELECT
            lineage_id,
            delta_version,
            model_version,
            model_alias,
            ROUND(precision, 4)  as precision,
            ROUND(roc_auc, 4)    as auc,
            status,
            created_at
        FROM {LINEAGE_TABLE}
        ORDER BY created_at DESC
    """).show(truncate=False)


log_lineage()

print("\n" + "="*55)
print(" ✅ Lineage Complete!")
print("="*55)
print(f"""
  Lineage Table: {LINEAGE_TABLE}

  Reproduce karna ho to:
  1. delta_version dekho
  2. RESTORE TABLE {CATALOG}.gold.drug_features
     TO VERSION AS OF <delta_version>
  3. Training job chalo
  4. Exact same model milega
""")
print("="*55)