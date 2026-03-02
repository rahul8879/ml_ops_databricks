# src/training/notify.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Training Pipeline Notification
# Task 3: Registration ke baad chalta hai
# ═══════════════════════════════════════════════════════

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="astrazeneca_dev")
parser.add_argument("--env",     default="dev")
args = parser.parse_args()

CATALOG = args.catalog
ENV     = args.env

print("\n" + "="*55)
print(" AstraZeneca — Pipeline Notification")
print("="*55)

# Latest registered run lo
df = spark.table(f"{CATALOG}.monitoring.training_runs") \
          .filter(F.col("environment") == ENV) \
          .filter(F.col("status") == "registered") \
          .orderBy(F.col("created_at").desc()) \
          .limit(1)

if df.count() == 0:
    print("⚠️  No registered runs found")
else:
    row = df.collect()[0]

    print(f"""
    ✅ TRAINING PIPELINE COMPLETE

    Environment:   {ENV}
    Catalog:       {CATALOG}
    Run ID:        {row['run_id']}
    Model Version: {row['model_version']}

    Metrics:
    ├── Precision:  {row['precision']:.4f}
    ├── AUC:        {row['roc_auc']:.4f}
    ├── F1:         {row['f1_score']:.4f}
    └── Accuracy:   {row['accuracy']:.4f}

    Action Required:
    → Review model in MLflow UI
    → Approve for Production?
    → Catalog → Models → {CATALOG}.ml.drug_efficacy_model
    """)

    # Audit trail mein log karo
    spark.sql(f"""
        UPDATE {CATALOG}.monitoring.training_runs
        SET status = 'notified'
        WHERE run_id = '{row['run_id']}'
    """)

    print("✅ Audit trail updated")
    print("✅ Pipeline complete!")

print("="*55)