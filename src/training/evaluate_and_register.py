# src/training/evaluate_and_register.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Model Evaluation + Registration
# Task 2: Train ke baad automatically chalta hai
# ═══════════════════════════════════════════════════════

import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

# ── Spark ────────────────────────────────────────────────
spark = SparkSession.builder.getOrCreate()

# ── Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--catalog",
                    default="astrazeneca_dev")
parser.add_argument("--experiment_name",
                    default="/astrazeneca/dev/drug_efficacy")
parser.add_argument("--env",
                    default="dev")
args   = parser.parse_args()

CATALOG         = args.catalog
ENV             = args.env
EXPERIMENT_NAME = f"/Users/rtiwarirahul123@gmail.com/astrazeneca_{ENV}_drug_efficacy"
MODEL_NAME      = f"{CATALOG}.ml.drug_efficacy_model"

# ── Thresholds ───────────────────────────────────────────
# Yeh pass hone chahiye registration ke liye
THRESHOLDS = {
    "precision": 0.85,    # Min 85% precision
    "roc_auc":   0.90,    # Min 90% AUC
    "f1_score":  0.85,    # Min 85% F1
}

print("\n" + "="*55)
print(" AstraZeneca — Evaluate + Register")
print("="*55)
print(f"  Catalog:    {CATALOG}")
print(f"  Experiment: {EXPERIMENT_NAME}")
print(f"  Model:      {MODEL_NAME}")
print("="*55)


# ════════════════════════════════════════════════════════
# STEP 1 — Latest Run ID load karo
# Train task ne Delta table mein save kiya tha
# ════════════════════════════════════════════════════════
def get_latest_run():
    print("\n[1/4] Loading latest run metadata...")

    df = spark.table(f"{CATALOG}.monitoring.training_runs") \
              .filter(F.col("environment") == ENV) \
              .filter(F.col("status") == "trained") \
              .orderBy(F.col("created_at").desc()) \
              .limit(1)

    if df.count() == 0:
        raise ValueError("❌ No trained runs found!")

    row = df.collect()[0]

    print(f"      Run ID:    {row['run_id']}")
    print(f"      Precision: {row['precision']:.4f}")
    print(f"      AUC:       {row['roc_auc']:.4f}")
    print(f"      F1:        {row['f1_score']:.4f}")

    return row


# ════════════════════════════════════════════════════════
# STEP 2 — Threshold Check
# Pass? → Register
# Fail? → Job fail karo
# ════════════════════════════════════════════════════════
def check_thresholds(row):
    print("\n[2/4] Checking quality thresholds...")

    passed  = True
    results = []

    for metric, threshold in THRESHOLDS.items():
        value  = row[metric]
        status = "✅ PASS" if value >= threshold else "❌ FAIL"
        passed = passed and (value >= threshold)
        results.append(
            f"      {metric:<12} {value:.4f} >= {threshold} {status}"
        )

    print(f"      {'Metric':<12} {'Value':>8}   {'Min':>6}  Status")
    print(f"      {'─'*45}")
    for r in results:
        print(r)

    if not passed:
        print("\n      ❌ Thresholds NOT met!")
        print("         Model will NOT be registered")
        print("         Check MLflow for details")

        # Status update karo
        spark.sql(f"""
            UPDATE {CATALOG}.monitoring.training_runs
            SET status = 'rejected'
            WHERE run_id = '{row['run_id']}'
        """)

        sys.exit(1)   # Job fail karo — notify task ko pata chalega

    print("\n      ✅ All thresholds passed!")
    return True


# ════════════════════════════════════════════════════════
# STEP 3 — Model Register karo
# Unity Catalog Registry mein
# ════════════════════════════════════════════════════════
def register_model(row):
    print(f"\n[3/4] Registering model in Unity Catalog...")

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    model_uri = f"runs:/{row['run_id']}/model"

    # Register karo
    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )
    version = result.version
    print(f"      ✅ Registered: {MODEL_NAME} v{version}")

    # ── Databricks Recommended Aliases ───────────────────
    # First model hai — champion + baseline dono set karo
    # Aage naye models sirf "challenger" se aayenge

    existing_aliases = [
        v.aliases for v in
        client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.version == version
    ]

    # Champion alias — kya pehle se koi champion hai?
    try:
        current_champion = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion"
        )
        # Champion already exists → naya model challenger hai
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="challenger",
            version=version
        )
        print(f"      ✅ Alias: challenger → v{version}")
        print(f"      ℹ️  Champion already exists: v{current_champion.version}")
        print(f"      ℹ️  Human review required to promote challenger → champion")

    except Exception:
        # Pehla model hai — seedha champion set karo
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=version
        )
        print(f"      ✅ Alias: champion → v{version}")

    # Baseline hamesha set karo — drift detection ke liye
    try:
        client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="baseline"
        )
        print(f"      ℹ️  Baseline already exists — not overwriting")
    except Exception:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="baseline",
            version=version
        )
        print(f"      ✅ Alias: baseline → v{version}")

    return version


# ════════════════════════════════════════════════════════
# STEP 4 — Status Update karo
# ════════════════════════════════════════════════════════
def update_status(row, version):
    print(f"\n[4/4] Updating run status...")

    spark.sql(f"""
        UPDATE {CATALOG}.monitoring.training_runs
        SET status = 'registered',
            model_version = '{version}'
        WHERE run_id = '{row['run_id']}'
    """)

    print(f"      ✅ Status: trained → registered")
    print(f"      ✅ Version: {version}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
row     = get_latest_run()
check_thresholds(row)
version = register_model(row)
update_status(row, version)

print("\n" + "="*55)
print(" ✅ Model Registered!")
print("="*55)
print(f"""
  Model:   {MODEL_NAME}
  Version: {version}
  Status:  Staging

  Next:
  → notify.py will send team alert
  → Manual review in MLflow UI
  → Production promote: Week 3 Day 4
""")
print("="*55)