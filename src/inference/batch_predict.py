# src/inference/batch_predict.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Batch Inference Pipeline
# Champion model se predictions karo
# Results Delta Lake mein save karo
# ═══════════════════════════════════════════════════════

import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
    IntegerType, BooleanType, TimestampType
)
from datetime import datetime

# ── Spark ────────────────────────────────────────────────
spark = SparkSession.builder.getOrCreate()

# ── Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="astrazeneca_dev")
parser.add_argument("--env",     default="dev")
args = parser.parse_args()

CATALOG = args.catalog
ENV     = args.env

# ── Config ───────────────────────────────────────────────
GOLD_TABLE        = f"{CATALOG}.gold.drug_features"
PREDICTION_TABLE  = f"{CATALOG}.monitoring.prediction_logs"
MODEL_NAME        = f"{CATALOG}.ml.drug_efficacy_model"
MODEL_ALIAS       = "champion"

# ── Feature Columns ──────────────────────────────────────
FEATURE_COLS = [
    "molecular_weight", "logP", "hbd_count",
    "hba_count", "rotatable_bonds", "tpsa",
    "patient_age", "patient_gender",
    "ecog_performance_score", "prior_chemotherapy",
    "biomarker_pdl1", "dosage_mg", "treatment_cycles",
    "cell_line_index", "age_group_index",
    "lipinski_score", "patient_risk_score",
    "dose_normalized_tpsa", "high_pdl1_flag",
    "molecular_weight_norm", "logP_norm",
    "tpsa_norm", "biomarker_pdl1_norm",
    "dosage_mg_norm", "patient_age_norm",
]

print("\n" + "="*55)
print(" AstraZeneca — Batch Inference Pipeline")
print("="*55)
print(f"  Environment: {ENV}")
print(f"  Catalog:     {CATALOG}")
print(f"  Model:       {MODEL_NAME}@{MODEL_ALIAS}")
print("="*55)


# ════════════════════════════════════════════════════════
# STEP 1 — Champion Model Load karo
# ════════════════════════════════════════════════════════
def load_champion_model():
    print("\n[1/4] Loading champion model from Registry...")

    mlflow.set_registry_uri("databricks-uc")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model     = mlflow.xgboost.load_model(model_uri)

    print(f"      ✅ Model loaded: {model_uri}")
    return model


# ════════════════════════════════════════════════════════
# STEP 2 — Data Load karo
# Validation split — "unseen" data jaisa
# ════════════════════════════════════════════════════════
def load_inference_data():
    print("\n[2/4] Loading inference data...")

    df_spark = spark.table(GOLD_TABLE)

    # Validation data use karo — production data jaisa
    df = df_spark.filter(
        F.col("data_split") == "validation"
    ).toPandas()

    print(f"      Rows loaded: {len(df):,}")
    print(f"      Features:    {len(FEATURE_COLS)}")

    return df


# ════════════════════════════════════════════════════════
# STEP 3 — Predictions karo
# ════════════════════════════════════════════════════════
def run_predictions(model, df):
    print("\n[3/4] Running predictions...")

    X = df[FEATURE_COLS]

    # Predict
    predictions      = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]

    # Results DataFrame
    df_results = pd.DataFrame({
        # Predictions
        "prediction":          predictions.astype(int),
        "prediction_prob":     prediction_probs.round(4),
        "prediction_label":    ["Effective" if p == 1
                                else "Not Effective"
                                for p in predictions],

        # Confidence
        "confidence":          np.where(
                                   prediction_probs >= 0.5,
                                   prediction_probs,
                                   1 - prediction_probs
                               ).round(4),

        # Key features — drift detection ke liye
        "molecular_weight":    df["molecular_weight"].values,
        "patient_age":         df["patient_age"].values,
        "dosage_mg":           df["dosage_mg"].values,
        "ecog_performance_score": df["ecog_performance_score"].values,
        "biomarker_pdl1":      df["biomarker_pdl1"].values,
        "logP":                df["logP"].values,

        # Actuals — evaluation ke liye
        "actual_label":        df["efficacy_label"].values.astype(int),

        # Metadata
        "model_name":          MODEL_NAME,
        "model_alias":         MODEL_ALIAS,
        "environment":         ENV,
        "batch_id":            datetime.now().strftime("%Y%m%d_%H%M%S"),
        "predicted_at":        datetime.now().isoformat(),
        "data_version":        df["data_version"].values,
        "is_correct":          (predictions ==
                                df["efficacy_label"].values).astype(bool),
    })

    # Quick stats
    total     = len(df_results)
    effective = df_results["prediction"].sum()
    correct   = df_results["is_correct"].sum()
    accuracy  = correct / total

    print(f"      Total predictions:  {total:,}")
    print(f"      Effective (1):      {effective:,} ({effective/total:.1%})")
    print(f"      Not Effective (0):  {total-effective:,}")
    print(f"      Correct:            {correct:,} ({accuracy:.2%})")
    print(f"      Avg confidence:     {df_results['confidence'].mean():.4f}")

    return df_results


# ════════════════════════════════════════════════════════
# STEP 4 — Save to Delta Lake
# ════════════════════════════════════════════════════════
def save_predictions(df_results):
    print(f"\n[4/4] Saving predictions to Delta Lake...")

    spark_df = spark.createDataFrame(df_results)

    # Append mode — har batch ka record rakho
    spark_df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(PREDICTION_TABLE)

    count = spark.sql(
        f"SELECT COUNT(*) as cnt FROM {PREDICTION_TABLE}"
    ).collect()[0]["cnt"]

    print(f"      ✅ Saved to: {PREDICTION_TABLE}")
    print(f"      ✅ Total prediction logs: {count:,}")

    # Sample dekho
    print("\n      Sample predictions:")
    spark.table(PREDICTION_TABLE) \
         .select(
             "prediction",
             "prediction_prob",
             "prediction_label",
             "molecular_weight",
             "patient_age",
             "is_correct"
         ) \
         .limit(5) \
         .show(truncate=False)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
model     = load_champion_model()
df        = load_inference_data()
df_results = run_predictions(model, df)
save_predictions(df_results)

print("\n" + "="*55)
print(" ✅ Batch Inference Complete!")
print("="*55)
print(f"""
  Predictions saved to:
  {PREDICTION_TABLE}

  Next Steps:
  → Week 4 Day 2: Drift data inject karo
  → Week 4 Day 3: Evidently AI drift detection
  → Week 4 Day 4: Agent trigger setup
""")
print("="*55)