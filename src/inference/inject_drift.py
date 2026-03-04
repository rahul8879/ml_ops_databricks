# src/inference/inject_drift.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Drift Data Injection
# Production mein naya drifted data aa raha hai
# Champion model isko predict karega
# Evidently AI baad mein drift detect karega
# ═══════════════════════════════════════════════════════

import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

# ── Spark ────────────────────────────────────────────────
spark = SparkSession.builder.getOrCreate()

# ── Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="astrazeneca_dev")
parser.add_argument("--env",     default="dev")
parser.add_argument("--n_rows",  default=2000, type=int)
args = parser.parse_args()

CATALOG  = args.catalog
ENV      = args.env
N_ROWS   = args.n_rows
SEED     = 99

# ── Config ───────────────────────────────────────────────
PREDICTION_TABLE = f"{CATALOG}.monitoring.prediction_logs"
MODEL_NAME       = f"{CATALOG}.ml.drug_efficacy_model"
MODEL_ALIAS      = "champion"

CELL_LINES = ["NSCLC", "Breast", "CRC", "Leukemia", "Lymphoma"]

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
print(" AstraZeneca — Drift Data Injection")
print("="*55)
print(f"  Catalog:  {CATALOG}")
print(f"  Rows:     {N_ROWS}")
print(f"  Shifts:   MW↑, Age↑, Dosage↑")
print("="*55)


# ════════════════════════════════════════════════════════
# STEP 1 — Drifted Data Generate karo
# 3 real distribution shifts
# ════════════════════════════════════════════════════════
def generate_drift_data(n=N_ROWS):
    print(f"\n[1/4] Generating {n} drifted records...")

    np.random.seed(SEED)

    # ── SHIFT 1: Molecular Weight ────────────────────────
    # Baseline: 340 Da → Drift: 460 Da (BiTE antibodies)
    molecular_weight = np.random.normal(460, 80, n).clip(200, 700)

    # ── SHIFT 2: Patient Age ─────────────────────────────
    # Baseline: 54 yrs → Drift: 61 yrs
    patient_age  = np.random.normal(61, 11, n).clip(30, 85).astype(int)
    ecog_score   = np.random.choice(
        [0, 1, 2, 3], n,
        p=[0.20, 0.35, 0.30, 0.15]  # Worse than baseline
    )

    # ── SHIFT 3: Dosage ──────────────────────────────────
    # Baseline: 125 mg → Drift: 210 mg
    dosage_mg    = np.random.normal(210, 60, n).clip(75, 350)

    # ── Other features — same as baseline ───────────────
    logP             = np.random.normal(2.5, 1.2, n).clip(-3, 7)
    hbd_count        = np.random.randint(0, 6, n)
    hba_count        = np.random.randint(0, 8, n)
    rotatable_bonds  = np.random.randint(0, 10, n)
    tpsa             = np.random.normal(80, 20, n).clip(20, 160)
    patient_gender   = np.random.randint(0, 2, n)
    prior_chemo      = np.random.randint(0, 2, n)
    biomarker_pdl1   = np.random.normal(45, 20, n).clip(0, 100)
    treatment_cycles = np.random.randint(1, 9, n)
    cell_line_type   = np.random.choice(CELL_LINES, n)

    # ── Derived features — same logic ───────────────────
    cell_line_map = {
        "NSCLC": 0, "BREAST": 1, "CRC": 2,
        "LEUKEMIA": 3, "LYMPHOMA": 4
    }
    cell_line_index = np.array([
        cell_line_map.get(c.upper(), 0)
        for c in cell_line_type
    ])

    age_group       = np.where(
        patient_age < 40, "young",
        np.where(patient_age < 60, "middle", "senior")
    )
    age_group_map   = {"young": 0, "middle": 1, "senior": 2}
    age_group_index = np.array([
        age_group_map.get(a, 1) for a in age_group
    ])

    lipinski_score = (
        (molecular_weight < 500).astype(int) +
        (logP < 5).astype(int) +
        (hbd_count < 5).astype(int) +
        (hba_count < 10).astype(int)
    ).astype(float)

    patient_risk_score = (
        (patient_age / 80.0) * 0.4 +
        (ecog_score / 4.0) * 0.4 +
        prior_chemo * 0.2
    ).round(4)

    dose_normalized_tpsa = (tpsa / dosage_mg).round(4)
    high_pdl1_flag       = (biomarker_pdl1 > 50).astype(int)

    # ── Normalized features ──────────────────────────────
    molecular_weight_norm = ((molecular_weight - 100) / 700).round(4)
    logP_norm             = ((logP - (-5)) / 15).round(4)
    tpsa_norm             = (tpsa / 200).round(4)
    biomarker_pdl1_norm   = (biomarker_pdl1 / 100).round(4)
    dosage_mg_norm        = ((dosage_mg - 10) / 390).round(4)
    patient_age_norm      = ((patient_age - 18) / 72).round(4)

    # ── Actual labels ────────────────────────────────────
    mw_penalty     = np.where(molecular_weight > 500, 0.15, 0.0)
    efficacy_score = (
        0.25 * (1 - (molecular_weight - 150) / 350) +
        0.20 * (logP.clip(0, 5) / 5) +
        0.20 * (1 - ecog_score / 3) +
        0.15 * (biomarker_pdl1 / 100) +
        0.10 * (1 - (patient_age - 25) / 55) +
        0.10 * np.random.normal(0, 0.1, n) -
        mw_penalty
    ).clip(0, 1)
    actual_label = (efficacy_score >= 0.5).astype(int)

    df = pd.DataFrame({
        col: globals().get(col, np.zeros(n))
        for col in FEATURE_COLS
    })
    df["efficacy_label"] = actual_label
    df["data_version"]   = "drift_v1"

    print(f"      ✅ Generated {len(df):,} drifted records")
    print(f"      Avg MW:    {molecular_weight.mean():.1f} Da  (baseline: ~340)")
    print(f"      Avg Age:   {patient_age.mean():.1f} yrs (baseline: ~54)")
    print(f"      Avg Dose:  {dosage_mg.mean():.1f} mg  (baseline: ~125)")

    return df


# ════════════════════════════════════════════════════════
# STEP 2 — Champion Model Load
# ════════════════════════════════════════════════════════
def load_champion_model():
    print(f"\n[2/4] Loading champion model...")

    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model     = mlflow.xgboost.load_model(model_uri)

    print(f"      ✅ Loaded: {model_uri}")
    return model


# ════════════════════════════════════════════════════════
# STEP 3 — Predict on Drifted Data
# Model has never seen this distribution!
# ════════════════════════════════════════════════════════
def predict_on_drift(model, df):
    print(f"\n[3/4] Running predictions on drifted data...")

    X                = df[FEATURE_COLS]
    predictions      = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]

    df_results = pd.DataFrame({
    "prediction":             predictions.astype(int),
    "prediction_prob":        prediction_probs.round(4),
    "prediction_label":       ["Effective" if p == 1
                               else "Not Effective"
                               for p in predictions],
    "confidence":             np.where(
                                  prediction_probs >= 0.5,
                                  prediction_probs,
                                  1 - prediction_probs
                              ).round(4),

    # ── Sab float karo — schema match karne ke liye ──
    "molecular_weight":       df["molecular_weight"].values.astype(float),
    "patient_age":            df["patient_age"].values.astype(float),  # ← float
    "dosage_mg":              df["dosage_mg"].values.astype(float),
    "ecog_performance_score": df["ecog_performance_score"].values.astype(float),
    "biomarker_pdl1":         df["biomarker_pdl1"].values.astype(float),
    "logP":                   df["logP"].values.astype(float),
    "actual_label":           df["efficacy_label"].values.astype(int),
    "model_name":             MODEL_NAME,
    "model_alias":            MODEL_ALIAS,
    "environment":            ENV,
    "batch_id":               datetime.now().strftime("%Y%m%d_%H%M%S"),
    "predicted_at":           datetime.now().isoformat(),
    "data_version":           "drift_v1",
    "is_correct":             (predictions ==
                               df["efficacy_label"].values).astype(bool),
})

    # Accuracy on drift data
    accuracy = df_results["is_correct"].mean()
    print(f"      Predictions:      {len(df_results):,}")
    print(f"      Accuracy on drift: {accuracy:.2%}  ← Should be lower!")
    print(f"      Baseline accuracy: ~94.96%")
    print(f"      Drop:              ~{(0.9496 - accuracy)*100:.1f}%")

    return df_results


# ════════════════════════════════════════════════════════
# STEP 4 — Save to prediction_logs
# Baseline + Drift dono same table mein
# Evidently AI compare karega
# ════════════════════════════════════════════════════════
def save_drift_predictions(df_results):
    print(f"\n[4/4] Saving drift predictions...")

    # Existing table ka schema match karo
    existing_schema = spark.table(PREDICTION_TABLE).schema

    # Pandas types ko existing schema ke according cast karo
    from pyspark.sql.types import LongType, DoubleType, IntegerType

    spark_df = spark.createDataFrame(df_results)

    # patient_age existing table jaisa cast karo
    for field in existing_schema.fields:
        if field.name in spark_df.columns:
            spark_df = spark_df.withColumn(
                field.name,
                F.col(field.name).cast(field.dataType)
            )

    spark_df.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable(PREDICTION_TABLE)

    # Summary
    summary = spark.sql(f"""
    SELECT
        data_version,
        COUNT(*)                                        as total,
        ROUND(AVG(CAST(is_correct AS INT)) * 100, 2)   as accuracy_pct,
        ROUND(AVG(molecular_weight), 1)                 as avg_mw,
        ROUND(AVG(CAST(patient_age AS DOUBLE)), 1)      as avg_age,
        ROUND(AVG(dosage_mg), 1)                        as avg_dosage
    FROM {PREDICTION_TABLE}
    GROUP BY data_version
    ORDER BY data_version
""")

    print(f"\n      Prediction Logs Summary:")
    summary.show(truncate=False)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
df         = generate_drift_data(N_ROWS)
model      = load_champion_model()
df_results = predict_on_drift(model, df)
save_drift_predictions(df_results)

print("\n" + "="*55)
print(" ✅ Drift Data Injected!")
print("="*55)
print(f"""
  prediction_logs now has:
  - baseline_v1: ~2,400 rows (normal distribution)
  - drift_v1:    ~2,000 rows (shifted distribution)

  Next:
  → Week 4 Day 3: Evidently AI drift detection
  → Compare baseline vs drift
  → PSI score calculate karo
  → Agent trigger setup
""")
print("="*55)