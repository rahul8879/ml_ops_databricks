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

    # ── Generate all arrays ──────────────────────────────
    mw      = np.random.normal(460, 80, n).clip(200, 700)
    age     = np.random.normal(61, 11, n).clip(30, 85).astype(int)
    dose    = np.random.normal(210, 60, n).clip(75, 350)
    logp    = np.random.normal(2.5, 1.2, n).clip(-3, 7)
    hbd     = np.random.randint(0, 6, n)
    hba     = np.random.randint(0, 8, n)
    rot     = np.random.randint(0, 10, n)
    tpsa    = np.random.normal(80, 20, n).clip(20, 160)
    gender  = np.random.randint(0, 2, n)
    chemo   = np.random.randint(0, 2, n)
    pdl1    = np.random.normal(45, 20, n).clip(0, 100)
    cycles  = np.random.randint(1, 9, n)
    ecog    = np.random.choice([0,1,2,3], n, p=[0.20,0.35,0.30,0.15])

    cell_line_type  = np.random.choice(CELL_LINES, n)
    cell_line_map   = {"NSCLC":0,"Breast":1,"CRC":2,"Leukemia":3,"Lymphoma":4}
    cli             = np.array([cell_line_map.get(c, 0) for c in cell_line_type])

    age_grp = np.where(age < 40, "young", np.where(age < 60, "middle", "senior"))
    agi     = np.array([{"young":0,"middle":1,"senior":2}.get(a,1) for a in age_grp])

    lip     = ((mw<500).astype(int)+(logp<5).astype(int)+
               (hbd<5).astype(int)+(hba<10).astype(int)).astype(float)
    risk    = ((age/80.0)*0.4 + (ecog/4.0)*0.4 + chemo*0.2).round(4)
    dntpsa  = (tpsa/dose).round(4)
    pdl1f   = (pdl1>50).astype(int)

    mw_norm   = ((mw-100)/700).round(4)
    logp_norm = ((logp-(-5))/15).round(4)
    tpsa_norm = (tpsa/200).round(4)
    pdl1_norm = (pdl1/100).round(4)
    dose_norm = ((dose-10)/390).round(4)
    age_norm  = ((age-18)/72).round(4)

    mw_penalty = np.where(mw > 500, 0.15, 0.0)
    eff_score  = (
        0.25*(1-(mw-150)/350) + 0.20*(logp.clip(0,5)/5) +
        0.20*(1-ecog/3) + 0.15*(pdl1/100) +
        0.10*(1-(age-25)/55) + 0.10*np.random.normal(0,0.1,n) - mw_penalty
    ).clip(0,1)
    label = (eff_score >= 0.5).astype(int)

    # ── Build DataFrame explicitly ───────────────────────
    df = pd.DataFrame({
        "molecular_weight":        mw.astype(float),
        "logP":                    logp.astype(float),
        "hbd_count":               hbd.astype(float),
        "hba_count":               hba.astype(float),
        "rotatable_bonds":         rot.astype(float),
        "tpsa":                    tpsa.astype(float),
        "patient_age":             age.astype(float),
        "patient_gender":          gender.astype(float),
        "ecog_performance_score":  ecog.astype(float),
        "prior_chemotherapy":      chemo.astype(float),
        "biomarker_pdl1":          pdl1.astype(float),
        "dosage_mg":               dose.astype(float),
        "treatment_cycles":        cycles.astype(float),
        "cell_line_index":         cli.astype(float),
        "age_group_index":         agi.astype(float),
        "lipinski_score":          lip,
        "patient_risk_score":      risk,
        "dose_normalized_tpsa":    dntpsa,
        "high_pdl1_flag":          pdl1f.astype(float),
        "molecular_weight_norm":   mw_norm,
        "logP_norm":               logp_norm,
        "tpsa_norm":               tpsa_norm,
        "biomarker_pdl1_norm":     pdl1_norm,
        "dosage_mg_norm":          dose_norm,
        "patient_age_norm":        age_norm,
        "efficacy_label":          label,
        "data_version":            "drift_v1",
    })

    print(f"      ✅ Generated {len(df):,} records")
    print(f"      Avg MW:   {df['molecular_weight'].mean():.1f} Da")
    print(f"      Avg Age:  {df['patient_age'].mean():.1f} yrs")
    print(f"      Avg Dose: {df['dosage_mg'].mean():.1f} mg")

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

    # Debug
    print(f"      Sample input:")
    print(df[["molecular_weight", "patient_age", "dosage_mg"]].head(3))

    X                = df[FEATURE_COLS]
    predictions      = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]

    # ── Explicitly from df — no ambiguity ────────────────
    mw      = df["molecular_weight"].tolist()
    age     = df["patient_age"].tolist()
    dose    = df["dosage_mg"].tolist()
    ecog    = df["ecog_performance_score"].tolist()
    pdl1    = df["biomarker_pdl1"].tolist()
    logp    = df["logP"].tolist()
    actual  = df["efficacy_label"].tolist()
    dv      = df["data_version"].tolist()

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
        "molecular_weight":       mw,       # ← explicit
        "patient_age":            [float(a) for a in age],
        "dosage_mg":              dose,
        "ecog_performance_score": [float(e) for e in ecog],
        "biomarker_pdl1":         pdl1,
        "logP":                   logp,
        "actual_label":           [int(a) for a in actual],
        "model_name":             MODEL_NAME,
        "model_alias":            MODEL_ALIAS,
        "environment":            ENV,
        "batch_id":               datetime.now().strftime("%Y%m%d_%H%M%S"),
        "predicted_at":           datetime.now().isoformat(),
        "data_version":           dv,
        "is_correct":             [bool(p == a)
                                   for p, a in zip(predictions, actual)],
    })

    accuracy = sum(df_results["is_correct"]) / len(df_results)
    print(f"      Predictions:       {len(df_results):,}")
    print(f"      Accuracy on drift: {accuracy:.2%}")
    print(f"      Sample results:")
    print(df_results[["molecular_weight", "patient_age",
                       "dosage_mg", "prediction"]].head(3))

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