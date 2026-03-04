# src/training/train_baseline.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Baseline Model Training
# DAB spark_python_task ke through run hoga
# ═══════════════════════════════════════════════════════

import argparse
import os
import warnings
from xmlrpc import client

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import SparkSession
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
import mlflow
from mlflow.tracking import MlflowClient
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# STEP 0 — Parse DAB Parameters
# argparse — no widgets, no manual intervention
# ════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description="AstraZeneca Drug Efficacy Model Training"
)
parser.add_argument(
    "--catalog",
    type=str,
    default="astrazeneca_dev",
    help="Unity Catalog name"
)

parser.add_argument(
    "--experiment_name",
    type=str,
    default="/astrazeneca/dev/drug_efficacy",
    help="MLflow experiment path"
)
parser.add_argument(
    "--env",
    type=str,
    default="dev",
    help="Environment: dev / staging / prod"
)
args = parser.parse_args()

# ── Config from DAB ──────────────────────────────────────
CATALOG         = args.catalog
EXPERIMENT_NAME = args.experiment_name
ENV             = args.env

# ── Derived Config ───────────────────────────────────────
GOLD_TABLE      = f"{CATALOG}.gold.drug_features"
FEATURE_TABLE   = f"{CATALOG}.ml.drug_efficacy_features"
MODEL_NAME      = f"{CATALOG}.ml.drug_efficacy_model"
TARGET          = "efficacy_label"
RANDOM_STATE    = 42

# MLflow experiment — absolute path
EXPERIMENT_NAME = f"/Users/rtiwarirahul123@gmail.com/astrazeneca_{ENV}_drug_efficacy"

# ── Feature Columns ──────────────────────────────────────
FEATURE_COLS = [
    # Raw molecular
    "molecular_weight", "logP", "hbd_count",
    "hba_count", "rotatable_bonds", "tpsa",
    # Raw patient
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

# ── Spark Session ────────────────────────────────────────
spark = SparkSession.builder.getOrCreate()

print("\n" + "="*55)
print(" AstraZeneca — Baseline Model Training")
print("="*55)
print(f"  Environment: {ENV}")
print(f"  Catalog:     {CATALOG}")
print(f"  Experiment:  {EXPERIMENT_NAME}")
print(f"  Gold Table:  {GOLD_TABLE}")
print("="*55)


# ════════════════════════════════════════════════════════
# STEP 1 — Load Data from Gold Table
# ════════════════════════════════════════════════════════
def load_training_data():
    print("\n[1/4] Loading data from Gold table...")

    df_spark = spark.table(GOLD_TABLE)

    df_train = df_spark.filter(
        df_spark.data_split == "train"
    ).toPandas()

    df_val = df_spark.filter(
        df_spark.data_split == "validation"
    ).toPandas()

    print(f"      Train rows:      {len(df_train):,}")
    print(f"      Validation rows: {len(df_val):,}")
    print(f"      Features:        {len(FEATURE_COLS)}")
    print(f"      Class balance:   {df_train[TARGET].mean():.2%} positive")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET]
    X_val   = df_val[FEATURE_COLS]
    y_val   = df_val[TARGET]

    return X_train, y_train, X_val, y_val


# ════════════════════════════════════════════════════════
# STEP 2 — Handle Class Imbalance
# 35/65 imbalance — sample weights
# ════════════════════════════════════════════════════════
def get_sample_weights(y_train):
    print("\n[2/4] Handling class imbalance...")

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    counts = y_train.value_counts()
    print(f"      Class 0 (Not Effective): {counts[0]:,}")
    print(f"      Class 1 (Effective):     {counts[1]:,}")
    print(f"      Strategy: sample_weight=balanced")

    return weights


# ════════════════════════════════════════════════════════
# STEP 3 — Train XGBoost
# ════════════════════════════════════════════════════════
def train_model(X_train, y_train, X_val, y_val, weights):
    print("\n[3/4] Training XGBoost model...")

    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print("      ✅ Training complete!")
    return model, params


def save_run_metadata(run_id, metrics):
    print("\nSaving run metadata for next task...")

    run_metadata = spark.createDataFrame([{
        "run_id":          str(run_id),
        "environment":     str(ENV),
        "catalog":         str(CATALOG),
        "experiment_name": str(EXPERIMENT_NAME),
        "precision":       float(metrics["precision"]),
        "recall":          float(metrics["recall"]),
        "f1_score":        float(metrics["f1_score"]),
        "roc_auc":         float(metrics["roc_auc"]),
        "accuracy":        float(metrics["accuracy"]),
        "status":          "trained",
        "model_version":   "None",
        "created_at":      str(pd.Timestamp.now().isoformat()),
    }])

    run_metadata.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.monitoring.training_runs")

    print(f"✅ Run metadata saved → {CATALOG}.monitoring.training_runs")
    print(f"   run_id: {run_id}")


def evaluate_and_log(model, params, X_train, X_val, y_val):
    print("\n[4/4] Evaluating + Logging to MLflow...")

    y_pred      = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy":      accuracy_score(y_val, y_pred),
        "precision":     precision_score(y_val, y_pred),
        "recall":        recall_score(y_val, y_pred),
        "f1_score":      f1_score(y_val, y_pred),
        "roc_auc":       roc_auc_score(y_val, y_pred_prob),
        "positive_rate": float(y_val.mean()),
        "val_samples":   float(len(y_val)),
    }

    print(f"\n      {'Metric':<15} {'Value':>10}")
    print(f"      {'─'*25}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k:<15} {v:>10.4f}")

    cm = confusion_matrix(y_val, y_pred)
    print(f"\n      Confusion Matrix:")
    print(f"      TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"      FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\n      ⚠️  FP={cm[0][1]:,} drugs wrongly marked effective")

 
  # ── MLflow Logging ───────────────────────────────────
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    # Absolute path — /Users/username/experiment_name
    experiment_name = f"/Users/rtiwarirahul123@gmail.com/astrazeneca_{ENV}_drug_efficacy"

    # Experiment exist karta hai ya nahi
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)
        print(f"      ✅ Experiment created: {experiment_name}")
    else:
        print(f"      ✅ Experiment found: {experiment_name}")

    mlflow.set_experiment(experiment_name)
        
    with mlflow.start_run(run_name=f"xgboost_{ENV}_baseline_v1") as run:

        # Parameters
        mlflow.log_params(params)
        mlflow.log_param("catalog",            CATALOG)
        mlflow.log_param("environment",        ENV)
        mlflow.log_param("feature_count",      len(FEATURE_COLS))
        mlflow.log_param("train_samples",      len(X_train))
        mlflow.log_param("imbalance_strategy", "sample_weight_balanced")
        mlflow.log_param("gold_table",         GOLD_TABLE)

        # Metrics
        mlflow.log_metrics(metrics)

        # Model — log only, NO registration
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5],
        )

        # Feature importance artifact
        feat_imp = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        feat_imp.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")

        print(f"\n      Top 5 Features:")
        print(f"      {'Feature':<25} {'Importance':>12}")
        print(f"      {'─'*37}")
        for _, row in feat_imp.head(5).iterrows():
            print(f"      {row['feature']:<25} {row['importance']:>12.4f}")

        # Tags
        mlflow.set_tags({
            "model_type":   "xgboost",
            "environment":  ENV,
            "data_version": "baseline_v1",
            "catalog":      CATALOG,
            "status":       "logged_not_registered",
        })

        run_id = run.info.run_id

    print(f"\n      ✅ MLflow Run ID:  {run_id}")
    print(f"      ✅ Experiment:     {EXPERIMENT_NAME}")
    print(f"      ℹ️   Model NOT registered yet")
    print(f"      ℹ️   Review metrics → then register manually")

    return run_id, metrics


# ════════════════════════════════════════════════════════
# MAIN EXECUTION
# spark_python_task seedha execute karta hai
# ════════════════════════════════════════════════════════
X_train, y_train, X_val, y_val = load_training_data()
weights                         = get_sample_weights(y_train)
model, params                   = train_model(
                                    X_train, y_train,
                                    X_val, y_val,
                                    weights
                                  )
run_id, metrics                   = evaluate_and_log(
                                    model, params,
                                    X_train, X_val, y_val
                                  )
save_run_metadata(run_id, metrics)
print("\n" + "="*55)
print(" ✅ Training Complete!")
print("="*55)
print(f"""
  Run ID:     {run_id}
  Experiment: {EXPERIMENT_NAME}

  Next Steps:
  1. MLflow UI mein metrics review karo
     Experiments → {EXPERIMENT_NAME}

  2. Precision > 0.90? AUC > 0.95?
     → Haan: register_model.py chalao
     → Nahi: hyperparameters tune karo

  3. Model register karo (manual decision)
     src/training/register_model.py
""")
print("="*55)