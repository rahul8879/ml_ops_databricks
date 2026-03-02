# src/training/train_baseline.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Baseline Model Training
# XGBoost + MLflow + Unity Catalog
#
# RUN KARO: Databricks Notebook pe (local nahi)
# ═══════════════════════════════════════════════════════

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")

# ── Only import Spark if running on Databricks ───────────
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    IS_DATABRICKS = True
    print("✅ Running on Databricks")
except Exception:
    IS_DATABRICKS = False
    print("⚠️  Running locally — use Databricks for training")

# ── Config ───────────────────────────────────────────────
CATALOG          = "astrazeneca_dev"
FEATURE_TABLE    = f"{CATALOG}.ml.drug_efficacy_features"
GOLD_TABLE       = f"{CATALOG}.gold.drug_features"
EXPERIMENT_NAME  = "/astrazeneca/dev/drug_efficacy"
MODEL_NAME       = f"{CATALOG}.ml.drug_efficacy_model"
TARGET           = "efficacy_label"
RANDOM_STATE     = 42

# ── Feature columns ──────────────────────────────────────
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

# ════════════════════════════════════════════════════════
# STEP 1 — Load Data from Feature Store
# ════════════════════════════════════════════════════════
def load_training_data():
    print("\n[1/5] Loading data from Feature Store...")

    # Feature Store se load karo
    df_spark = spark.table(GOLD_TABLE)

    # Train split only
    df_train = df_spark.filter(
        df_spark.data_split == "train"
    ).toPandas()

    df_val = df_spark.filter(
        df_spark.data_split == "validation"
    ).toPandas()

    print(f"      Train rows:      {len(df_train):,}")
    print(f"      Validation rows: {len(df_val):,}")
    print(f"      Features:        {len(FEATURE_COLS)}")
    print(f"      Class balance:   "
          f"{df_train[TARGET].mean():.2%} positive")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET]
    X_val   = df_val[FEATURE_COLS]
    y_val   = df_val[TARGET]

    return X_train, y_train, X_val, y_val


# ════════════════════════════════════════════════════════
# STEP 2 — Handle Class Imbalance
# 35/65 imbalance — sample weights use karenge
# ════════════════════════════════════════════════════════
def get_sample_weights(y_train):
    print("\n[2/5] Handling class imbalance...")

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    class_counts = y_train.value_counts()
    print(f"      Class 0 (Not Effective): {class_counts[0]:,}")
    print(f"      Class 1 (Effective):     {class_counts[1]:,}")
    print(f"      Strategy: sample_weight='balanced'")

    return weights


# ════════════════════════════════════════════════════════
# STEP 3 — Train XGBoost Model
# ════════════════════════════════════════════════════════
def train_model(X_train, y_train, X_val, y_val, weights):
    print("\n[3/5] Training XGBoost model...")

    # Hyperparameters
    params = {
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  5,
        "gamma":             0.1,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=50              # Print every 50 rounds
    )

    print("      ✅ Training complete!")
    return model, params


# ════════════════════════════════════════════════════════
# STEP 4 — Evaluate Model
# Primary metric: Precision (pharma mein FP costly)
# ════════════════════════════════════════════════════════
def evaluate_model(model, X_val, y_val):
    print("\n[4/5] Evaluating model...")

    y_pred      = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy":          accuracy_score(y_val, y_pred),
        "precision":         precision_score(y_val, y_pred),
        "recall":            recall_score(y_val, y_pred),
        "f1_score":          f1_score(y_val, y_pred),
        "roc_auc":           roc_auc_score(y_val, y_pred_prob),
        "val_samples":       len(y_val),
        "positive_rate":     float(y_val.mean()),
    }

    print(f"\n      {'Metric':<15} {'Value':>10}")
    print(f"      {'─'*25}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k:<15} {v:>10.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n      Confusion Matrix:")
    print(f"      TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"      FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\n      ⚠️  FP={cm[0][1]:,} — "
          f"drugs wrongly marked effective")

    return metrics, cm


# ════════════════════════════════════════════════════════
# STEP 5 — MLflow Logging
# ════════════════════════════════════════════════════════
def log_to_mlflow(model, params, metrics, X_train):
    print("\n[5/5] Logging to MLflow...")

    # Experiment set karo
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_baseline_v1") as run:

        # ── Log Parameters ───────────────────────────────
        mlflow.log_params(params)
        mlflow.log_param("feature_count",      len(FEATURE_COLS))
        mlflow.log_param("train_samples",      len(X_train))
        mlflow.log_param("imbalance_strategy", "sample_weight_balanced")
        mlflow.log_param("catalog",            CATALOG)
        mlflow.log_param("feature_table",      FEATURE_TABLE)

        # ── Log Metrics ──────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Log Model (without registering inside run) ───
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5],
        )

        # ── Log Feature Importance ───────────────────────
        feat_imp = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\n      Top 5 Important Features:")
        print(f"      {'Feature':<25} {'Importance':>12}")
        print(f"      {'─'*37}")
        for _, row in feat_imp.head(5).iterrows():
            print(f"      {row.feature:<25} {row.importance:>12.4f}")

        # ── Tags ─────────────────────────────────────────
        mlflow.set_tags({
            "model_type":   "xgboost",
            "use_case":     "drug_efficacy_prediction",
            "data_version": "baseline_v1",
            "environment":  "dev",
            "week":         "week3_day1",
        })

        run_id = run.info.run_id
        print(f"\n      ✅ Run ID: {run_id}")

    # ── Register AFTER run completes ─────────────────────
    # Unity Catalog ke saath run ke baad register karna chahiye
    print(f"\n      Registering in Unity Catalog...")
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        print(f"      ✅ Model registered: {MODEL_NAME}")
    except Exception as e:
        print(f"      ⚠️  Registration issue: {e}")
        print(f"      Manual register karo:")
        print(f"      URI: runs:/{run_id}/model")

    return run_id


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    if not IS_DATABRICKS:
        print("""
╔══════════════════════════════════════════════════╗
║  ⚠️  Run this on Databricks, not locally!        ║
║                                                  ║
║  Steps:                                          ║
║  1. Git push karo                                ║
║  2. Databricks Repo pull karo                    ║
║  3. New notebook banao                           ║
║  4. Yeh run karo:                                ║
║     %run ./src/training/train_baseline           ║
╚══════════════════════════════════════════════════╝
        """)
    else:
        print("\n" + "="*55)
        print(" AstraZeneca — Baseline Model Training")
        print("="*55)

        X_train, y_train, X_val, y_val = load_training_data()
        weights                         = get_sample_weights(y_train)
        model, params                   = train_model(
                                            X_train, y_train,
                                            X_val, y_val, weights
                                          )
        metrics, cm                     = evaluate_model(
                                            model, X_val, y_val
                                          )
        run_id                          = log_to_mlflow(
                                            model, params,
                                            metrics, X_train
                                          )

        print("\n" + "="*55)
        print(" ✅ Week 3 Day 1 Complete!")
        print("="*55)
        print(f"""
  MLflow Experiment: {EXPERIMENT_NAME}
  Run ID:            {run_id}
  Model:             {MODEL_NAME}

  Next Steps:
  → Databricks Experiments tab mein dekho
  → Run metrics compare karo
  → Day 2: Hyperparameter tuning
        """)
        print("="*55)