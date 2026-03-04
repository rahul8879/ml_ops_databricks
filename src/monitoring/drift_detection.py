# src/monitoring/drift_detection.py
# ═══════════════════════════════════════════════════════
# AstraZeneca — Evidently AI Drift Detection
# baseline_v1 vs drift_v1 compare karo
# PSI + drift report Delta Lake mein save karo
# ═══════════════════════════════════════════════════════

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
)

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
PREDICTION_TABLE = f"{CATALOG}.monitoring.prediction_logs"
DRIFT_REPORT_TABLE = f"{CATALOG}.monitoring.drift_reports"
DRIFT_LOG_TABLE    = f"{CATALOG}.monitoring.drift_logs"

# ── Features to monitor ──────────────────────────────────
DRIFT_FEATURES = [
    "molecular_weight",
    "patient_age",
    "dosage_mg",
    "ecog_performance_score",
    "biomarker_pdl1",
    "logP",
]

# ── PSI Thresholds ───────────────────────────────────────
PSI_WARNING  = 0.10
PSI_MODERATE = 0.20
PSI_SEVERE   = 0.25
PSI_CRITICAL = 0.35

print("\n" + "="*55)
print(" AstraZeneca — Drift Detection")
print("="*55)
print(f"  Catalog:    {CATALOG}")
print(f"  Reference:  baseline_v1")
print(f"  Current:    drift_v1")
print("="*55)


# ════════════════════════════════════════════════════════
# STEP 1 — Reference + Current Data Load
# ════════════════════════════════════════════════════════
def load_data():
    print("\n[1/4] Loading reference + current data...")

    df_all = spark.table(PREDICTION_TABLE)

    # Reference — baseline data
    df_ref = df_all.filter(
        F.col("data_version") == "baseline_v1"
    ).select(DRIFT_FEATURES).toPandas()

    # Current — drift data
    df_cur = df_all.filter(
        F.col("data_version") == "drift_v1"
    ).select(DRIFT_FEATURES).toPandas()

    print(f"      Reference rows: {len(df_ref):,}")
    print(f"      Current rows:   {len(df_cur):,}")

    # Quick distribution comparison
    print(f"\n      {'Feature':<22} {'Ref Mean':>10} {'Cur Mean':>10} {'Diff':>8}")
    print(f"      {'─'*52}")
    for feat in DRIFT_FEATURES:
        ref_mean = df_ref[feat].mean()
        cur_mean = df_cur[feat].mean()
        diff     = cur_mean - ref_mean
        arrow    = "↑" if diff > 0 else "↓"
        print(f"      {feat:<22} {ref_mean:>10.2f} {cur_mean:>10.2f} "
              f"{arrow}{abs(diff):>6.2f}")

    return df_ref, df_cur


# ════════════════════════════════════════════════════════
# STEP 2 — Evidently AI Report
# ════════════════════════════════════════════════════════
from evidently.calculations.stattests import StatTest
from evidently.metrics import ColumnDriftMetric

def run_evidently_report(df_ref, df_cur):
    print("\n[2/4] Running Evidently AI drift report...")

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(stattest="psi",          # ← PSI explicitly
                        stattest_threshold=0.25),
        ColumnDriftMetric(column_name="molecular_weight",
                          stattest="psi",
                          stattest_threshold=0.25),
        ColumnDriftMetric(column_name="patient_age",
                          stattest="psi",
                          stattest_threshold=0.25),
        ColumnDriftMetric(column_name="dosage_mg",
                          stattest="psi",
                          stattest_threshold=0.25),
        ColumnDriftMetric(column_name="ecog_performance_score",
                          stattest="psi",
                          stattest_threshold=0.25),
        ColumnDriftMetric(column_name="biomarker_pdl1",
                          stattest="psi",
                          stattest_threshold=0.25),
        ColumnDriftMetric(column_name="logP",
                          stattest="psi",
                          stattest_threshold=0.25),
    ])

    report.run(
        reference_data=df_ref,
        current_data=df_cur
    )

    print("      ✅ Evidently report complete!")
    return report, report.as_dict()


# ════════════════════════════════════════════════════════
# STEP 3 — PSI + Drift Summary
# ════════════════════════════════════════════════════════
def extract_drift_summary(results):
    print("\n[3/4] Extracting drift metrics...")

    metrics         = results["metrics"]
    dataset_metric  = metrics[0]["result"]
    dataset_drifted = dataset_metric.get("dataset_drift", False)
    share_drifted   = dataset_metric.get("share_of_drifted_columns", 0)

    print(f"\n      Dataset Drift Detected: {dataset_drifted}")
    print(f"      Share of drifted cols:  {share_drifted:.1%}")

    print(f"\n      {'Feature':<22} {'PSI':>8} {'Drifted':>10} {'Severity':>12}")
    print(f"      {'─'*54}")

    drift_summary = {}
    col_metrics   = metrics[2:]  # Skip dataset level

    for m in col_metrics:
        col_name = m["result"].get("column_name", None)

        # ── "unknown" skip karo ──────────────────────────
        if col_name is None or col_name == "unknown":
            continue

        score   = m["result"].get("drift_score", 0)
        drifted = m["result"].get("drift_detected", False)

        if score >= PSI_CRITICAL:
            severity = "🔴 CRITICAL"
        elif score >= PSI_SEVERE:
            severity = "🟠 SEVERE"
        elif score >= PSI_MODERATE:
            severity = "🟡 MODERATE"
        elif score >= PSI_WARNING:
            severity = "🟢 WARNING"
        else:
            severity = "✅ STABLE"

        drift_summary[col_name] = {
            "score":    round(score, 4),
            "drifted":  drifted,
            "severity": severity.split(" ")[1]
        }

        print(f"      {col_name:<22} {score:>8.4f} "
              f"{'YES' if drifted else 'NO':>10} {severity:>12}")

    max_psi     = max([v["score"] for v in drift_summary.values()])
    max_feature = max(drift_summary, key=lambda x: drift_summary[x]["score"])

    print(f"\n      Max PSI:  {max_psi:.4f} ({max_feature})")

    if max_psi >= PSI_CRITICAL:
        action = "IMMEDIATE_ROLLBACK"
        print(f"      🔴 CRITICAL — Rollback required!")
    elif max_psi >= PSI_SEVERE:
        action = "RETRAIN"
        print(f"      🟠 SEVERE — Retraining required!")
    elif max_psi >= PSI_MODERATE:
        action = "SCHEDULE_RETRAIN"
        print(f"      🟡 MODERATE — Schedule retraining")
    elif max_psi >= PSI_WARNING:
        action = "MONITOR"
        print(f"      🟢 WARNING — Increase monitoring")
    else:
        action = "NO_ACTION"
        print(f"      ✅ STABLE — No action needed")

    return drift_summary, max_psi, action, dataset_drifted


# ════════════════════════════════════════════════════════
# STEP 4 — Save Drift Report to Delta Lake
# Agent yahan se padhega
# ════════════════════════════════════════════════════════
def save_drift_report(drift_summary, max_psi, action, dataset_drifted):
    print(f"\n[4/4] Saving drift report to Delta Lake...")

    now = datetime.now().isoformat()

    # Drift log — ek row per feature
    drift_rows = []
    for feature, stats in drift_summary.items():
        drift_rows.append({
            "detected_at":       now,
            "environment":       ENV,
            "catalog":           CATALOG,
            "reference_version": "baseline_v1",
            "current_version":   "drift_v1",
            "feature":           feature,
            "drift_score":       float(stats["score"]),
            "drift_detected":    bool(stats["drifted"]),
            "severity":          stats["severity"],
            "max_psi":           float(max_psi),
            "recommended_action": action,
            "dataset_drifted":   bool(dataset_drifted),
            "status":            "new",
        })

    df_drift = pd.DataFrame(drift_rows)
    spark_df = spark.createDataFrame(df_drift)

    spark_df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(DRIFT_LOG_TABLE)

    print(f"      ✅ Drift log saved: {DRIFT_LOG_TABLE}")
    print(f"      ✅ {len(drift_rows)} feature reports saved")

    # Summary table
    spark.sql(f"""
        SELECT
            feature,
            ROUND(drift_score, 4)  as psi_score,
            drift_detected,
            severity,
            recommended_action
        FROM {DRIFT_LOG_TABLE}
        WHERE detected_at = '{now}'
        ORDER BY drift_score DESC
    """).show(truncate=False)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
df_ref, df_cur              = load_data()
report, results             = run_evidently_report(df_ref, df_cur)
drift_summary, max_psi, \
action, dataset_drifted     = extract_drift_summary(results)
save_drift_report(
    drift_summary, max_psi,
    action, dataset_drifted
)

print("\n" + "="*55)