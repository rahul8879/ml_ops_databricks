import numpy as np
import pandas as pd
from databricks.connect import DatabricksSession
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
CATALOG  = "astrazeneca_dev"
SCHEMA   = "bronze"
TABLE    = "drug_raw"
SEED     = 42
N_ROWS   = 12000

np.random.seed(SEED)

# ── Cell line types ──────────────────────────────────────
CELL_LINES = ["NSCLC", "Breast", "CRC", "Leukemia", "Lymphoma"]

# ════════════════════════════════════════════════════════
# BASELINE DATA GENERATOR
# Pre-drift — normal distribution
# ════════════════════════════════════════════════════════
def generate_baseline(n=N_ROWS, seed=SEED):
    np.random.seed(seed)

    print(f"Generating {n} baseline records...")

    # ── Drug molecular features ──────────────────────────
    molecular_weight = np.random.normal(340, 60, n).clip(150, 500)
    logP             = np.random.normal(2.5, 1.2, n).clip(-3, 7)
    hbd_count        = np.random.randint(0, 6, n)
    hba_count        = np.random.randint(0, 8, n)
    rotatable_bonds  = np.random.randint(0, 10, n)
    tpsa             = np.random.normal(80, 20, n).clip(20, 160)

    # ── Patient features ─────────────────────────────────
    patient_age      = np.random.normal(54, 10, n).clip(25, 80).astype(int)
    patient_gender   = np.random.randint(0, 2, n)
    ecog_score       = np.random.choice([0,1,2,3], n, p=[0.3,0.4,0.2,0.1])
    prior_chemo      = np.random.randint(0, 2, n)
    biomarker_pdl1   = np.random.normal(45, 20, n).clip(0, 100)
    dosage_mg        = np.random.normal(125, 40, n).clip(50, 200)
    treatment_cycles = np.random.randint(1, 9, n)
    cell_line_type   = np.random.choice(CELL_LINES, n)

    # ── Label generation (realistic logic) ───────────────
    # Higher logP + lower MW + younger + lower ecog = more effective
    efficacy_score = (
        0.25 * (1 - (molecular_weight - 150) / 350) +  # smaller = better
        0.20 * (logP.clip(0,5) / 5) +                  # moderate logP good
        0.20 * (1 - ecog_score / 3) +                  # lower ecog = fitter
        0.15 * (biomarker_pdl1 / 100) +                # high PDL1 = better
        0.10 * (1 - (patient_age - 25) / 55) +         # younger = better
        0.10 * np.random.normal(0, 0.1, n)             # noise
    ).clip(0, 1)

    efficacy_label = (efficacy_score >= 0.5).astype(int)

    # ── Assemble DataFrame ───────────────────────────────
    df = pd.DataFrame({
        "molecular_weight":     molecular_weight.round(2),
        "logP":                 logP.round(3),
        "hbd_count":            hbd_count,
        "hba_count":            hba_count,
        "rotatable_bonds":      rotatable_bonds,
        "tpsa":                 tpsa.round(2),
        "cell_line_type":       cell_line_type,
        "patient_age":          patient_age,
        "patient_gender":       patient_gender,
        "ecog_performance_score": ecog_score,
        "prior_chemotherapy":   prior_chemo,
        "biomarker_pdl1":       biomarker_pdl1.round(2),
        "dosage_mg":            dosage_mg.round(1),
        "treatment_cycles":     treatment_cycles,
        "efficacy_score":       efficacy_score.round(4),
        "efficacy_label":       efficacy_label,
        "data_split":           np.where(
                                    np.arange(n) < int(n*0.8),
                                    "train", "validation"
                                ),
        "data_version":         "baseline_v1",
        "is_drift":             False
    })

    return df


# ════════════════════════════════════════════════════════
# SAVE TO UNITY CATALOG — BRONZE LAYER
# ════════════════════════════════════════════════════════
def save_to_bronze(df: pd.DataFrame, table_name: str):
    full_table = f"{CATALOG}.{SCHEMA}.{table_name}"
    print(f"\nSaving to {full_table}...")

    # Convert to Spark
    spark_df = spark.createDataFrame(df)

    # Save as Delta table
    spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(full_table)

    print(f"✅ Saved {df.shape[0]} rows to {full_table}")

    # Verify
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table}") \
                 .collect()[0]["cnt"]
    print(f"✅ Verified: {count} rows in table")

    return full_table


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*55)
    print(" AstraZeneca — Data Simulation Starting")
    print("="*55)

    # Generate baseline data
    df_baseline = generate_baseline(n=12000)

    # Quick stats
    print(f"\nBaseline Data Stats:")
    print(f"  Total rows:          {len(df_baseline)}")
    print(f"  Train rows:          {(df_baseline.data_split=='train').sum()}")
    print(f"  Validation rows:     {(df_baseline.data_split=='validation').sum()}")
    print(f"  Effective drugs (1): {df_baseline.efficacy_label.sum()}")
    print(f"  Not effective (0):   {(df_baseline.efficacy_label==0).sum()}")
    print(f"  Class balance:       {df_baseline.efficacy_label.mean():.2%}")
    print(f"  Avg mol. weight:     {df_baseline.molecular_weight.mean():.1f} Da")
    print(f"  Avg patient age:     {df_baseline.patient_age.mean():.1f} yrs")
    print(f"  Avg dosage:          {df_baseline.dosage_mg.mean():.1f} mg")

    # Save to Unity Catalog
    save_to_bronze(df_baseline, "drug_raw")

    print("\n" + "="*55)
    print("  Day 1 Complete!")
    print(" Next: Silver layer cleaning + drift data")
    print("="*55)