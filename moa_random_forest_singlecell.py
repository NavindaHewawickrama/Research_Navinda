"""
Corrected Random Forest Baseline for MoA Prediction
✓ Single cell line (MCF7)
✓ Samples BEFORE loading GCTX (no full 473k load)
✓ Rare class removal (<10)
✓ Valid sig_id matching
✓ Progress steps + timing
Author: Nav Hewawickrama (Updated)
"""

import pandas as pd
import numpy as np
import time
from cmapPy.pandasGEXpress import parse, parse_gctx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump
import os

# ---------------- CONFIG ---------------- #
DATA_DIR = "D:/4thyear/Research/Implementations/models/RFB"

LEVEL5_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIG_INFO_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_sig_info.txt"
PERT_INFO_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_pert_info.txt"
DRH_FILE = f"{DATA_DIR}/repurposing_drugs_20200324.txt"

RESULTS_DIR = f"{DATA_DIR}/resultsnew"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_CELL_LINE = "MCF7"
SAMPLE_SIZE = 60000          # number of signatures to load
MIN_SAMPLES_PER_CLASS = 10    # drop MoA classes with less than 10 samples
TEST_SIZE = 0.2
N_ESTIMATORS = 300
RANDOM_SEED = 42


# ---------------- STEP 1: LOAD VALID GCTX SIG_IDS ---------------- #
print("=== STEP 1: Loading valid sig_ids from GCTX metadata ===")
t0 = time.time()

meta_only = parse_gctx.parse(LEVEL5_FILE, col_meta_only=True)

if hasattr(meta_only, "col_metadata_df"):
    valid_sig_ids = meta_only.col_metadata_df.index.astype(str)
else:
    valid_sig_ids = meta_only.index.astype(str)

valid_sig_ids = {x.strip().upper() for x in valid_sig_ids}

print(f"✔ Valid sig_ids in GCTX: {len(valid_sig_ids):,}")
print(f"⏳ Time: {time.time() - t0:.2f}s")


# ---------------- STEP 2: LOAD METADATA ---------------- #
print("\n=== STEP 2: Loading metadata ===")
sig = pd.read_csv(SIG_INFO_FILE, sep="\t", low_memory=False, dtype=str)
pert = pd.read_csv(PERT_INFO_FILE, sep="\t", low_memory=False, dtype=str)
drh = pd.read_csv(DRH_FILE, sep="\t", comment="!", low_memory=False, dtype=str)

# Normalize sig_info formats
sig["sig_id"] = sig["sig_id"].astype(str).str.strip().str.upper()
sig["cell_id"] = sig["cell_id"].astype(str).str.strip().str.upper()

TARGET = TARGET_CELL_LINE.strip().upper()
sig = sig[sig["cell_id"] == TARGET]

print(f"✔ Total MCF7 sig_info entries: {len(sig):,}")

# Keep sig_ids that exist in GCTX
sig = sig[sig["sig_id"].isin(valid_sig_ids)]
print(f"✔ MCF7 sig_ids valid in GCTX: {len(sig):,}")


# ---------------- STEP 3: SAMPLE BEFORE LOADING EXPRESSION ---------------- #
print("\n=== STEP 3: Sampling signatures BEFORE loading expression ===")

available = len(sig)
take_n = min(SAMPLE_SIZE, available)

print(f"Available for MCF7: {available:,}")
print(f"Sampling: {take_n:,}")

sig_sample = sig.sample(n=take_n, random_state=RANDOM_SEED)
sample_sig_ids = sig_sample["sig_id"].tolist()


# ---------------- STEP 4: LOAD EXPRESSION MATRIX ---------------- #
print("\n=== STEP 4: Loading expression data (this may take a moment) ===")
t0 = time.time()

gctoo = parse.parse(LEVEL5_FILE, cid=sample_sig_ids)

# ALTERNATIVE APPROACH: Don't transpose, work with the original structure
expr = gctoo.data_df.copy()  # Remove .T

# The expression data is now genes x samples
# We need to transpose it to samples x genes for ML
expr = expr.T

# Now create sig_id column from the index (which should be the sample IDs)
expr = expr.reset_index()
if "index" in expr.columns:
    expr = expr.rename(columns={"index": "sig_id"})
elif "rid" in expr.columns:
    expr = expr.rename(columns={"rid": "sig_id"})
else:
    # If we don't know the column name, use the first column
    first_col = expr.columns[0]
    expr = expr.rename(columns={first_col: "sig_id"})

expr["sig_id"] = expr["sig_id"].astype(str).str.strip().str.upper()

print("✔ Expression matrix after processing:", expr.shape)
print(f"⏳ Time: {time.time() - t0:.2f}s")


# ---------------- STEP 5: MERGE METADATA + DROP NA MOA ---------------- #
print("\n=== STEP 5: Merging metadata (pert + DRH) ===")

pert_reduced = pert[["pert_id", "pert_type", "inchi_key", "pubchem_cid"]]

meta = sig_sample.merge(pert_reduced, on="pert_id", how="left")
meta = meta.merge(drh, on="pert_iname", how="left")
meta = meta.dropna(subset=["moa"])

meta["sig_id"] = meta["sig_id"].astype(str).str.strip().str.upper()

print(f"✔ Metadata entries with MoA: {len(meta):,}")


# ---------------- STEP 6: MERGE EXPRESSION + META ---------------- #
print("\n=== STEP 6: Merging expression + metadata ===")

full = expr.merge(meta, on="sig_id", how="inner")
print(f"✔ Final merged dataset: {full.shape}")

if full.shape[0] == 0:
    raise RuntimeError("❌ No rows after merging! Check sig_id casing differences.")


# ---------------- STEP 7: FEATURE + LABELS ---------------- #
print("\n=== STEP 7: Preparing features & labels ===")

X = full.select_dtypes(include=[np.number])
y = full["moa"].astype(str)

value_counts = y.value_counts()
rare = value_counts[value_counts < MIN_SAMPLES_PER_CLASS].index

print(f"Removing {len(rare)} rare MoA classes (<{MIN_SAMPLES_PER_CLASS})")

mask = ~y.isin(rare)
X = X[mask]
y = y[mask]

print("✔ Final usable samples:", len(y))

if len(y) == 0:
    raise RuntimeError("❌ All classes removed. Lower MIN_SAMPLES_PER_CLASS.")


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ---------------- STEP 8: TRAIN/TEST SPLIT ---------------- #
print("\n=== STEP 8: Train/Test Split ===")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )
except:
    print("⚠ Stratified split failed. Falling back to non-stratified.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")


# ---------------- STEP 9: TRAIN MODEL ---------------- #
print("\n=== STEP 9: Training Random Forest ===")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------- STEP 10: EVALUATE ---------------- #
print("\n=== STEP 10: Evaluation ===")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
target_names = le.inverse_transform(np.unique(y_test))
print(classification_report(y_test, y_pred, target_names=target_names))


# ---------------- STEP 11: SAVE RESULTS ---------------- #
pred_df = pd.DataFrame({
    "true_label": le.inverse_transform(y_test),
    "predicted_label": le.inverse_transform(y_pred)
})

pred_df.to_csv(f"{RESULTS_DIR}/rf_predictions_sampled.csv", index=False)
dump(model, f"{RESULTS_DIR}/rf_model_sampled.joblib")

print("\n✔ DONE! Saved results to:", RESULTS_DIR)