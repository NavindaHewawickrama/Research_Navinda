"""
Support Vector Machine (SVM) Baseline for MoA Prediction using LINCS L1000 + DRH data
Author: Nav Hewawickrama
Based on Random Forest baseline structure.
"""

import os
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress import parse, parse_gctx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump

# ---------------- CONFIGURATION ---------------- #
DATA_DIR = r"D:\4thyear\Research\Implementations\models\SVM"

LEVEL5_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
SIG_INFO_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_sig_info.txt")
PERT_INFO_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_pert_info.txt")
DRH_FILE = os.path.join(DATA_DIR, "repurposing_drugs_20200324.txt")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- STEP 1: LOAD GCTX METADATA ---------------- #
print("=== Loading metadata from GCTX file ===")
meta_only = parse_gctx.parse(LEVEL5_FILE, col_meta_only=True)
if hasattr(meta_only, "col_metadata_df"):
    all_sig_ids = meta_only.col_metadata_df.index.astype(str).tolist()
else:
    all_sig_ids = meta_only.index.astype(str).tolist()
print(f"✅ Total signatures available: {len(all_sig_ids):,}")

# ---------------- STEP 2: RANDOM SUBSET ---------------- #
np.random.seed(42)
subset_size = min(3000, len(all_sig_ids))
subset_ids = np.random.choice(all_sig_ids, size=subset_size, replace=False)
print(f"📦 Loading subset of {len(subset_ids):,} signatures...")

# ---------------- STEP 3: LOAD EXPRESSION MATRIX ---------------- #
gctoo = parse.parse(LEVEL5_FILE, cid=subset_ids)
expr = gctoo.data_df.T.copy()
expr.index.name = "sig_id"
expr.reset_index(inplace=True)
expr["sig_id"] = expr["sig_id"].astype(str)
print(f"✅ Expression matrix shape: {expr.shape}")

# ---------------- STEP 4: LOAD MoA + DRH METADATA ---------------- #
print("=== Loading MoA and Drug metadata ===")
sig = pd.read_csv(SIG_INFO_FILE, sep="\t", low_memory=False)
pert = pd.read_csv(PERT_INFO_FILE, sep="\t", low_memory=False)
drh = pd.read_csv(DRH_FILE, sep="\t", comment="!", low_memory=False)

pert_reduced = pert[["pert_id", "pert_type", "inchi_key", "pubchem_cid"]]
meta = sig.merge(pert_reduced, on="pert_id", how="left")
meta["sig_id"] = meta["sig_id"].astype(str)

meta = meta.merge(drh, on="pert_iname", how="left")
meta = meta.dropna(subset=["moa"])
print(f"✅ Metadata entries with MoA labels: {len(meta):,}")

# ---------------- STEP 5: ALIGN KEYS ---------------- #
expr["sig_id"] = expr["sig_id"].str.strip().str.upper()
meta["sig_id"] = meta["sig_id"].str.strip().str.upper()

common_keys = set(expr["sig_id"]).intersection(set(meta["sig_id"]))
print(f"🔗 Common sig_id matches found: {len(common_keys):,}")

full = expr.merge(meta, on="sig_id", how="inner")
print(f"✅ Merged dataset shape: {full.shape}")

# ---------------- STEP 6: FEATURE + LABEL PREPARATION ---------------- #
X = full.select_dtypes(include=[np.number])
y = full["moa"].astype(str)

# Filter out rare MoA classes
value_counts = y.value_counts()
rare_classes = value_counts[value_counts < 2].index
mask = ~y.isin(rare_classes)
X, y = X[mask], y[mask]

# Scale features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- STEP 7: SPLIT DATA ---------------- #
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
except ValueError as e:
    print(f"⚠️ Stratified split failed ({e}), using non-stratified split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=None
    )

print(f"🧪 Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

# ---------------- STEP 8: TRAIN SVM ---------------- #
print("=== Training Support Vector Machine (SVM) ===")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

# ---------------- STEP 9: EVALUATE MODEL ---------------- #
print("\n=== Evaluating Model ===")
y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1-score: {f1:.4f}")

unique_labels = np.unique(y_test)
target_names = [le.classes_[i] for i in unique_labels]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ---------------- STEP 10: SAVE RESULTS ---------------- #
pred_df = pd.DataFrame({
    "true_label": le.inverse_transform(y_test),
    "predicted_label": le.inverse_transform(y_pred)
})
pred_df.to_csv(os.path.join(RESULTS_DIR, "svm_predictions.csv"), index=False)
dump(svm, os.path.join(RESULTS_DIR, "svm_model.joblib"))
dump(scaler, os.path.join(RESULTS_DIR, "svm_scaler.joblib"))

# ---------------- SAVE TEST SET FOR EVALUATION ---------------- #
np.save(os.path.join(RESULTS_DIR, "x_test.npy"), X_test)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

print("\n📁 Saved test data for evaluation:")
print("   x_test.npy")
print("   y_test.npy")


print("\n✅ SVM baseline model and results saved successfully.")
print(f"📁 Predictions: {os.path.join(RESULTS_DIR, 'svm_predictions.csv')}")
print(f"📁 Model: {os.path.join(RESULTS_DIR, 'svm_model.joblib')}")
print("🎯 Done!")
