"""
XGBoost Baseline for MoA Prediction using LINCS L1000 + DRH data
Author: Nav Hewawickrama
Stable version – follows same structure as RF + SVM baselines
"""

import os
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress import parse, parse_gctx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier
from joblib import dump

# ---------------- CONFIGURATION ---------------- #
DATA_DIR = r"D:\4thyear\Research\Implementations\models\dataset"
RESULT_DIR = r"D:\4thyear\Research\Implementations\models\XGB"

LEVEL5_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
SIG_INFO_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_sig_info.txt")
PERT_INFO_FILE = os.path.join(DATA_DIR, "GSE92742_Broad_LINCS_pert_info.txt")
DRH_FILE = os.path.join(DATA_DIR, "repurposing_drugs_20200324.txt")

RESULTS_DIR = os.path.join(RESULT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------ #
print("=== Loading metadata from GCTX file ===")
meta_only = parse_gctx.parse(LEVEL5_FILE, col_meta_only=True)
if hasattr(meta_only, "col_metadata_df"):
    all_sig_ids = meta_only.col_metadata_df.index.astype(str).tolist()
else:
    all_sig_ids = meta_only.index.astype(str).tolist()
print(f"✅ Total signatures available: {len(all_sig_ids):,}")

# ---------------- SUBSET ---------------- #
np.random.seed(42)
subset_size = min(3000, len(all_sig_ids))
subset_ids = np.random.choice(all_sig_ids, size=subset_size, replace=False)
print(f"📦 Loading subset of {subset_size} signatures...")

# ---------------- LOAD EXPRESSIONS ---------------- #
gctoo = parse.parse(LEVEL5_FILE, cid=subset_ids)
expr = gctoo.data_df.T.copy()
expr.index.name = "sig_id"
expr.reset_index(inplace=True)
expr["sig_id"] = expr["sig_id"].astype(str)
print(f"✅ Expression matrix shape: {expr.shape}")

# ---------------- LOAD METADATA ---------------- #
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

# ---------------- MATCH KEYS ---------------- #
expr["sig_id"] = expr["sig_id"].str.strip().str.upper()
meta["sig_id"] = meta["sig_id"].str.strip().str.upper()

common_keys = set(expr["sig_id"]).intersection(set(meta["sig_id"]))
print(f"🔗 Common sig_id matches found: {len(common_keys):,}")

if len(common_keys) == 0:
    raise ValueError("No matching sig_id values between expression and metadata!")

# ---------------- MERGE ---------------- #
full = expr.merge(meta, on="sig_id", how="inner")
print(f"✅ Merged dataset shape: {full.shape}")

# ---------------- PREP FEATURES + LABELS ---------------- #
X = full.select_dtypes(include=[np.number])
y = full["moa"].astype(str)

# Remove rare MoA classes (<2 samples)
value_counts = y.value_counts()
rare = value_counts[value_counts < 2].index

if len(rare) > 0:
    print(f"⚠️ Removing {len(rare)} rare classes")
    mask = ~y.isin(rare)
    X = X[mask]
    y = y[mask]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- SCALING ---------------- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN/TEST SPLIT ---------------- #
n_classes = len(np.unique(y_encoded))
test_size = max(0.2, n_classes / len(y_encoded))
test_size = min(test_size, 0.5)  # avoid > 50%

print(f"📊 Adjusted test size: {test_size:.2f}")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )
except:
    print("⚠️ Stratified split failed — using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42
    )

print(f"🧪 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------- TRAIN XGBOOST ---------------- #
print("=== Training XGBoost Classifier ===")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

xgb.fit(X_train, y_train)

# ---------------- EVALUATE ---------------- #
print("\n=== Evaluating Model ===")
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1-score: {f1:.4f}")

unique = np.unique(y_test)
target_names = [le.classes_[i] for i in unique]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=target_names))

# ---------------- SAVE RESULTS ---------------- #
pred_df = pd.DataFrame({
    "true_label": le.inverse_transform(y_test),
    "predicted_label": le.inverse_transform(y_pred)
})

pred_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_predictions.csv"), index=False)
dump(xgb, os.path.join(RESULTS_DIR, "xgboost_model.joblib"))
# ---------------- SAVE TEST SET FOR EVALUATION ---------------- #
np.save(os.path.join(RESULTS_DIR, "x_test.npy"), X_test)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

print("\n📁 Saved test data for evaluation:")
print("   x_test.npy")
print("   y_test.npy")

print("\n✅ XGBoost baseline model and results saved.")
print(f"📁 Predictions: {os.path.join(RESULTS_DIR, 'xgboost_predictions.csv')}")
print(f"📁 Model: {os.path.join(RESULTS_DIR, 'xgboost_model.joblib')}")
print("🎯 Done!")
