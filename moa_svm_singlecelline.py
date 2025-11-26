"""
Literature-Accurate SVM Pipeline for MoA Prediction
Author: Nav Hewawickrama
Matches MoA prediction methodology used in:
 - Iorio et al.
 - Aliper et al.
 - Wang et al.
 - LINCS MoA baseline pipelines
"""

import os
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress import parse, parse_gctx

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump

# ============================== CONFIG ============================== #

DATA_DIR = r"D:\4thyear\Research\Implementations\models\SVM"

LEVEL5_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIG_INFO_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_sig_info.txt"
PERT_INFO_FILE = f"{DATA_DIR}/GSE92742_Broad_LINCS_pert_info.txt"
DRH_FILE = f"{DATA_DIR}/repurposing_drugs_20200324.txt"

RESULTS_DIR = f"{DATA_DIR}/results_litSVM"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_CELL_LINE = "MCF7"
TOP_N_MOA = 15                 # literature picks ~10–20 major MoA classes
MAX_SAMPLES = 3000            # safe for 8GB RAM
PCA_COMPONENTS = 300           # consistent with literature

# ============================== STEP 1: Load GCTX sig_ids ============================== #

print("=== STEP 1: Loading valid sig_ids from GCTX ===")
meta_only = parse_gctx.parse(LEVEL5_FILE, col_meta_only=True)

valid_sig_ids = set(meta_only.col_metadata_df.index.astype(str))
print(f"✔ Valid sig_ids in GCTX: {len(valid_sig_ids):,}")

# ============================== STEP 2: Load metadata ============================== #

print("\n=== STEP 2: Loading metadata (sig_info + pert_info + DRH) ===")
sig = pd.read_csv(SIG_INFO_FILE, sep="\t", low_memory=False)
pert = pd.read_csv(PERT_INFO_FILE, sep="\t", low_memory=False)
drh = pd.read_csv(DRH_FILE, sep="\t", comment="!", low_memory=False)

# Filter cell line
sig = sig[sig["cell_id"] == TARGET_CELL_LINE]
sig["sig_id"] = sig["sig_id"].astype(str)
sig = sig[sig["sig_id"].isin(valid_sig_ids)]
print(f"✔ {TARGET_CELL_LINE} signatures present in GCTX: {len(sig):,}")

# Keep treatment compounds only
pert = pert[pert["pert_type"] == "trt_cp"]

# Merge pert metadata
meta = sig.merge(
    pert[["pert_id", "pert_iname", "pert_type"]],
    on="pert_id",
    how="left"
)

# Merge MoA from DRH using pert_iname
meta = meta.merge(
    drh[["pert_iname", "moa"]],
    on="pert_iname",
    how="left"
)

# Keep MoA labeled only
meta = meta.dropna(subset=["moa"])
print(f"✔ MoA-labeled entries: {len(meta):,}")

# ============================== STEP 3: Select major MoA classes ============================== #

print("\n=== STEP 3: Selecting top MoA classes ===")

counts = meta["moa"].value_counts()
top_moas = counts.head(TOP_N_MOA).index.tolist()

meta = meta[meta["moa"].isin(top_moas)]
print(f"✔ After selecting top {TOP_N_MOA} MoAs: {len(meta):,}")

# Balanced sampling per class
samples_per_class = max(1, MAX_SAMPLES // len(top_moas))

rows = []
for moa in top_moas:
    subset = meta[meta["moa"] == moa]
    take = min(samples_per_class, len(subset))
    rows.append(subset.sample(n=take, random_state=42))

meta = pd.concat(rows)
meta = meta.drop_duplicates(subset=["sig_id"])
print(f"✔ Total sampled: {len(meta):,}")

needed_sig_ids = meta["sig_id"].astype(str).tolist()

# ============================== STEP 4: Load Expression Matrix ============================== #

print("\n=== STEP 4: Loading expression for selected sig_ids ===")

gct = parse.parse(LEVEL5_FILE, cid=needed_sig_ids)

expr = gct.data_df.T.copy()
expr.index = expr.index.astype(str)
expr.reset_index(inplace=True)

# fix naming
first_col = expr.columns[0]
if first_col != "sig_id":
    expr.rename(columns={first_col: "sig_id"}, inplace=True)

expr["sig_id"] = expr["sig_id"].astype(str)
print("✔ Expression matrix:", expr.shape)

# ============================== STEP 5: Merge Expression + Metadata ============================== #

print("\n=== STEP 5: Merging expression + metadata ===")

full = expr.merge(meta, on="sig_id", how="inner")
print("✔ Final merged dataset:", full.shape)

X = full.select_dtypes(include=[np.number])
y = full["moa"].astype(str)

# ============================== STEP 6: Scale + PCA ============================== #

print("\n=== STEP 6: Scaling + PCA ===")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA components must <= min(n_samples, n_features)
n_samples, n_features = X_scaled.shape
n_comp = min(PCA_COMPONENTS, n_samples - 1, n_features)

from sklearn.decomposition import PCA
pca = PCA(n_components=n_comp, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✔ PCA components used: {n_comp}")
print("✔ X_pca shape:", X_pca.shape)

# ============================== STEP 7: Train/Test Split ============================== #

print("\n=== STEP 7: Train/Test Split ===")

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"✔ Train: {len(X_train)} | Test: {len(X_test)}")

# ============================== STEP 8: Train SVM with class_weight ============================== #

print("\n=== STEP 8: Training Literature-Accurate SVM ===")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
weights = {i: w for i, w in enumerate(class_weights)}

svm = SVC(
    kernel="rbf",
    C=3,
    gamma="scale",
    class_weight=weights,
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)
print("✔ SVM trained successfully.")

# ============================== STEP 9: Evaluation ============================== #

print("\n=== STEP 9: Evaluation ===")

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================== SAVE ============================== #

dump(svm, f"{RESULTS_DIR}/svm_model.joblib")
dump(scaler, f"{RESULTS_DIR}/scaler.joblib")
dump(pca, f"{RESULTS_DIR}/pca.joblib")

print("\n✔ DONE! Literature-accurate SVM saved in:", RESULTS_DIR)
