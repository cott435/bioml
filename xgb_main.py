import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'/data_name
emb_path = base_data_dir / f'{model_name}_embeddings.h5'
df_path = base_data_dir / 'finalized_50_df.parquet'

PCA_COMPONENTS = 256  # Set to None to skip PCA
N_SPLITS = 5  # or 10 if you have many clusters
RANDOM_STATE = 42

# XGBoost hyperparameters (good starting point for regression)
xgb_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",  # use "gpu_hist" if you have GPU
    "n_jobs": -1,
    "verbosity": 0,
}
# ============================================================

# ------------------- Load data -------------------
print("Loading embeddings...")
with h5py.File(emb_path, "r") as f:
    print("Available H5 keys:", list(f.keys()))
    # Adjust key if yours is not 'embeddings' (common: 'X', 'emb', 'embeddings')
    X = np.asarray(f["embeddings"])  # shape (N_total_residues, 960)

print("Loading dataframe...")
# Use pd.read_pickle or pd.read_csv depending on your file
df = pd.read_pickle(df_path)  # <-- change to pd.read_csv(df_path) if CSV
print(f"DataFrame shape: {df.shape}")

assert len(df) == X.shape[
    0], f"Mismatch! DF has {len(df)} rows, embeddings have {X.shape[0]} samples. Make sure they are aligned (same order)."

y = df["y"].values.astype(np.float32)  # <-- change column name if needed (e.g. "binding_size")
groups = df["cluster_group"].values.astype(int)  # <-- change column name if needed ("clustering_group_number")

print(f"Total residues: {len(y):,}")
print(f"Unique clusters: {len(np.unique(groups))}")

# ------------------- Optional PCA -------------------
if PCA_COMPONENTS is not None and PCA_COMPONENTS < X.shape[1]:
    print(f"Applying PCA to {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X = pca.fit_transform(X)
    cum_var = np.sum(pca.explained_variance_ratio_)
    print(f"Reduced to {X.shape[1]} dims | Explained variance: {cum_var:.4f}")
else:
    print("Using full 960 dimensions (no PCA)")

# ------------------- Grouped K-Fold CV -------------------
gkf = GroupKFold(n_splits=N_SPLITS)

mse_scores = []
r2_scores = []
print(f"\nStarting {N_SPLITS}-fold Grouped CV...\n")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"Fold {fold + 1:2d} | MSE = {mse:8.4f} | R² = {r2:6.4f} | samples = {len(val_idx):,}")

print("\n" + "=" * 60)
print(f"CV Results (mean ± std)")
print(f"MSE : {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print("=" * 60)

# Optional: final model on all data
print("\nTraining final model on full dataset...")
final_model = xgb.XGBRegressor(**xgb_params)
final_model.fit(X, y, verbose=False)
print("Final model trained. You can save it with final_model.save_model('xgboost_protein_regressor.json')")