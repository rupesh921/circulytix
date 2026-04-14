"""
train_and_bundle_two_stage.py

Trains:
 - Stage A: Multi-output model for Energy (log1p) and Water (linear)
 - Stage B: CO2 model using original features + predicted Energy (from Stage A)
Saves a bundle with both models and metadata for inference.

Usage: edit DATA_PATH to point to your CSV.
"""

import os
import time
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform
# -------------------------
# Accuracy helpers (regression)
# -------------------------
def regression_accuracy(y_true, y_pred):
    """
    Returns:
      mae_acc: MAE-based accuracy (%) = max(0, (1 - MAE / mean(y_true)) * 100)
      r2_acc : R2-based accuracy (%) = max(0, R2 * 100)
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mean_true = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    mae_acc = 0.0
    if mean_true > 0:
        mae_acc = max(0.0, (1.0 - mae / mean_true) * 100.0)
    r2_acc = max(0.0, r2_score(y_true, y_pred) * 100.0) if len(y_true) > 0 else 0.0
    return mae_acc, r2_acc

def overall_model_accuracy(acc_dict):
    """
    acc_dict example:
      {'energy': {'mae_acc': x, 'r2': y}, 'water': {...}, 'co2': {...}}
    Returns dict with overall MAE-based avg, R2-based avg and combined mean.
    """
    mae_accs = [v['mae_acc'] for v in acc_dict.values() if 'mae_acc' in v]
    r2_accs  = [v['r2'] for v in acc_dict.values() if 'r2' in v]
    overall_mae_acc = float(np.mean(mae_accs)) if mae_accs else 0.0
    overall_r2_acc  = float(np.mean(r2_accs)) if r2_accs else 0.0
    combined = (overall_mae_acc + overall_r2_acc) / 2.0
    return {"MAE_based_overall": overall_mae_acc, "R2_based_overall": overall_r2_acc, "Combined_overall_accuracy": combined}


# -------------------------
# CONFIG
# -------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set DATA_PATH to your CSV file; if not found a ValueError will be raised.
DATA_PATH = "/content/Dataset.csv" #may need to edit it
OUT_BUNDLE = "lci_model_bundle_two_stage.joblib"
TRAINED_ON_FU = 1000.0  # baseline FU in dataset (kg)

# -------------------------
# Helpers
# -------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# -------------------------
# 1) LOAD DATA
# -------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please set DATA_PATH to your CSV.")

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH, encoding='latin1')

# expected canonical columns in df:
# 'primary_content', 'secondary_content', 'end_of_life_recycling',
# 'transport_distance_km', 'production_efficiency',
# 'energy_consumption', 'water_consumption', 'co2_emissions', 'functional_unit_kg' (optional)

required_cols = ['primary_content','secondary_content','end_of_life_recycling',
                 'transport_distance_km','production_efficiency',
                 'energy_consumption','water_consumption','co2_emissions']

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

# -------------------------
# 2) FEATURE ENGINEERING
# -------------------------
df = df.copy()
df['recycled_ratio'] = df['secondary_content'] / 100.0
df['pri_eff'] = df['primary_content'] * df['production_efficiency'] / 100.0
df['log_dist'] = np.log1p(df['transport_distance_km'])

features_stageA = ['primary_content','secondary_content','end_of_life_recycling',
                   'transport_distance_km','production_efficiency','recycled_ratio','pri_eff','log_dist']
# Stage B uses original features + predicted energy (we'll append 'pred_energy' to features)
features_stageB = features_stageA + ['pred_energy']

# Targets
# Stage A targets: energy (log1p), water (linear)
# Stage B target: co2 (log1p)
# Remove rows with NaN in features/targets
df = df.dropna(subset=features_stageA + ['energy_consumption','water_consumption','co2_emissions']).reset_index(drop=True)

# Optionally sample down (uncomment to speed up on limited RAM)
# df = df.sample(n=50000, random_state=RANDOM_SEED).reset_index(drop=True)

# Train/test split (for final evaluation)
X_all = df[features_stageA].values
y_energy = df['energy_consumption'].values
y_water = df['water_consumption'].values
y_co2 = df['co2_emissions'].values

X_train_full, X_test, y_train_full_energy, y_test_energy, y_train_full_water, y_test_water, y_train_full_co2, y_test_co2 = train_test_split(
    X_all, y_energy, y_water, y_co2, test_size=0.18, random_state=RANDOM_SEED
)

# -------------------------
# 3) TRAIN STAGE A (Energy & Water)
# We'll train multi-output on [log1p(energy), water]
# -------------------------
print("\n--- Stage A: Train Energy & Water model ---")
# Prepare transformed targets for Stage A
y_train_A = np.column_stack([np.log1p(y_train_full_energy), y_train_full_water])
y_test_A = None  # we'll evaluate after training by predicting and inverting

# Light hyperparam search for HistGradientBoosting (single-target) to get decent params, but we keep small n_iter
param_dist_A = {
    "learning_rate": uniform(0.01, 0.1),
    "max_iter": randint(80, 240),
    "max_leaf_nodes": randint(10, 80),
    "min_samples_leaf": randint(3, 40),
    "l2_regularization": uniform(0.0, 1.0)
}
baseA = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
# We'll search for good params for energy only (subsample)
subsample_size = min(10000, X_train_full.shape[0])
idxs = np.random.choice(range(X_train_full.shape[0]), size=subsample_size, replace=False)
X_sub = X_train_full[idxs]
y_sub_energy = y_train_full_energy[idxs]

rndA = RandomizedSearchCV(baseA, param_distributions=param_dist_A, n_iter=8, cv=3,
                          scoring='neg_mean_absolute_error', random_state=RANDOM_SEED, n_jobs=1, verbose=0)
print("Stage A hyperparam search (energy)...")
rndA.fit(X_sub, y_sub_energy)
best_params_A = rndA.best_params_
print("Stage A best params:", best_params_A)

# Build final StageA estimator with best params
hgbA = HistGradientBoostingRegressor(random_state=RANDOM_SEED, **best_params_A)
pipelineA = make_pipeline(StandardScaler(), hgbA)
multiA = MultiOutputRegressor(pipelineA, n_jobs=1)

print("Training Stage A on full training set...")
t0 = time.time()
multiA.fit(X_train_full, y_train_A)
t1 = time.time()
print(f"Stage A trained in {(t1-t0):.1f}s")

# Evaluate Stage A
y_pred_A_t = multiA.predict(X_test)  # transformed predictions: [log1p(energy_pred), water_pred]
pred_energy_per1000 = np.expm1(y_pred_A_t[:,0])  # invert log1p
pred_water_per1000 = y_pred_A_t[:,1]

r2_energy = r2_score(y_test_energy, pred_energy_per1000)
mae_energy = mean_absolute_error(y_test_energy, pred_energy_per1000)
rmse_energy = rmse(y_test_energy, pred_energy_per1000)

r2_water = r2_score(y_test_water, pred_water_per1000)
mae_water = mean_absolute_error(y_test_water, pred_water_per1000)
rmse_water = rmse(y_test_water, pred_water_per1000)

print("\nStage A evaluation (test):")
print(f"Energy - R2: {r2_energy:.4f}, MAE: {mae_energy:.3f}, RMSE: {rmse_energy:.3f}")
print(f"Water  - R2: {r2_water:.4f}, MAE: {mae_water:.3f}, RMSE: {rmse_water:.3f}")

# -------------------------
# 4) Create predicted energy on full training set (simulate inference)
# We'll predict on X_train_full and use those predicted values as feature for Stage B
# -------------------------
print("\nGenerating predicted energy on training set (for Stage B training)...")
pred_train_A_t = multiA.predict(X_train_full)
pred_train_energy = np.expm1(pred_train_A_t[:,0])  # predicted energy per-1000kg (training set)
# Build X for Stage B training: original features + pred_train_energy
X_train_B = np.hstack([X_train_full, pred_train_energy.reshape(-1,1)])

# Build X_test_B for evaluation (features + predicted energy on X_test)
pred_test_A_t = y_pred_A_t  # already computed above for X_test
pred_test_energy = np.expm1(pred_test_A_t[:,0])
X_test_B = np.hstack([X_test, pred_test_energy.reshape(-1,1)])

# -------------------------
# 5) TRAIN STAGE B (CO2) using pred_energy as an input
# We'll train log1p(CO2)
# -------------------------
print("\n--- Stage B: Train CO2 model using predicted energy feature ---")
y_train_B = np.log1p(y_train_full_co2)  # transformed target
y_test_B_true = y_test_co2  # for metrics we will compare inverted preds

# Light hyperparam search for CO2 as well (smaller search)
param_dist_B = {
    "learning_rate": uniform(0.01, 0.1),
    "max_iter": randint(80, 240),
    "max_leaf_nodes": randint(10, 80),
    "min_samples_leaf": randint(3, 40),
    "l2_regularization": uniform(0.0, 1.0)
}

baseB = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
rndB = RandomizedSearchCV(baseB, param_distributions=param_dist_B, n_iter=8, cv=3,
                          scoring='neg_mean_absolute_error', random_state=RANDOM_SEED, n_jobs=1, verbose=0)
print("Stage B hyperparam search (CO2) on subsample...")
subsample_size_B = min(8000, X_train_B.shape[0])
idxsB = np.random.choice(range(X_train_B.shape[0]), size=subsample_size_B, replace=False)
rndB.fit(X_train_B[idxsB], y_train_B[idxsB])
best_params_B = rndB.best_params_
print("Stage B best params:", best_params_B)

# Final CO2 model
hgbB = HistGradientBoostingRegressor(random_state=RANDOM_SEED, **best_params_B)
pipeB = make_pipeline(StandardScaler(), hgbB)
print("Training Stage B on full Stage B training set...")
t4 = time.time()
pipeB.fit(X_train_B, y_train_B)
t5 = time.time()
print(f"Stage B trained in {(t5-t4):.1f}s")

# Evaluate Stage B
y_pred_B_t = pipeB.predict(X_test_B)  # predictions in log1p space
y_pred_B = np.expm1(y_pred_B_t)       # invert log1p

r2_co2 = r2_score(y_test_B_true, y_pred_B)
mae_co2 = mean_absolute_error(y_test_B_true, y_pred_B)
rmse_co2 = rmse(y_test_B_true, y_pred_B)

print("\nStage B evaluation (test):")
print(f"CO2 - R2: {r2_co2:.4f}, MAE: {mae_co2:.3f}, RMSE: {rmse_co2:.3f}")

# -------------------------
# 6) Bundle models and metadata into a single joblib
# -------------------------
print("\nSaving model bundle to:", OUT_BUNDLE)
bundle = {
    "stageA_model": multiA,           # predicts [log1p(energy), water]
    "stageA_features": features_stageA,
    "stageB_model": pipeB,            # predicts log1p(co2), expects stageB features including 'pred_energy'
    "stageB_features": features_stageB,
    "targets": {"stageA": ["energy (log1p)", "water"], "stageB": ["co2 (log1p)"]},
    "trained_on_fu_kg": TRAINED_ON_FU,
    "metrics": {
        "stageA": {
            "energy": {"r2": r2_energy, "mae": mae_energy, "rmse": rmse_energy},
            "water": {"r2": r2_water, "mae": mae_water, "rmse": rmse_water}
        },
        "stageB": {"co2": {"r2": r2_co2, "mae": mae_co2, "rmse": rmse_co2}}
    },
    "stageA_best_params": best_params_A,
    "stageB_best_params": best_params_B
}
joblib.dump(bundle, OUT_BUNDLE, compress=3)
print("Bundle saved.")

# -------------------------
# 7) Provide a predict() helper showing how to use the bundle
# -------------------------
def predict_from_bundle(bundle, input_dict, functional_unit_kg=None):
    """
    input_dict: must contain the keys in features_stageA (primary_content, secondary_content, end_of_life_recycling,
                transport_distance_km, production_efficiency). The engineered features pri_eff, log_dist, recycled_ratio
                will be computed if missing.
    functional_unit_kg: if provided, predictions will be scaled from trained_on_fu_kg to this FU.
    Returns: dict with 'per_1000kg' and 'scaled_to_fu' predictions.
    """
    # prepare inputs
    featsA = bundle['stageA_features']
    x = {}
    for f in featsA:
        if f in input_dict:
            x[f] = float(input_dict[f])
    # compute missing engineered features if needed
    if 'recycled_ratio' in featsA and 'recycled_ratio' not in x:
        x['recycled_ratio'] = x.get('secondary_content', 100.0 - x.get('primary_content', 0.0)) / 100.0
    if 'pri_eff' in featsA and 'pri_eff' not in x:
        x['pri_eff'] = x.get('primary_content', 0.0) * x.get('production_efficiency', 0.0) / 100.0
    if 'log_dist' in featsA and 'log_dist' not in x:
        x['log_dist'] = np.log1p(x.get('transport_distance_km', 0.0))

    X_row = np.array([[x[f] for f in featsA]])

    # Stage A predict
    stageA_model = bundle['stageA_model']
    yA_t = stageA_model.predict(X_row)[0]  # [log1p(energy), water]
    pred_energy_per1000 = float(np.expm1(yA_t[0]))
    pred_water_per1000 = float(yA_t[1])

    # Stage B predict (CO2) uses features_stageB = featsA + ['pred_energy']
    featsB = bundle['stageB_features']
    rowB = []
    for f in featsB:
        if f == 'pred_energy':
            rowB.append(pred_energy_per1000)
        else:
            rowB.append(x[f])
    rowB = np.array([rowB])
    stageB_model = bundle['stageB_model']
    yB_t = stageB_model.predict(rowB)[0]
    pred_co2_per1000 = float(np.expm1(yB_t))

    # scaling
    trained_fu = float(bundle.get('trained_on_fu_kg', TRAINED_ON_FU))
    fu = trained_fu if functional_unit_kg is None else float(functional_unit_kg)
    scale = fu / trained_fu
    result = {
        "per_1000kg": {
            "energy_kwh": pred_energy_per1000,
            "water_l": pred_water_per1000,
            "co2_kg": pred_co2_per1000
        },
        "scaled_to_fu": {
            "functional_unit_kg": fu,
            "energy_kwh": pred_energy_per1000 * scale,
            "water_l": pred_water_per1000 * scale,
            "co2_kg": pred_co2_per1000 * scale
        }
    }
    return result

# Example usage:
if __name__ == "__main__":
    print("\nExample prediction using the saved bundle:")
    b = joblib.load(OUT_BUNDLE)
    sample_input = {
        "primary_content": 30.0,
        "secondary_content": 70.0,
        "end_of_life_recycling": 70.0,
        "transport_distance_km": 2000.0,
        "production_efficiency": 55.0
    }
    pred = predict_from_bundle(b, sample_input, functional_unit_kg=50.0)
    print(pred)
    print("\nStage B evaluation (test):")
print(f"CO2 - R2: {r2_co2:.4f}, MAE: {mae_co2:.3f}, RMSE: {rmse_co2:.3f}")
# -------------------------
# Compute per-target accuracies and combined accuracy
# -------------------------
acc_energy_mae, acc_energy_r2 = regression_accuracy(y_test_energy, pred_energy_per1000)
acc_water_mae, acc_water_r2 = regression_accuracy(y_test_water, pred_water_per1000)
acc_co2_mae, acc_co2_r2 = regression_accuracy(y_test_B_true, y_pred_B)

print(f"Energy Accuracy (MAE-based): {acc_energy_mae:.2f}%  | R²-based: {acc_energy_r2:.2f}%")
print(f"Water  Accuracy (MAE-based): {acc_water_mae:.2f}%  | R²-based: {acc_water_r2:.2f}%")
print(f"CO2   Accuracy (MAE-based): {acc_co2_mae:.2f}%  | R²-based: {acc_co2_r2:.2f}%")

acc_summary = {
    "energy": {"mae_acc": acc_energy_mae, "r2": acc_energy_r2},
    "water":  {"mae_acc": acc_water_mae, "r2": acc_water_r2},
    "co2":    {"mae_acc": acc_co2_mae, "r2": acc_co2_r2}
}
combined_acc = overall_model_accuracy(acc_summary)

print("\n=== COMBINED MODEL ACCURACY ===")
print(f"MAE-based Overall Accuracy: {combined_acc['MAE_based_overall']:.2f}%")
print(f"R²-based Overall Accuracy: {combined_acc['R2_based_overall']:.2f}%")
print(f"Final Combined Overall Accuracy: {combined_acc['Combined_overall_accuracy']:.2f}%\n")
