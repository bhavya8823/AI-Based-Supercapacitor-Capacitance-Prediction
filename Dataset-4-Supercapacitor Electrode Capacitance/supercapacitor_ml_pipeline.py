"""
=============================================================================
Supercapacitor Capacitance Prediction — Full ML Pipeline
=============================================================================
Phases covered:
  • Data Cleaning & Missing-Value Handling
  • Phase 3 – Model Development  (Linear Regression, Random Forest, XGBoost)
  • Phase 4 – Model Evaluation   (CV, leakage-safe split, R², RMSE, MAE, plots)
  • Phase 5 – Interpretability   (Feature importance, SHAP)
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection       import permutation_importance
from sklearn.impute           import SimpleImputer

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] xgboost not installed — XGBoost phase will be skipped.")
    print("         Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] shap not installed — SHAP phase will use permutation importance.")
    print("         Install with: pip install shap")


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("Supplementary_Dataset.xlsx")   # ← update if needed
OUTPUT_DIR  = Path("ml_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# ░░  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(name: str):
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

def metrics_dict(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"Model": label, "R²": round(r2, 4),
            "RMSE": round(rmse, 4), "MAE": round(mae, 4)}

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PHASE 1 – DATA LOADING & INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 1 — DATA LOADING & INSPECTION")
print("="*70)

df_raw = pd.read_excel(DATA_PATH, sheet_name="Supercapacitor_database")
# Strip trailing spaces from column names
df_raw.columns = df_raw.columns.str.strip()
print(f"Raw shape : {df_raw.shape}")
print("\nColumn dtypes:\n", df_raw.dtypes)

# Drop fully-empty / metadata columns
KEEP_COLS = [
    "PW (V)", "SSA (m2/g)", "PV (cm3/g)", "PS (nm)", "Id/Ig",
    "C%", "N%", "O %", "S%",
    "current density (A/g)",
    "Electrode", "Electrolyte", "Method",
    "Capacitance (F/g)"           # ← target
]
df = df_raw[KEEP_COLS].copy()
print(f"\nWorking shape after column selection : {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PHASE 2 – DATA CLEANING & MISSING-VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 2 — DATA CLEANING & MISSING-VALUE HANDLING")
print("="*70)

# 2a. Remove rows where target is missing or physically invalid
target = "Capacitance (F/g)"
before = len(df)
df = df[df[target].notna() & (df[target] > 0)]
print(f"Rows after removing null/zero capacitance : {len(df)}  (removed {before-len(df)})")

# 2b. Normalise free-text categorical columns
def normalise_category(series: pd.Series) -> pd.Series:
    """Lower-case, strip whitespace, collapse common synonyms."""
    s = series.astype(str).str.lower().str.strip()
    # Method synonyms
    s = s.str.replace(r"three[\s\-]?electrode[\s\-]?(system|method)", "3-electrode", regex=True)
    s = s.str.replace(r"two[\s\-]?electrode[\s\-]?(system|method)",   "2-electrode", regex=True)
    return s

df["Method"]      = normalise_category(df["Method"])
df["Electrode"]   = df["Electrode"].astype(str).str.strip()
df["Electrolyte"] = df["Electrolyte"].astype(str).str.strip()

# 2c. Missing-value summary
print("\nMissing values per column:")
print(df.isnull().sum().to_string())

# 2d. Numeric features — median imputation
NUMERIC_FEATS = ["PW (V)", "SSA (m2/g)", "PV (cm3/g)", "PS (nm)",
                 "Id/Ig", "C%", "N%", "O %", "S%", "current density (A/g)"]

num_imputer = SimpleImputer(strategy="median")
df[NUMERIC_FEATS] = num_imputer.fit_transform(df[NUMERIC_FEATS])

# 2e. Categorical features — mode imputation
CAT_FEATS = ["Electrode", "Electrolyte", "Method"]
for col in CAT_FEATS:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

print("\n✓ After imputation — missing values per column:")
print(df.isnull().sum().to_string())

# 2f. Outlier capping on target (IQR-based)
Q1, Q3 = df[target].quantile(0.01), df[target].quantile(0.99)
df = df[(df[target] >= Q1) & (df[target] <= Q3)]
print(f"\nRows after 1st–99th percentile outlier capping on target : {len(df)}")

# 2g. Encode categorical features
le = {}
for col in CAT_FEATS:
    le[col] = LabelEncoder()
    df[col + "_enc"] = le[col].fit_transform(df[col])

ENC_CAT_FEATS = [c + "_enc" for c in CAT_FEATS]
ALL_FEATURES  = NUMERIC_FEATS + ENC_CAT_FEATS

X = df[ALL_FEATURES].values
y = df[target].values

print(f"\nFinal dataset : {X.shape[0]} samples × {X.shape[1]} features")
print("Feature names :", ALL_FEATURES)

# 2h. Save cleaned data
df_clean = df[ALL_FEATURES + [target]]
df_clean.columns = [c.replace("_enc","(encoded)") for c in df_clean.columns]
df_clean.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
print(f"  [saved] {OUTPUT_DIR/'cleaned_dataset.csv'}")

# 2i. Missing-value heatmap (raw)
plt.figure(figsize=(12, 5))
raw_miss = df_raw[KEEP_COLS].isnull()
sns.heatmap(raw_miss, yticklabels=False, cbar=True, cmap="viridis")
plt.title("Missing Value Map — Raw Dataset", fontsize=13)
plt.tight_layout()
save_fig("00_missing_value_map")

# 2j. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=50, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Capacitance (F/g)"); axes[0].set_ylabel("Count")
axes[0].set_title("Target Distribution")
axes[1].hist(np.log1p(y), bins=50, color="#DD8452", edgecolor="white")
axes[1].set_xlabel("log(1 + Capacitance)"); axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Target Distribution")
plt.suptitle("Capacitance Distribution", fontsize=14, y=1.01)
plt.tight_layout()
save_fig("01_target_distribution")

# 2k. Correlation heatmap
plt.figure(figsize=(12, 9))
corr_df = pd.DataFrame(X, columns=ALL_FEATURES)
corr_df[target] = y
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
plt.title("Feature Correlation Heatmap", fontsize=13)
plt.tight_layout()
save_fig("02_correlation_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PHASE 3 – MODEL DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 3 — MODEL DEVELOPMENT")
print("="*70)

# Leakage-safe split: performed ONCE, before any model sees any data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"Train : {X_train.shape[0]} samples  |  Test : {X_test.shape[0]} samples")

# Scale for Linear Regression
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)   # fit ONLY on train
X_te_sc  = scaler.transform(X_test)

# --- 3.1  Linear Regression (baseline) ---
print("\n[3.1] Linear Regression …")
lr = LinearRegression()
lr.fit(X_tr_sc, y_train)
print("  ✓ trained")

# --- 3.2  Random Forest ---
print("[3.2] Random Forest Regression …")
rf = RandomForestRegressor(
    n_estimators=300, max_features="sqrt",
    min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED
)
rf.fit(X_train, y_train)
print("  ✓ trained")

# --- 3.3  XGBoost ---
if XGB_AVAILABLE:
    print("[3.3] XGBoost Regression …")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, eval_metric="rmse",
        random_state=RANDOM_SEED, verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("  ✓ trained")
else:
    xgb_model = None
    print("[3.3] XGBoost skipped (not installed).")

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PHASE 4 – MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 4 — MODEL EVALUATION")
print("="*70)

# 4a. Cross-validation (5-fold, on TRAIN set only → leakage-safe)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

def run_cv(model, X_cv, y_cv, label):
    cv_res = cross_validate(model, X_cv, y_cv, cv=kf, scoring=cv_scoring, n_jobs=-1)
    print(f"\n  {label} — 5-Fold CV (train set only):")
    print(f"    R²   = {cv_res['test_r2'].mean():.4f} ± {cv_res['test_r2'].std():.4f}")
    print(f"    RMSE = {-cv_res['test_neg_root_mean_squared_error'].mean():.4f}")
    print(f"    MAE  = {-cv_res['test_neg_mean_absolute_error'].mean():.4f}")
    return cv_res

cv_lr = run_cv(LinearRegression(), X_tr_sc, y_train, "Linear Regression")
cv_rf = run_cv(rf,                 X_train, y_train, "Random Forest")
if xgb_model:
    cv_xgb = run_cv(xgb_model, X_train, y_train, "XGBoost")

# 4b. Test-set metrics
results = []

y_pred_lr  = lr.predict(X_te_sc)
results.append(metrics_dict(y_test, y_pred_lr, "Linear Regression"))

y_pred_rf  = rf.predict(X_test)
results.append(metrics_dict(y_test, y_pred_rf, "Random Forest"))

if xgb_model:
    y_pred_xgb = xgb_model.predict(X_test)
    results.append(metrics_dict(y_test, y_pred_xgb, "XGBoost"))

metrics_df = pd.DataFrame(results)
print("\n📊 Hold-out Test Set Performance:")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
print(f"  [saved] {OUTPUT_DIR/'model_metrics.csv'}")

# 4c. Metrics bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
palette = ["#4C72B0", "#DD8452", "#55A868"]
for i, (col, title) in enumerate(zip(["R²","RMSE","MAE"],
                                     ["R² (higher is better)",
                                      "RMSE (lower is better)",
                                      "MAE (lower is better)"])):
    bars = axes[i].bar(metrics_df["Model"], metrics_df[col], color=palette[:len(metrics_df)])
    axes[i].set_title(title, fontsize=11)
    axes[i].set_ylabel(col)
    axes[i].tick_params(axis="x", rotation=20)
    for bar in bars:
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005*metrics_df[col].max(),
                     f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
plt.suptitle("Model Performance Comparison — Hold-out Test Set", fontsize=13)
plt.tight_layout()
save_fig("03_model_metrics_comparison")

# 4d. Predicted vs Experimental plots
def parity_plot(y_true, y_pred, label, color, ax):
    ax.scatter(y_true, y_pred, alpha=0.35, s=10, color=color, label="Samples")
    lims = [min(y_true.min(), y_pred.min())-5,
            max(y_true.max(), y_pred.max())+5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Ideal (y=x)")
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7))
    ax.set_xlabel("Experimental Capacitance (F/g)")
    ax.set_ylabel("Predicted Capacitance (F/g)")
    ax.set_title(label)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(markerscale=2, fontsize=8)

preds_info = [("Linear Regression", y_pred_lr, "#4C72B0"),
              ("Random Forest",      y_pred_rf, "#DD8452")]
if xgb_model:
    preds_info.append(("XGBoost", y_pred_xgb, "#55A868"))

ncols = len(preds_info)
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6))
if ncols == 1: axes = [axes]
for ax, (label, y_pred, color) in zip(axes, preds_info):
    parity_plot(y_test, y_pred, label, color, ax)
plt.suptitle("Predicted vs Experimental Capacitance (Test Set)", fontsize=14)
plt.tight_layout()
save_fig("04_parity_plots")

# 4e. Residual plots
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
if ncols == 1: axes = [axes]
for ax, (label, y_pred, color) in zip(axes, preds_info):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.35, s=10, color=color)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel("Predicted Capacitance (F/g)")
    ax.set_ylabel("Residuals (F/g)")
    ax.set_title(f"Residuals — {label}")
plt.suptitle("Residual Analysis (Test Set)", fontsize=14)
plt.tight_layout()
save_fig("05_residual_plots")

# 4f. CV score comparison (box plots)
cv_r2_data = {"Linear Regression": cv_lr["test_r2"],
              "Random Forest":      cv_rf["test_r2"]}
if xgb_model:
    cv_r2_data["XGBoost"] = cv_xgb["test_r2"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(cv_r2_data.values(), labels=cv_r2_data.keys(), patch_artist=True,
           boxprops=dict(facecolor="#AEC6E8"), medianprops=dict(color="red", lw=2))
ax.set_ylabel("R² (5-Fold CV)")
ax.set_title("Cross-Validation R² Distribution (Train Set)", fontsize=13)
plt.tight_layout()
save_fig("06_cv_r2_boxplot")

# ─────────────────────────────────────────────────────────────────────────────
# ░░  PHASE 5 – INTERPRETABILITY & PHYSICAL INSIGHT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PHASE 5 — INTERPRETABILITY & PHYSICAL INSIGHT")
print("="*70)

FEAT_LABELS = [
    "Potential Window (V)",
    "SSA (m²/g)",
    "Pore Volume (cm³/g)",
    "Pore Size (nm)",
    "Id/Ig Ratio",
    "Carbon %",
    "Nitrogen %",
    "Oxygen %",
    "Sulphur %",
    "Current Density (A/g)",
    "Electrode (encoded)",
    "Electrolyte (encoded)",
    "Method (encoded)"
]

# 5a. Random Forest built-in feature importance
rf_imp = rf.feature_importances_
imp_df = pd.DataFrame({"Feature": FEAT_LABELS, "RF Importance": rf_imp})
imp_df = imp_df.sort_values("RF Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="RF Importance", y="Feature", data=imp_df, palette="Blues_d")
plt.title("Random Forest — Feature Importance (MDI)", fontsize=13)
plt.xlabel("Mean Decrease in Impurity")
plt.tight_layout()
save_fig("07_rf_feature_importance")
print("\nRandom Forest — top features:")
print(imp_df.to_string(index=False))

# 5b. XGBoost feature importance
if xgb_model:
    xgb_imp = xgb_model.feature_importances_
    xgb_imp_df = pd.DataFrame({"Feature": FEAT_LABELS, "XGB Importance": xgb_imp})
    xgb_imp_df = xgb_imp_df.sort_values("XGB Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="XGB Importance", y="Feature", data=xgb_imp_df, palette="Oranges_d")
    plt.title("XGBoost — Feature Importance (Gain)", fontsize=13)
    plt.xlabel("Gain")
    plt.tight_layout()
    save_fig("08_xgb_feature_importance")
    print("\nXGBoost — top features:")
    print(xgb_imp_df.to_string(index=False))

# 5c. Linear Regression coefficients
coef_df = pd.DataFrame({"Feature": FEAT_LABELS,
                         "LR Coefficient": lr.coef_})
coef_df["abs"] = coef_df["LR Coefficient"].abs()
coef_df = coef_df.sort_values("abs", ascending=False).drop(columns="abs")

plt.figure(figsize=(10, 6))
colors = ["#4C72B0" if v > 0 else "#C44E52" for v in coef_df["LR Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["LR Coefficient"], color=colors)
plt.axvline(0, color="black", lw=0.8)
plt.title("Linear Regression — Standardised Coefficients", fontsize=13)
plt.xlabel("Coefficient")
plt.tight_layout()
save_fig("09_lr_coefficients")

# 5d. Permutation Importance (model-agnostic, evaluated on TEST set)
print("\n[5d] Computing Permutation Importance on test set …")
best_model  = rf           # use best tree model
best_X_test = X_test

perm_imp = permutation_importance(
    best_model, best_X_test, y_test,
    n_repeats=30, random_state=RANDOM_SEED, n_jobs=-1
)
perm_df = pd.DataFrame({
    "Feature":  FEAT_LABELS,
    "Mean Decrease R²": perm_imp.importances_mean,
    "Std": perm_imp.importances_std
}).sort_values("Mean Decrease R²", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(perm_df["Feature"], perm_df["Mean Decrease R²"],
         xerr=perm_df["Std"], color="#55A868", ecolor="grey", capsize=3)
plt.axvline(0, color="black", lw=0.8)
plt.title("Permutation Importance (Random Forest, Test Set)", fontsize=13)
plt.xlabel("Mean Decrease in R²")
plt.tight_layout()
save_fig("10_permutation_importance")
print("  ✓ done")
perm_df.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

# 5e. SHAP analysis
if SHAP_AVAILABLE:
    print("\n[5e] Computing SHAP values …")
    explainer   = shap.TreeExplainer(rf)
    shap_sample = X_test[:500]     # subsample for speed

    shap_values = explainer.shap_values(shap_sample)

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample,
                      feature_names=FEAT_LABELS, show=False)
    plt.title("SHAP Summary Plot (Random Forest)", fontsize=13)
    plt.tight_layout()
    save_fig("11_shap_summary")

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample,
                      feature_names=FEAT_LABELS,
                      plot_type="bar", show=False)
    plt.title("SHAP Mean |value| — Global Importance", fontsize=13)
    plt.tight_layout()
    save_fig("12_shap_bar")

    # Dependence plot for top feature
    top_feat_idx = np.argsort(np.abs(shap_values).mean(0))[-1]
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(top_feat_idx, shap_values, shap_sample,
                         feature_names=FEAT_LABELS, show=False)
    plt.title(f"SHAP Dependence — {FEAT_LABELS[top_feat_idx]}", fontsize=13)
    plt.tight_layout()
    save_fig("13_shap_dependence_top")

    print("  ✓ SHAP plots saved")
else:
    print("[5e] SHAP not available — permutation importance used instead (fig 10).")
    print("     Install shap (pip install shap) to enable full SHAP analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# ░░  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PIPELINE COMPLETE — SUMMARY")
print("="*70)
print(f"\n{'Model':<22} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("-"*54)
for _, row in metrics_df.iterrows():
    print(f"{row['Model']:<22} {row['R²']:>8.4f} {row['RMSE']:>10.4f} {row['MAE']:>10.4f}")

best_row = metrics_df.loc[metrics_df["R²"].idxmax()]
print(f"\n🏆 Best model : {best_row['Model']} (R² = {best_row['R²']:.4f})")

print(f"\nAll outputs saved in → ./{OUTPUT_DIR}/")
print("Files generated:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
