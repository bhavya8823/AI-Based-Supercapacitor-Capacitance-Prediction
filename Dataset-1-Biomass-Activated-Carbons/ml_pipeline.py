import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ──────────────────────────────────────────────────────────────
# PHASE 1: DATA CLEANING
# ──────────────────────────────────────────────────────────────

df = pd.read_csv('/mnt/user-data/uploads/table_S1_final.csv')

# --- 1a. Forward-fill Sample column, then rename duplicates ---
df['Sample'] = df['Sample'].ffill()

# Rename duplicate samples: corn husks_1, corn husks_2, ...
counts = {}
new_names = []
for name in df['Sample']:
    if name not in counts:
        counts[name] = 0
    counts[name] += 1

# Count total occurrences per name
total = df['Sample'].value_counts().to_dict()
seen   = {}
for name in df['Sample']:
    if total[name] > 1:
        seen[name] = seen.get(name, 0) + 1
        new_names.append(f"{name}_{seen[name]}")
    else:
        new_names.append(name)

df['Sample'] = new_names

# --- 1b. Convert Activation time to minutes ---
def to_minutes(val):
    val = str(val).strip()
    if val.endswith('h'):
        return int(float(val[:-1]) * 60)
    else:
        return float(val)

df['Activation time'] = df['Activation time'].apply(to_minutes)

# --- 1c. Drop References column ---
df.drop(columns=['References'], inplace=True)

# --- 1d. Encode categorical columns ---
cat_cols = ['agent', 'sample/agent']
le = {}
for col in cat_cols:
    le[col] = LabelEncoder()
    df[col + '_enc'] = le[col].fit_transform(df[col].astype(str))

# Save cleaned CSV
df.to_csv('/home/claude/cleaned_data.csv', index=False)
print("Cleaned data shape:", df.shape)
print(df[['Sample', 'Activation time', 'agent']].head(15))

# ──────────────────────────────────────────────────────────────
# PHASE 2: FEATURE MATRIX
# ──────────────────────────────────────────────────────────────

feature_cols = ['Activation temperature', 'Activation time',
                'SBET', 'C', 'O', 'N', 'atom', 'conditions',
                'agent_enc', 'sample/agent_enc']
target_col   = 'capacitance'

df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols].values
y = df_model[target_col].values

feature_names = ['Act. Temp (°C)', 'Act. Time (min)', 'SBET (m²/g)',
                 'C (%)', 'O (%)', 'N (%)', 'Other atoms (%)',
                 'Current density', 'Agent (enc)', 'Sample/Agent ratio (enc)']

print(f"\nModel dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ──────────────────────────────────────────────────────────────
# PHASE 3 & 4: MODEL TRAINING + EVALUATION
# ──────────────────────────────────────────────────────────────

models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':         RandomForestRegressor(n_estimators=200, max_depth=8,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting\n(XGBoost equivalent)':
                             GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                       learning_rate=0.05,
                                                       subsample=0.8, random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    r2   = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae  = mean_absolute_error(y, y_pred_cv)
    results[name] = {'model': model, 'y_pred': y_pred_cv, 'R2': r2,
                     'RMSE': rmse, 'MAE': mae}
    model.fit(X, y)  # refit on all data for importance
    print(f"{name.replace(chr(10),' '):<40}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# ──────────────────────────────────────────────────────────────
# PHASE 5: PLOTS
# ──────────────────────────────────────────────────────────────

COLORS = {'Linear Regression': '#4e8cff',
          'Random Forest': '#2ec4b6',
          'Gradient Boosting\n(XGBoost equivalent)': '#e86452'}

# ── Figure 1: Predicted vs Experimental ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f1117')

for ax, (name, res) in zip(axes, results.items()):
    c = COLORS[name]
    ax.set_facecolor('#1a1d27')
    ax.scatter(y, res['y_pred'], color=c, alpha=0.75, edgecolors='white',
               linewidths=0.4, s=60, zorder=3)
    mn, mx = min(y.min(), res['y_pred'].min()) - 10, max(y.max(), res['y_pred'].max()) + 10
    ax.plot([mn, mx], [mn, mx], 'w--', lw=1.2, label='Ideal', zorder=2)
    ax.set_xlabel('Experimental Capacitance (F/g)', color='white', fontsize=11)
    ax.set_ylabel('Predicted Capacitance (F/g)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#444')
    label = f"R² = {res['R2']:.4f}\nRMSE = {res['RMSE']:.2f}\nMAE  = {res['MAE']:.2f}"
    ax.text(0.04, 0.97, label, transform=ax.transAxes, color='white',
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(facecolor='#2a2d3a', alpha=0.8, boxstyle='round,pad=0.4'))
    ax.grid(color='#2a2d3a', lw=0.6, zorder=1)

plt.suptitle('Predicted vs Experimental Capacitance — 5-Fold CV',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/fig1_pred_vs_exp.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig1")

# ── Figure 2: Metrics Comparison ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('#0f1117')

metrics = ['R2', 'RMSE', 'MAE']
titles  = ['R² (higher = better)', 'RMSE (lower = better)', 'MAE (lower = better)']
model_labels = [n.replace('\n', '\n') for n in results.keys()]
colors = list(COLORS.values())

for ax, metric, title in zip(axes, metrics, titles):
    vals = [res[metric] for res in results.values()]
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.55)
    ax.set_facecolor('#1a1d27')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(model_labels, color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(axis='y', color='#2a2d3a', lw=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=9)

plt.suptitle('Model Performance Metrics (5-Fold Cross-Validation)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig2_metrics.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig2")

# ── Figure 3: Feature Importance (RF + GB) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0f1117')

for ax, (model_key, color) in zip(axes,
        [('Random Forest', '#2ec4b6'),
         ('Gradient Boosting\n(XGBoost equivalent)', '#e86452')]):
    model = results[model_key]['model']
    imp = model.feature_importances_
    idx = np.argsort(imp)
    ax.set_facecolor('#1a1d27')
    bars = ax.barh([feature_names[i] for i in idx], imp[idx],
                   color=color, edgecolor='white', linewidth=0.3, alpha=0.9)
    ax.set_xlabel('Feature Importance', color='white', fontsize=11)
    ax.set_title(model_key.replace('\n', ' '), color=color,
                 fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(axis='x', color='#2a2d3a', lw=0.6)

plt.suptitle('Feature Importance Analysis', color='white',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig3_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved fig3")

# ── Figure 4: SHAP-style permutation importance ──────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0f1117')

for ax, (model_key, color) in zip(axes,
        [('Random Forest', '#2ec4b6'),
         ('Gradient Boosting\n(XGBoost equivalent)', '#e86452')]):
    model = results[model_key]['model']
    perm = permutation_importance(model, X, y, n_repeats=30,
                                  random_state=42, scoring='r2')
    mean_imp = perm.importances_mean
    std_imp  = perm.importances_std
    idx = np.argsort(mean_imp)
    ax.set_facecolor('#1a1d27')
    ax.barh([feature_names[i] for i in idx], mean_imp[idx],
            xerr=std_imp[idx], color=color, edgecolor='white',
            linewidth=0.3, alpha=0.9, capsize=3,
            error_kw={'ecolor': 'white', 'alpha': 0.5})
    ax.set_xlabel('Permutation Importance (ΔR²)', color='white', fontsize=11)
    ax.set_title(model_key.replace('\n', ' '), color=color,
                 fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(axis='x', color='#2a2d3a', lw=0.6)

plt.suptitle('Permutation-Based Feature Importance (SHAP-Equivalent)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig4_permutation_importance.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved fig4")

# ── Figure 5: Residual Analysis ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0f1117')

for ax, (name, res) in zip(axes, results.items()):
    c = COLORS[name]
    residuals = y - res['y_pred']
    ax.set_facecolor('#1a1d27')
    ax.scatter(res['y_pred'], residuals, color=c, alpha=0.7,
               edgecolors='white', linewidths=0.3, s=50)
    ax.axhline(0, color='white', ls='--', lw=1.2)
    ax.set_xlabel('Predicted Capacitance (F/g)', color='white', fontsize=11)
    ax.set_ylabel('Residual (F/g)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(color='#2a2d3a', lw=0.6)

plt.suptitle('Residual Analysis', color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig5_residuals.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig5")

print("\n✅ All figures and cleaned CSV saved successfully.")
print("\n=== FINAL METRICS SUMMARY ===")
for name, res in results.items():
    print(f"{name.replace(chr(10),' '):<45} R²={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  MAE={res['MAE']:.2f}")
