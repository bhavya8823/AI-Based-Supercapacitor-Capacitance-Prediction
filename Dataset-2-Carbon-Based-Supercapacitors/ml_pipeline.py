"""
Full ML Pipeline for Capacitance Prediction
============================================
Phases: Data Cleaning → Feature Engineering → Model Dev → Evaluation → Interpretability
"""

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
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ══════════════════════════════════════════════════════════════════
# PHASE 1 & 2: DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

df = pd.read_excel('/mnt/user-data/uploads/2.xlsx')
print(f"Raw shape: {df.shape}")

# ── Step 1: Drop DOI (identifier, not a feature) ─────────────────
df.drop(columns=['DOI'], inplace=True)

# ── Step 2: Handle Materials-2 (594/681 missing = 87%) ───────────
# Strategy: This column indicates a secondary/composite material.
# Missing → the electrode is a SINGLE-MATERIAL system.
# Encode: "none" for single-material, or the actual material name.
# Then binary flag: is_composite (1 = two materials, 0 = single)
df['Materials-2'] = df['Materials-2'].fillna('none')
df['Materials-2'] = df['Materials-2'].str.lower().str.strip()
df['is_composite'] = (df['Materials-2'] != 'none').astype(int)

# ── Step 3: Handle Materials-note (453/681 missing = 66%) ─────────
# Strategy: This is a free-text sub-label within a material group.
# It has too many unique values (~200+) to encode directly.
# Option A: Drop it (it encodes experimental sub-variants already
#           captured by other numeric columns).
# Option B: Binary flag: has_note vs no_note.
# We use both: drop the raw text but keep a binary flag.
df['has_material_note'] = df['Materials-note'].notna().astype(int)
df.drop(columns=['Materials-note'], inplace=True)

# ── Step 4: Normalise & encode remaining categoricals ─────────────
# Materials-1: fill 9 NaN with 'unknown'
df['Materials-1'] = df['Materials-1'].fillna('unknown').str.strip()

# Electrolyte: fix typos/variants, fill 13 NaN
electrolyte_map = {
    'teabf4 an': 'teabf4_an', 'teabf4/an': 'teabf4_an',
    'lipf6 ec/dmc': 'lipf6_ec_dmc',
    'pure emitfsi': 'emitfsi', 'pure emimbf4': 'emimbf4',
    'et4nbf4': 'et4NBF4', 'et4nbf4': 'et4NBF4',
    'bmimbf4': 'bmimbf4', 'bminbf6': 'bminbf6',
    'h2so4 pva': 'h2so4_pva',
    '10.1039/c3ta11051f': np.nan,   # clearly a DOI entered by mistake
    'sf': np.nan, 'cl': np.nan,     # undecodable
}
df['Electrolyte'] = (df['Electrolyte']
                     .str.lower().str.strip()
                     .replace(electrolyte_map))
df['Electrolyte'] = df['Electrolyte'].fillna('unknown')

# Label-encode categoricals
le_m1  = LabelEncoder()
le_m2  = LabelEncoder()
le_el  = LabelEncoder()

df['mat1_enc'] = le_m1.fit_transform(df['Materials-1'])
df['mat2_enc'] = le_m2.fit_transform(df['Materials-2'])
df['elec_enc'] = le_el.fit_transform(df['Electrolyte'])

# ── Step 5: Remove rows with target = 0 (non-physical / missing) ──
df_clean = df[df['Capacitance (F/g)'] > 0].copy()
print(f"After removing zero-capacitance rows: {df_clean.shape}")

# ── Step 6: Remove extreme outliers in target (>3 IQR) ────────────
Q1, Q3 = df_clean['Capacitance (F/g)'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df_clean[df_clean['Capacitance (F/g)'] <= Q3 + 3 * IQR]
print(f"After outlier removal: {df_clean.shape}")

# Save cleaned data
df_clean.to_csv('/home/claude/cleaned_data_2.csv', index=False)

# ── Step 7: Feature matrix ─────────────────────────────────────────
feature_cols = [
    'Voltage window (V)',
    'Specific surface area (m2/g)',
    'Pore volume (cm3/g)',
    'Pore size (nm)',
    'Micropore volume (cm3/g)',
    'SSA of micropores (m2/g)',
    'Id/Ig',
    'N (at. %)', 'O (at. %)', 'B (at. %)',
    'S (at. %)', 'F (at. %)', 'P (at. %)',
    'mat1_enc', 'mat2_enc', 'elec_enc',
    'is_composite', 'has_material_note',
]
target_col = 'Capacitance (F/g)'

feature_labels = [
    'Voltage Window (V)',
    'SSA (m²/g)',
    'Pore Volume (cm³/g)',
    'Pore Size (nm)',
    'Micropore Volume (cm³/g)',
    'SSA Micropores (m²/g)',
    'Id/Ig ratio',
    'N (%)', 'O (%)', 'B (%)', 'S (%)', 'F (%)', 'P (%)',
    'Material-1 (enc)',
    'Material-2 (enc)',
    'Electrolyte (enc)',
    'Is Composite',
    'Has Material Note',
]

X = df_clean[feature_cols].values
y = df_clean[target_col].values
print(f"\nFeature matrix: {X.shape}  |  Target range: {y.min():.1f} – {y.max():.1f} F/g")

# ══════════════════════════════════════════════════════════════════
# PHASE 3: MODEL DEVELOPMENT
# ══════════════════════════════════════════════════════════════════

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        max_features='sqrt', random_state=42, n_jobs=-1),
    'Gradient Boosting\n(XGBoost-equiv.)': GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=2, random_state=42),
}

# ══════════════════════════════════════════════════════════════════
# PHASE 4: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════

# Leakage-safe split FIRST, then cross-validate only on train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n=== 5-Fold CV on TRAIN set ===")
for name, model in models.items():
    y_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    r2_cv   = r2_score(y_train, y_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_train, y_cv))
    mae_cv  = mean_absolute_error(y_train, y_cv)

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    r2_test   = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test  = mean_absolute_error(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'y_cv': y_cv, 'y_test': y_pred_test,
        'cv':   {'R2': r2_cv,   'RMSE': rmse_cv,   'MAE': mae_cv},
        'test': {'R2': r2_test, 'RMSE': rmse_test, 'MAE': mae_test},
    }
    tag = name.replace('\n', ' ')
    print(f"  {tag:<45}  CV  R²={r2_cv:.4f} RMSE={rmse_cv:.2f} MAE={mae_cv:.2f}")
    print(f"  {'':<45}  Tst R²={r2_test:.4f} RMSE={rmse_test:.2f} MAE={mae_test:.2f}")

# ── Refit all models on FULL data for importance ──────────────────
for name, res in results.items():
    res['model'].fit(X, y)

# ══════════════════════════════════════════════════════════════════
# PLOT STYLE HELPERS
# ══════════════════════════════════════════════════════════════════

BG_DARK   = '#0f1117'
BG_AX     = '#1a1d27'
GRID_COL  = '#2a2d3a'
COLORS = {
    'Linear Regression':            '#4e8cff',
    'Random Forest':                '#2ec4b6',
    'Gradient Boosting\n(XGBoost-equiv.)': '#e86452',
}

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors='white', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(color=GRID_COL, lw=0.6, zorder=0)

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 – Predicted vs Experimental (CV + Test)
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(BG_DARK)

for col, (name, res) in enumerate(results.items()):
    c = COLORS[name]

    # Top row: CV predictions on train
    ax = axes[0, col]
    style_ax(ax)
    ax.scatter(y_train, res['y_cv'], color=c, alpha=0.6,
               edgecolors='white', lw=0.3, s=30, zorder=3)
    mn = min(y_train.min(), res['y_cv'].min()) - 10
    mx = max(y_train.max(), res['y_cv'].max()) + 10
    ax.plot([mn, mx], [mn, mx], 'w--', lw=1.2)
    ax.set_xlabel('Experimental (F/g)', color='white', fontsize=10)
    ax.set_ylabel('Predicted (F/g)',    color='white', fontsize=10)
    ax.set_title(f"{name.replace(chr(10),' ')}\n5-Fold CV", color=c, fontsize=11, fontweight='bold')
    m = res['cv']
    ax.text(0.04, 0.97, f"R²={m['R2']:.4f}\nRMSE={m['RMSE']:.2f}\nMAE={m['MAE']:.2f}",
            transform=ax.transAxes, color='white', fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#2a2d3a', alpha=0.85, boxstyle='round,pad=0.4'))

    # Bottom row: held-out test predictions
    ax = axes[1, col]
    style_ax(ax)
    ax.scatter(y_test, res['y_test'], color=c, alpha=0.7,
               edgecolors='white', lw=0.3, s=40, zorder=3)
    mn = min(y_test.min(), res['y_test'].min()) - 10
    mx = max(y_test.max(), res['y_test'].max()) + 10
    ax.plot([mn, mx], [mn, mx], 'w--', lw=1.2)
    ax.set_xlabel('Experimental (F/g)', color='white', fontsize=10)
    ax.set_ylabel('Predicted (F/g)',    color='white', fontsize=10)
    ax.set_title(f"Hold-out Test Set (20%)", color='#aaa', fontsize=10)
    m = res['test']
    ax.text(0.04, 0.97, f"R²={m['R2']:.4f}\nRMSE={m['RMSE']:.2f}\nMAE={m['MAE']:.2f}",
            transform=ax.transAxes, color='white', fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#2a2d3a', alpha=0.85, boxstyle='round,pad=0.4'))

fig.suptitle('Predicted vs Experimental Capacitance  —  CV (top) & Test Set (bottom)',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/fig1_pred_vs_exp.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print("\nSaved Fig 1")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 – Metrics comparison bar chart
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG_DARK)

metrics = ['R2', 'RMSE', 'MAE']
titles  = ['R²  ↑ higher = better', 'RMSE  ↓ lower = better', 'MAE  ↓ lower = better']
model_labels = [n.replace('\n(', '\n(') for n in results.keys()]
colors = list(COLORS.values())

for ax, metric, title in zip(axes, metrics, titles):
    cv_vals   = [res['cv'][metric]   for res in results.values()]
    test_vals = [res['test'][metric] for res in results.values()]
    x = np.arange(len(cv_vals))
    w = 0.35
    style_ax(ax)
    b1 = ax.bar(x - w/2, cv_vals,   w, color=colors, alpha=0.9,
                edgecolor='white', lw=0.4, label='CV')
    b2 = ax.bar(x + w/2, test_vals, w, color=colors, alpha=0.45,
                edgecolor='white', lw=0.4, label='Test', hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, color='white', fontsize=8.5)
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='white')
    for bar, v in zip(list(b1)+list(b2), cv_vals+test_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(cv_vals+test_vals)*0.01,
                f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=8)
    ax.legend(fontsize=8, facecolor='#2a2d3a', labelcolor='white', framealpha=0.8)

fig.suptitle('Model Performance Metrics  —  CV vs Hold-out Test',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig2_metrics.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print("Saved Fig 2")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 – Residual Analysis
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG_DARK)

for ax, (name, res) in zip(axes, results.items()):
    c = COLORS[name]
    resid = y_test - res['y_test']
    style_ax(ax)
    ax.scatter(res['y_test'], resid, color=c, alpha=0.7,
               edgecolors='white', lw=0.3, s=40)
    ax.axhline(0, color='white', ls='--', lw=1.2)
    ax.set_xlabel('Predicted Capacitance (F/g)', color='white', fontsize=10)
    ax.set_ylabel('Residual (F/g)', color='white', fontsize=10)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=11, fontweight='bold')
    ax.text(0.04, 0.97, f"Bias={resid.mean():.2f}\nStd={resid.std():.2f}",
            transform=ax.transAxes, color='white', fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#2a2d3a', alpha=0.85, boxstyle='round,pad=0.4'))

fig.suptitle('Residual Analysis  —  Hold-out Test Set',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig3_residuals.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print("Saved Fig 3")

# ══════════════════════════════════════════════════════════════════
# PHASE 5 – INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════

# ── FIGURE 4 – Intrinsic Feature Importance (RF + GB) ─────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG_DARK)

for ax, name in zip(axes, ['Random Forest', 'Gradient Boosting\n(XGBoost-equiv.)']):
    model = results[name]['model']
    imp   = model.feature_importances_
    idx   = np.argsort(imp)
    c     = COLORS[name]
    style_ax(ax)
    bars = ax.barh([feature_labels[i] for i in idx], imp[idx],
                   color=c, edgecolor='white', lw=0.3, alpha=0.9)
    # Annotate top 5
    top5 = np.argsort(imp)[-5:]
    for bar, i in zip(reversed(bars), reversed(idx)):
        if i in top5:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{imp[i]:.3f}', va='center', color='white', fontsize=8.5)
    ax.set_xlabel('Feature Importance (Gini)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(axis='x', color=GRID_COL, lw=0.6)

fig.suptitle('Feature Importance Analysis  —  Intrinsic (Gini Impurity)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig4_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print("Saved Fig 4")

# ── FIGURE 5 – Permutation Importance (SHAP-equivalent) ──────────

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG_DARK)

for ax, name in zip(axes, ['Random Forest', 'Gradient Boosting\n(XGBoost-equiv.)']):
    model = results[name]['model']
    perm  = permutation_importance(model, X_test, y_test,
                                   n_repeats=30, random_state=42, scoring='r2')
    mean_imp = perm.importances_mean
    std_imp  = perm.importances_std
    idx  = np.argsort(mean_imp)
    c    = COLORS[name]
    style_ax(ax)
    ax.barh([feature_labels[i] for i in idx], mean_imp[idx],
            xerr=std_imp[idx], color=c, edgecolor='white', lw=0.3,
            alpha=0.9, capsize=3,
            error_kw={'ecolor': 'white', 'alpha': 0.5, 'lw': 1.2})
    ax.axvline(0, color='white', lw=0.9, ls='--', alpha=0.5)
    ax.set_xlabel('Permutation Importance (ΔR²)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(axis='x', color=GRID_COL, lw=0.6)

fig.suptitle('Permutation-Based Feature Importance  —  SHAP-Equivalent',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig5_permutation_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print("Saved Fig 5")

# ── FIGURE 6 – Physical Insight: Top Features vs Capacitance ─────

# Identify top 4 physical features from RF importance
rf_imp = results['Random Forest']['model'].feature_importances_
phys_idx = np.argsort(rf_imp)[-4:][::-1]  # top 4

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(BG_DARK)
axes_flat = axes.flatten()

for ax, fi in zip(axes_flat, phys_idx):
    feat_name = feature_labels[fi]
    x_vals = X[:, fi]
    c_scatter = plt.cm.plasma((y - y.min()) / (y.max() - y.min()))
    style_ax(ax)
    sc = ax.scatter(x_vals, y, c=y, cmap='plasma', alpha=0.65,
                    edgecolors='white', lw=0.2, s=30, zorder=3)
    # Trend line
    finite = np.isfinite(x_vals) & np.isfinite(y)
    if finite.sum() > 5:
        z = np.polyfit(x_vals[finite], y[finite], 1)
        p = np.poly1d(z)
        xs = np.linspace(x_vals[finite].min(), x_vals[finite].max(), 100)
        ax.plot(xs, p(xs), 'w--', lw=1.5, alpha=0.7)
    ax.set_xlabel(feat_name, color='white', fontsize=10)
    ax.set_ylabel('Capacitance (F/g)', color='white', fontsize=10)
    ax.set_title(f'{feat_name}  |  imp={rf_imp[fi]:.4f}', color='#2ec4b6',
                 fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Capacitance (F/g)').ax.yaxis.label.set_color('white')

fig.suptitle('Physical Insight: Top 4 Features Governing Capacitance',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig6_physical_insight.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print("Saved Fig 6")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL METRICS SUMMARY")
print("="*70)
header = f"{'Model':<45} {'Set':<6} {'R²':>7} {'RMSE':>8} {'MAE':>8}"
print(header)
print("-"*70)
for name, res in results.items():
    tag = name.replace('\n', ' ')
    for split, m in [('CV', res['cv']), ('Test', res['test'])]:
        print(f"{tag:<45} {split:<6} {m['R2']:>7.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")
    print()

print("\n=== DOMINANT PHYSICAL PARAMETERS (RF Permutation Importance) ===")
rf_perm = permutation_importance(results['Random Forest']['model'],
                                  X_test, y_test, n_repeats=20, random_state=42)
ranked = np.argsort(rf_perm.importances_mean)[::-1]
for rank, fi in enumerate(ranked[:8], 1):
    print(f"  {rank}. {feature_labels[fi]:<35}  ΔR² = {rf_perm.importances_mean[fi]:+.4f} ± {rf_perm.importances_std[fi]:.4f}")

print("\n✅ All outputs saved.")
