"""
Supercapacitor Carbon Materials — Full ML Pipeline
====================================================
Data source : Table 1 (Materials Advances review, extracted from screenshot)
Phases      : Data cleaning → Feature engineering → Model dev →
              Evaluation → Interpretability & Physical insight
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer

# ══════════════════════════════════════════════════════════════════
# RAW DATA  (extracted from screenshot)
# ══════════════════════════════════════════════════════════════════
nan = np.nan
raw = {
    'Material': [
        'N-doped porous carbon nanosheets',
        'N/O/S-doped hierarchical porous carbon materials',
        'Mesoporous hollow carbon spheres',
        'N/S-co-doped carbon nanobowls',
        'N-doped multi-chamber carbon microspheres',
        'N-rich porous carbon nanosheets',
        '2D porous carbon nanosheets',
        'Porous carbon nanorods',
        'N-doped porous carbon nanofibrous microspheres',
        'Spheres-in-tube hierarchical porous carbon',
        'N-doped porous carbon superstructures',
        'Porous interconnected carbon nanosheets',
        'Ultramicroporous carbon materials',
        'N/S-doped porous carbon',
        'Order mesoporous carbon spheres',
        'N-rich mesoporous carbons',
        'N/O-doped hierarchical porous carbon nanorods',
        'Honeycomb-like porous carbon foam',
        'B/N-codoped carbon nanosheets',
        'N-doped microporous carbon spheres',
    ],
    'Surface area (m2/g)': [
        1786.1, 3519.5, 1582.0, 1567.0, 1797.0, 2406.0, 1907.0, 1559.0,
        1147.0, 318.0, 1375.0, 2220.0, 1312.0, 1339.0, 1186.0, 458.0,
        1882.0, 1313.0, 2362.0, 1478.0,
    ],
    'Pore volume (cm3/g)': [
        0.8157, 2.68, 2.45, 2.25, 0.96, nan, 0.77, nan,
        2.12, 0.78, 0.996, 1.11, 0.67, 0.96, 0.27, 0.42,
        nan, 0.716, 1.448, 0.76,
    ],
    'N (at%)': [
        2.10, nan, nan, 3.30, 4.58, 9.40, 1.54, 1.47,
        2.40, 8.74, 3.46, nan, nan, 4.50, nan, 19.10,
        8.10, 1.10, 3.10, 8.71,
    ],
    'O (at%)': [
        7.11, nan, nan, nan, 2.12, 4.70, 6.59, 0.62,
        6.10, 3.39, 7.99, nan, nan, nan, nan, nan,
        10.00, 11.20, nan, 7.89,
    ],
    'S (at%)': [
        nan, nan, nan, 1.70, nan, nan, nan, nan,
        nan, nan, nan, nan, nan, 5.80, nan, nan,
        nan, nan, nan, nan,
    ],
    'B (at%)': [
        nan, nan, nan, nan, nan, nan, nan, nan,
        nan, nan, nan, nan, nan, nan, nan, nan,
        nan, nan, 0.50, nan,
    ],
    'Electrolyte': [
        '6M KOH', '1M H2SO4', '6M KOH', '6M KOH', '6M KOH',
        'EMIMBF4', 'EMIMBF4', '1M H2SO4', 'EMIMTFSI',
        '1M Na2SO4', '6M KOH', '1M TEABF4/ACN', 'EMIMBF4',
        '6M KOH', '6M KOH', '1M H2SO4', 'EMIMBF4',
        '1M Na2SO4', '1M Na2SO4', '6M KOH',
    ],
    'Current density (A/g)': [
        0.25, 1.0, 1.0, 0.1, 0.2, 0.5, 1.0, 0.05,
        0.005, 0.2, 0.6, 0.1, 0.2, 0.2, 1.0, 0.2,
        0.2, 0.002, 0.5, 1.0,
    ],
    'Capacitance (F/g)': [
        339, 549, 310, 279, 301, 250, 221, 187,
        113, nan, 364, 150, 223, 464, 226.1, 252,
        214, 260, 235.6, 292,
    ],
    'Energy density (Wh/kg)': [
        11.77, 12.70, nan, 9.60, nan, 139.0, 94.0, nan,
        58.70, 29.50, nan, nan, 32.50, 16.20, 27.0, nan,
        116.50, 29.30, 30.10, 8.75,
    ],
    'Power density (W/kg)': [
        34.11, nan, nan, 25.0, nan, 500.0, 1800.0, nan,
        300.0, 401.0, nan, 13000.0, nan, 50.0, 980.0, nan,
        472.0, nan, 225.1, 500.0,
    ],
    'Ref': [29,30,34,35,36,40,46,53,54,59,65,72,75,78,85,91,105,119,124,130],
}

df = pd.DataFrame(raw)
print("Raw data shape:", df.shape)
print("\nMissing values per column (before cleaning):")
print(df.isnull().sum().to_string())

# ══════════════════════════════════════════════════════════════════
# PHASE 1 & 2 — DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

# ── 1. Infer heteroatom doping flags from Material name ──────────
# Many N/O/S/B values are missing despite being named in the material
# Use the name as a semantic signal → binary doping flags

def has_element(name, element):
    """Return 1 if element symbol found in material name."""
    import re
    pattern = rf'\b{element}[/-]?'
    return int(bool(re.search(pattern, name, re.IGNORECASE)))

df['N_doped'] = df['Material'].apply(lambda x: has_element(x, 'N'))
df['O_doped'] = df['Material'].apply(lambda x: has_element(x, 'O'))
df['S_doped'] = df['Material'].apply(lambda x: has_element(x, 'S'))
df['B_doped'] = df['Material'].apply(lambda x: has_element(x, 'B'))

# Total heteroatom content: sum available values; 0 for truly undoped
df['Total_heteroatom (at%)'] = df[['N (at%)', 'O (at%)', 'S (at%)', 'B (at%)']].sum(axis=1, min_count=1)

# ── 2. Pore volume — 3 missing → KNN impute (k=3) using SSA ──────
#    Physical rationale: SSA and pore volume are strongly correlated
imputer = KNNImputer(n_neighbors=3)
df[['Surface area (m2/g)', 'Pore volume (cm3/g)']] = imputer.fit_transform(
    df[['Surface area (m2/g)', 'Pore volume (cm3/g)']])

# ── 3. Individual heteroatom columns — high missingness ──────────
#    Strategy: fill with 0 when the material name gives no evidence
#    of that element; keep actual values where measured.
for col, flag in [('N (at%)', 'N_doped'), ('O (at%)', 'O_doped'),
                  ('S (at%)', 'S_doped'), ('B (at%)', 'B_doped')]:
    # If doped but value missing → median impute among doped samples
    doped_mask  = df[flag] == 1
    undoped_mask = df[flag] == 0
    median_val  = df.loc[doped_mask & df[col].notna(), col].median()
    df.loc[doped_mask & df[col].isna(), col] = median_val
    df.loc[undoped_mask & df[col].isna(), col] = 0.0

# ── 4. Electrolyte — encode electrolyte type ──────────────────────
#    Group into: aqueous-alkaline, aqueous-acid, aqueous-neutral, ionic-liquid, organic
def elec_group(e):
    e = e.upper()
    if 'KOH' in e:             return 'aqueous_alkaline'
    if 'H2SO4' in e or 'HCL' in e: return 'aqueous_acid'
    if 'NA2SO4' in e or 'NACL' in e: return 'aqueous_neutral'
    if 'EMIM' in e or 'BMIM' in e: return 'ionic_liquid'
    return 'organic'  # TEABF4/ACN etc.

df['Electrolyte_group'] = df['Electrolyte'].apply(elec_group)
le = LabelEncoder()
df['Electrolyte_enc'] = le.fit_transform(df['Electrolyte_group'])

# ── 5. Target: Capacitance — 1 missing (row 9) ───────────────────
#    Drop row 9 (target unknown → cannot train/evaluate on it)
df_model = df.dropna(subset=['Capacitance (F/g)']).copy()
print(f"\nRows after dropping missing target: {len(df_model)}")

# ── 6. Energy & Power density ─────────────────────────────────────
#    These are DERIVED from capacitance → would cause leakage.
#    Drop them from feature set (keep in full df for reference only).

# ── 7. Ref column — identifier, drop ─────────────────────────────
df_model.drop(columns=['Ref'], inplace=True)

# Save cleaned dataset
df_model.to_csv('/home/claude/cleaned_supercap_img.csv', index=False)

# Recompute Total_heteroatom after all individual cols are fully imputed
df_model["Total_heteroatom (at%)"] = df_model[["N (at%)", "O (at%)", "S (at%)", "B (at%)"]].sum(axis=1)


print("\nCleaned dataset:")
print(df_model[['Material','Surface area (m2/g)','Pore volume (cm3/g)',
                'N (at%)','O (at%)','S (at%)','B (at%)',
                'Total_heteroatom (at%)','Electrolyte_group',
                'Current density (A/g)','Capacitance (F/g)']].to_string())

# ── 8. Feature matrix ─────────────────────────────────────────────
feature_cols = [
    'Surface area (m2/g)',
    'Pore volume (cm3/g)',
    'N (at%)',
    'O (at%)',
    'S (at%)',
    'B (at%)',
    'Total_heteroatom (at%)',
    'Current density (A/g)',
    'Electrolyte_enc',
    'N_doped', 'O_doped', 'S_doped', 'B_doped',
]
feature_labels = [
    'Surface Area (m²/g)',
    'Pore Volume (cm³/g)',
    'N content (at%)',
    'O content (at%)',
    'S content (at%)',
    'B content (at%)',
    'Total Heteroatom (at%)',
    'Current Density (A/g)',
    'Electrolyte Type',
    'N-doped (binary)',
    'O-doped (binary)',
    'S-doped (binary)',
    'B-doped (binary)',
]

X = df_model[feature_cols].values.astype(float)
y = df_model['Capacitance (F/g)'].values.astype(float)

print(f"\nFeature matrix : {X.shape}")
print(f"Target range   : {y.min():.1f} – {y.max():.1f} F/g  |  mean={y.mean():.1f}")

# ══════════════════════════════════════════════════════════════════
# PHASE 3 — MODEL DEVELOPMENT
# ══════════════════════════════════════════════════════════════════
# n=19 samples → use Leave-One-Out CV (most robust for tiny datasets)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=2,
        max_features='sqrt', random_state=42),
    'Gradient Boosting\n(XGBoost-equiv.)': GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=2, random_state=42),
}

# ══════════════════════════════════════════════════════════════════
# PHASE 4 — MODEL EVALUATION  (LOO-CV + 5-Fold CV)
# ══════════════════════════════════════════════════════════════════

loo = LeaveOneOut()
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print("\n" + "="*72)
print("MODEL EVALUATION")
print("="*72)

for name, model in models.items():
    tag = name.replace('\n', ' ')

    # LOO-CV (n=19 → gold standard for small n)
    y_loo = cross_val_predict(model, X, y, cv=loo)
    r2_loo   = r2_score(y, y_loo)
    rmse_loo = np.sqrt(mean_squared_error(y, y_loo))
    mae_loo  = mean_absolute_error(y, y_loo)

    # 5-Fold CV
    y_5f = cross_val_predict(model, X, y, cv=kf5)
    r2_5f   = r2_score(y, y_5f)
    rmse_5f = np.sqrt(mean_squared_error(y, y_5f))
    mae_5f  = mean_absolute_error(y, y_5f)

    model.fit(X, y)   # refit on full data for importance

    results[name] = {
        'model': model,
        'y_loo': y_loo, 'y_5f': y_5f,
        'loo': {'R2': r2_loo, 'RMSE': rmse_loo, 'MAE': mae_loo},
        '5f':  {'R2': r2_5f,  'RMSE': rmse_5f,  'MAE': mae_5f},
    }
    print(f"\n  {tag}")
    print(f"    LOO-CV   R²={r2_loo:.4f}  RMSE={rmse_loo:.2f}  MAE={mae_loo:.2f}")
    print(f"    5-Fold   R²={r2_5f:.4f}  RMSE={rmse_5f:.2f}  MAE={mae_5f:.2f}")

# ══════════════════════════════════════════════════════════════════
# PLOT STYLING
# ══════════════════════════════════════════════════════════════════
BG   = '#0f1117'
AX   = '#1a1d27'
GRID = '#252836'
COLORS = {
    'Linear Regression':                    '#4e8cff',
    'Random Forest':                        '#2ec4b6',
    'Gradient Boosting\n(XGBoost-equiv.)':  '#e86452',
}

def style(ax, legend=False):
    ax.set_facecolor(AX)
    ax.tick_params(colors='white', labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#3a3d4a')
    ax.grid(color=GRID, lw=0.6, zorder=0)
    if legend:
        ax.legend(facecolor='#252836', labelcolor='white',
                  fontsize=8, framealpha=0.9)

def stat_box(ax, metrics, pos=(0.04, 0.97)):
    txt = (f"R²   = {metrics['R2']:+.4f}\n"
           f"RMSE = {metrics['RMSE']:.2f}\n"
           f"MAE  = {metrics['MAE']:.2f}")
    ax.text(*pos, txt, transform=ax.transAxes, color='white',
            fontsize=8.5, va='top', fontfamily='monospace',
            bbox=dict(facecolor='#252836', alpha=0.9, boxstyle='round,pad=0.45'))

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Predicted vs Experimental  (LOO top, 5-Fold bottom)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(BG)

for col, (name, res) in enumerate(results.items()):
    c = COLORS[name]
    for row, (split_key, y_pred, title_suffix) in enumerate([
            ('loo', res['y_loo'], 'Leave-One-Out CV'),
            ('5f',  res['y_5f'],  '5-Fold CV')]):
        ax = axes[row, col]
        style(ax)
        ax.scatter(y, y_pred, color=c, s=60, alpha=0.85,
                   edgecolors='white', lw=0.5, zorder=3)
        # Label each point with material abbreviation
        abbrevs = [m.split()[0][:4] for m in df_model['Material']]
        for xi, yi, lbl in zip(y, y_pred, abbrevs):
            ax.annotate(lbl, (xi, yi), fontsize=6, color='#aaa',
                        xytext=(3, 3), textcoords='offset points')
        mn = min(y.min(), y_pred.min()) - 15
        mx = max(y.max(), y_pred.max()) + 15
        ax.plot([mn, mx], [mn, mx], 'w--', lw=1.3, label='Ideal (y=x)')
        ax.set_xlabel('Experimental Capacitance (F/g)', color='white', fontsize=10)
        ax.set_ylabel('Predicted Capacitance (F/g)',    color='white', fontsize=10)
        ax.set_title(f"{name.replace(chr(10),' ')}  ·  {title_suffix}",
                     color=c, fontsize=10.5, fontweight='bold')
        stat_box(ax, res[split_key])

fig.suptitle('Predicted vs Experimental Capacitance  —  LOO-CV (top) & 5-Fold CV (bottom)',
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/fig1_pred_vs_exp.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("\nSaved Fig 1")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Metrics comparison  (LOO vs 5-Fold, grouped)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(BG)
metrics_list = ['R2', 'RMSE', 'MAE']
metric_titles = ['R²  ↑ higher = better', 'RMSE  ↓ lower = better', 'MAE  ↓ lower = better']
mnames = list(results.keys())
colors = [COLORS[n] for n in mnames]
x = np.arange(len(mnames))
w = 0.35

for ax, met, mtitle in zip(axes, metrics_list, metric_titles):
    style(ax)
    loo_vals = [results[n]['loo'][met] for n in mnames]
    f5_vals  = [results[n]['5f'][met]  for n in mnames]
    b1 = ax.bar(x - w/2, loo_vals, w, color=colors, alpha=0.95,
                edgecolor='white', lw=0.5, label='LOO-CV')
    b2 = ax.bar(x + w/2, f5_vals,  w, color=colors, alpha=0.45,
                edgecolor='white', lw=0.5, label='5-Fold', hatch='///')
    top = max(loo_vals + f5_vals)
    for bar, v in zip(list(b1)+list(b2), loo_vals+f5_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + top*0.012,
                f'{v:.3f}', ha='center', va='bottom',
                color='white', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('\n', '\n') for n in mnames],
                        color='white', fontsize=8.5)
    ax.set_title(mtitle, color='white', fontsize=11)
    ax.tick_params(colors='white')
    loo_p = mpatches.Patch(facecolor='#888', label='LOO-CV')
    f5_p  = mpatches.Patch(facecolor='#888', alpha=0.45, hatch='///', label='5-Fold')
    ax.legend(handles=[loo_p, f5_p], facecolor='#252836',
              labelcolor='white', fontsize=8, framealpha=0.9)

fig.suptitle('Model Performance Metrics  —  LOO-CV vs 5-Fold CV',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig2_metrics.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 2")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Residual Analysis
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG)

for ax, (name, res) in zip(axes, results.items()):
    c = COLORS[name]
    resid = y - res['y_loo']
    style(ax)
    ax.scatter(res['y_loo'], resid, color=c, s=60, alpha=0.85,
               edgecolors='white', lw=0.4, zorder=3)
    abbrevs = [m.split()[0][:4] for m in df_model['Material']]
    for xi, ri, lbl in zip(res['y_loo'], resid, abbrevs):
        ax.annotate(lbl, (xi, ri), fontsize=6, color='#aaa',
                    xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='white', ls='--', lw=1.3)
    ax.set_xlabel('Predicted Capacitance (F/g)', color='white', fontsize=10)
    ax.set_ylabel('Residual (F/g)', color='white', fontsize=10)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=11, fontweight='bold')
    ax.text(0.04, 0.97,
            f"Bias = {resid.mean():.2f}\nStd  = {resid.std():.2f}",
            transform=ax.transAxes, color='white', fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#252836', alpha=0.9, boxstyle='round,pad=0.4'))

fig.suptitle('Residual Analysis  —  LOO-CV Predictions',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig3_residuals.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 3")

# ══════════════════════════════════════════════════════════════════
# PHASE 5 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════

# ── FIGURE 4 — Intrinsic Feature Importance (RF + GB) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)

for ax, name in zip(axes, ['Random Forest',
                             'Gradient Boosting\n(XGBoost-equiv.)']):
    model = results[name]['model']
    imp   = model.feature_importances_
    idx   = np.argsort(imp)
    c     = COLORS[name]
    style(ax)
    colors_bar = [c if imp[i] >= np.percentile(imp, 70) else '#555' for i in idx]
    bars = ax.barh([feature_labels[i] for i in idx], imp[idx],
                   color=colors_bar, edgecolor='white', lw=0.3, alpha=0.92)
    for bar, i in zip(bars, idx):
        if imp[i] >= np.percentile(imp, 70):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height()/2,
                    f'{imp[i]:.4f}', va='center', color='white', fontsize=8.5)
    ax.set_xlabel('Feature Importance (Gini)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_color('#3a3d4a')

fig.suptitle('Feature Importance  —  Intrinsic (Gini / Mean Decrease Impurity)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig4_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 4")

# ── FIGURE 5 — Permutation Importance (SHAP-equivalent) ──────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)

perm_results = {}
for ax, name in zip(axes, ['Random Forest',
                             'Gradient Boosting\n(XGBoost-equiv.)']):
    model = results[name]['model']
    perm  = permutation_importance(model, X, y, n_repeats=50,
                                   random_state=42, scoring='r2')
    mean_imp = perm.importances_mean
    std_imp  = perm.importances_std
    perm_results[name] = {'mean': mean_imp, 'std': std_imp}
    idx  = np.argsort(mean_imp)
    c    = COLORS[name]
    colors_bar = [c if mean_imp[i] >= 0 else '#c0392b' for i in idx]
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_color('#3a3d4a')
    ax.barh([feature_labels[i] for i in idx], mean_imp[idx],
            xerr=std_imp[idx], color=colors_bar, edgecolor='white',
            lw=0.3, alpha=0.92, capsize=4,
            error_kw={'ecolor': 'white', 'alpha': 0.55, 'lw': 1.3})
    ax.axvline(0, color='white', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('Permutation Importance (ΔR²)', color='white', fontsize=11)
    ax.set_title(name.replace('\n', ' '), color=c, fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(axis='x', color=GRID, lw=0.6)

fig.suptitle('Permutation-Based Feature Importance  —  SHAP Equivalent\n'
             '(Red bars = feature hurts the model when shuffled → unreliable / correlated)',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig5_permutation_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 5")

# ── FIGURE 6 — Physical Insight: Top 4 features vs Capacitance ────
rf_perm_mean = perm_results['Random Forest']['mean']
top4_idx = np.argsort(rf_perm_mean)[::-1][:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(BG)

for ax, fi in zip(axes.flatten(), top4_idx):
    x_vals = X[:, fi]
    style(ax)
    sc = ax.scatter(x_vals, y, c=y, cmap='plasma', s=70, alpha=0.85,
                    edgecolors='white', lw=0.4, zorder=3,
                    vmin=y.min(), vmax=y.max())
    # Material labels
    abbrevs = [m.split()[0][:6] for m in df_model['Material']]
    for xi, yi, lbl in zip(x_vals, y, abbrevs):
        ax.annotate(lbl, (xi, yi), fontsize=6.5, color='#ccc',
                    xytext=(4, 4), textcoords='offset points')
    # Trend line
    finite = np.isfinite(x_vals) & np.isfinite(y)
    if finite.sum() > 4:
        z  = np.polyfit(x_vals[finite], y[finite], 1)
        xs = np.linspace(x_vals[finite].min(), x_vals[finite].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), 'w--', lw=1.6, alpha=0.7)
        # Pearson r
        r = np.corrcoef(x_vals[finite], y[finite])[0, 1]
        ax.text(0.97, 0.04, f'r = {r:+.3f}', transform=ax.transAxes,
                color='white', fontsize=9, ha='right', va='bottom',
                fontfamily='monospace',
                bbox=dict(facecolor='#252836', alpha=0.85, boxstyle='round,pad=0.3'))
    ax.set_xlabel(feature_labels[fi], color='white', fontsize=10)
    ax.set_ylabel('Capacitance (F/g)', color='white', fontsize=10)
    ax.set_title(f'#{list(top4_idx).index(fi)+1}  {feature_labels[fi]}'
                 f'   ΔR²={rf_perm_mean[fi]:+.4f}',
                 color='#2ec4b6', fontsize=11, fontweight='bold')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Capacitance (F/g)', color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

fig.suptitle('Physical Insight  —  Top 4 Features Governing Supercapacitor Capacitance',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig6_physical_insight.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 6")

# ── FIGURE 7 — Feature Correlation Heatmap ────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX)

feat_df = pd.DataFrame(X, columns=feature_labels)
feat_df['Capacitance (F/g)'] = y
corr = feat_df.corr()

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', color='white', fontsize=8)
ax.set_yticklabels(corr.columns, color='white', fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                color='white' if abs(v) > 0.5 else '#aaa', fontsize=7)
cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label('Pearson r', color='white', fontsize=10)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
ax.set_title('Feature Correlation Matrix  (incl. Target)', color='white',
             fontsize=13, fontweight='bold', pad=15)
for sp in ax.spines.values(): sp.set_color('#3a3d4a')

plt.tight_layout()
plt.savefig('/home/claude/fig7_correlation.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved Fig 7")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("FINAL METRICS SUMMARY")
print("="*72)
print(f"{'Model':<45} {'CV':<7} {'R²':>7} {'RMSE':>8} {'MAE':>8}")
print("-"*72)
for name, res in results.items():
    tag = name.replace('\n', ' ')
    for split, key in [('LOO', 'loo'), ('5-Fold', '5f')]:
        m = res[key]
        print(f"{tag:<45} {split:<7} {m['R2']:>7.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")
    print()

print("\n=== DOMINANT PHYSICAL PARAMETERS (RF Permutation Importance) ===")
rf_perm_full = perm_results['Random Forest']
ranked = np.argsort(rf_perm_full['mean'])[::-1]
for rank, fi in enumerate(ranked, 1):
    sign = '+' if rf_perm_full['mean'][fi] >= 0 else ''
    print(f"  {rank:>2}. {feature_labels[fi]:<35}  "
          f"ΔR² = {sign}{rf_perm_full['mean'][fi]:.4f} "
          f"± {rf_perm_full['std'][fi]:.4f}")

print("\n✅ All figures and cleaned CSV saved.")
