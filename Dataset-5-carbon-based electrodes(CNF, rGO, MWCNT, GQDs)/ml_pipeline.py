"""
Redesigned Fig 9 — Permutation-Based Feature Importance (SHAP-Equivalent)
Style matches Image 2: horizontal bars, error bars, ΔR² on x-axis, white background.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import os

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ── Reproduce dataset exactly ─────────────────────────────────────────────────
NaN = np.nan
raw = [
    ("Carbon nanohair/CQDs/AC", "Thermal annealing", 0.5, "H2SO4",   988.8,  0.5,  0, 260.0, 1200),
    ("Mn3O4/CQDs",              "Hydrothermal",      1.0, "Na2SO4",  612.0,  1.0,  0, 96.3,  5000),
    ("NiCo2O4/GQDs",            "Hydrothermal",      1.0, "Li2SO4",  3940.0, 0.5,  0, 98.0,  5000),
    ("NiCo-LDH/GQDs",           "Solvothermal",      6.0, "KOH",     1628.0, 1.0,  0, 86.2,  8000),
    ("CuCo-LDH/N-doped GQDs",   "Hydrothermal",      6.0, "KOH",     1009.0, 1.0,  0, NaN,   NaN),
    ("MnO2/CNF",                "Electrospinning",   6.0, "KOH",     1114.0, 1.0,  0, 70.0,  5000),
    ("CNx/CNF",                 "Electrospinning",   6.0, "KOH",      483.0, 1.0,  0, NaN,   NaN),
    ("CNx/CNF",                 "Electrospinning",   1.0, "H2SO4",    456.0, 1.0,  0, NaN,   NaN),
    ("MnO2/Fe2O3/CNF",          "Electrospinning",   6.0, "KOH",      567.0, 1.0,  0, 94.0,  10000),
    ("Co-doped SnS/CNF",        "Electrospinning",   6.0, "KOH",      750.0, 1.0,  0, 95.6,  10000),
    ("FeP/CNF",                 "Hydrothermal",      3.0, "KOH",      299.2, 1.0,  0, NaN,   NaN),
    ("MnO2/CNF",                "Hydrothermal",      6.0, "KOH",      770.8, 0.5,  0, NaN,   NaN),
    ("Si/CNF",                  "Thermal annealing", 1.0, "H2SO4",    206.0, 1.0,  0, 94.6,  5000),
    ("N-doped NiO/MXene/CNF",   "Electrospinning",   3.0, "KOH",      871.0, 1.0,  0, 98.0,  5000),
    ("Gd2O3/CNF",               "Electrospinning",   1.0, "H2SO4",    162.3, 1.0,  0, 97.0,  10000),
    ("Sb2O5/N,S co-doped CNF",  "Electrospinning",   6.0, "KOH",      354.4, 1.0,  0, NaN,   NaN),
    ("CoS2/CoSe2/CNF",          "Electrospinning",   4.0, "KOH",      292.2, 1.0,  0, 130.8, 10000),
    ("MnSnO3/rGO",              "Microwave",         3.0, "KOH",      195.8, 2.0,  0, 124.0, 15000),
    ("Fe3O4/rGO",               "Microwave",         3.0, "KOH",      136.0, 1.0,  0, NaN,   NaN),
    ("Co3O4/SnO2/rGO",          "Microwave",         3.0, "KOH",      146.4, 1.0,  0, 102.4, 10000),
    ("FeNiS2/rGO",              "Hydrothermal",      2.0, "KOH",      1013.0,2.0,  0, 90.0,  10000),
    ("rGO/FeNi2S4/Co9S8",       "Hydrothermal",      6.0, "KOH",      1308.0,1.0,  0, 93.75, 8500),
    ("Ni-MOF/rGO aerogel",      "Solvothermal",      6.0, "KOH",      1644.0,1.0,  0, 87.8,  10000),
    ("Cu/Mn MOF/rGO",           "Hydrothermal",      3.0, "KOH",      362.2, 1.0,  0, 87.2,  1000),
    ("Pd nanoparticles/rGO",    "Microwave",         1.0, "KOH",      666.8, 5.0,  1, 92.5,  5500),
    ("CoO/N-doped rGO",         "Microwave",         1.0, "KOH",      744.1, 5.0,  1, 91.3,  5000),
    ("Fe3O4/rGO",               "Microwave",         1.0, "KOH",      771.3, 5.0,  1, 95.1,  5000),
    ("Co3O4-Fe2O3-NiO/graphene","Hydrothermal",      2.0, "KOH",      1040.0,1.0,  0, 92.0,  5000),
    ("Ni2P/rGO",                "Calcination",       6.0, "KOH",      2333.3,1.0,  0, 88.2,  10000),
    ("Ni-MOF/cellulose nf/rGO", "Freeze-drying",     1.0, "KOH",      189.9, 3.0,  1, 97.42, 20000),
    ("CuO/MnO2/N-doped MWCNT", "Hydrothermal",      5.0, "KOH",      184.0, 0.5,  0, 98.5,  5000),
    ("Polydopamine/PANI/MWCNT","In situ polymer.",   0.5, "H2SO4",    970.0, 5.0,  1, 91.0,  6000),
    ("NiFe2O4/CNT",             "Thermal annealing", 2.0, "KOH",      670.0,10.0,  1, 89.16, 5000),
    ("CoMn2O4/CNT",             "Electrodeposition", 2.0, "KOH",      732.0, 2.0,  1, NaN,   NaN),
    ("NiCo2S4/N-doped MWCNT",  "Hydrothermal",      6.0, "KOH",      254.0, 1.0,  0, 95.7,  10000),
    ("NiCo2S4/N-MWCNT/MOF-67", "Hydrothermal",      6.0, "KOH",      455.0, 1.0,  0, 98.43, 10000),
    ("Ni3P2O8/MWCNT",           "Chemical bath dep.",1.0, "Na2SO4",   839.3, 5.0,  1, 95.0,  5000),
    ("MoS2/Mn-MOF/MWCNT",      "Hydrothermal",      3.0, "KOH",      862.73,1.0,  0, 71.4,  5000),
]

cols = ["Electrode", "Synthesis", "ElecConc_M", "ElecType",
        "Capacitance_F_g", "MeasVal", "MeasType", "CycleStab_pct", "CycleCount"]
df = pd.DataFrame(raw, columns=cols)

# Imputation
med_stab   = df["CycleStab_pct"].median()
med_cycles = df["CycleCount"].median()
df["CycleStab_missing"]  = df["CycleStab_pct"].isnull().astype(int)
df["CycleCount_missing"] = df["CycleCount"].isnull().astype(int)
df["CycleStab_pct"].fillna(med_stab,   inplace=True)
df["CycleCount"].fillna(med_cycles, inplace=True)

le_syn  = LabelEncoder()
le_elec = LabelEncoder()
df["Synthesis_enc"] = le_syn.fit_transform(df["Synthesis"])
df["ElecType_enc"]  = le_elec.fit_transform(df["ElecType"])

feature_cols = [
    "ElecConc_M", "ElecType_enc", "Synthesis_enc",
    "MeasVal", "MeasType",
    "CycleStab_pct", "CycleCount",
    "CycleStab_missing", "CycleCount_missing",
]
FEAT_LABELS = [
    "Electrolyte Conc. (M)",
    "Electrolyte Type (enc.)",
    "Synthesis Process (enc.)",
    "Current Density / Scan Rate",
    "Measurement Type (CD/CV)",
    "Cycling Stability (%)",
    "Cycle Count",
    "CycleStab Missing Flag",
    "CycleCount Missing Flag",
]

X_raw = df[feature_cols].values
y     = df["Capacitance_F_g"].values
X     = SimpleImputer(strategy="median").fit_transform(X_raw)
X_sc  = StandardScaler().fit_transform(X)

# ── Fit models on full data ────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=2, random_state=42)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=3,
                                subsample=0.8, min_samples_leaf=2, random_state=42)
rf.fit(X_sc, y)
gb.fit(X_sc, y)

# ── Permutation importance with std (n_repeats=50) ────────────────────────────
perm_rf = permutation_importance(rf, X_sc, y, n_repeats=50, random_state=42,
                                  scoring="r2")
perm_gb = permutation_importance(gb, X_sc, y, n_repeats=50, random_state=42,
                                  scoring="r2")

imp_rf_mean = perm_rf.importances_mean
imp_rf_std  = perm_rf.importances_std
imp_gb_mean = perm_gb.importances_mean
imp_gb_std  = perm_gb.importances_std

# ── Sort by RF importance (descending) ───────────────────────────────────────
order_rf = np.argsort(imp_rf_mean)          # ascending → plot bottom = highest
order_gb = np.argsort(imp_gb_mean)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
TEAL = "#20B2AA"
SALM = "#E8735A"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                facecolor="white",
                                gridspec_kw={"wspace": 0.55})
fig.patch.set_facecolor("white")

# shared y-tick positions
y_pos = np.arange(len(FEAT_LABELS))

# ── LEFT panel: Random Forest ─────────────────────────────────────────────────
ax1.set_facecolor("white")

ax1.barh(
    y_pos,
    imp_rf_mean[order_rf],
    xerr=imp_rf_std[order_rf],
    color=TEAL,
    edgecolor="white",
    height=0.55,
    capsize=4,
    error_kw={"elinewidth": 1.4, "ecolor": "#555555", "capthick": 1.4},
    alpha=0.90,
    zorder=3,
)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([FEAT_LABELS[i] for i in order_rf], fontsize=10.5)
ax1.set_xlabel("Permutation Importance (ΔR²)", fontsize=11, labelpad=8)
ax1.set_title("Random Forest", fontsize=13, fontweight="bold",
              color=TEAL, pad=10)
ax1.axvline(0, color="#999999", lw=1.2, linestyle="--", zorder=2)
ax1.grid(axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
ax1.grid(axis="y", visible=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_color("#CCCCCC")
ax1.spines["bottom"].set_color("#CCCCCC")
ax1.tick_params(axis="both", labelsize=10)

# ── RIGHT panel: Gradient Boosting ───────────────────────────────────────────
ax2.set_facecolor("white")

ax2.barh(
    y_pos,
    imp_gb_mean[order_gb],
    xerr=imp_gb_std[order_gb],
    color=SALM,
    edgecolor="white",
    height=0.55,
    capsize=4,
    error_kw={"elinewidth": 1.4, "ecolor": "#555555", "capthick": 1.4},
    alpha=0.90,
    zorder=3,
)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([FEAT_LABELS[i] for i in order_gb], fontsize=10.5)
ax2.set_xlabel("Permutation Importance (ΔR²)", fontsize=11, labelpad=8)
ax2.set_title("Gradient Boosting (XGBoost-equiv.)", fontsize=13, fontweight="bold",
              color=SALM, pad=10)
ax2.axvline(0, color="#999999", lw=1.2, linestyle="--", zorder=2)
ax2.grid(axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
ax2.grid(axis="y", visible=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#CCCCCC")
ax2.spines["bottom"].set_color("#CCCCCC")
ax2.tick_params(axis="both", labelsize=10)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Permutation-Based Feature Importance  —  SHAP-Equivalent",
    fontsize=14, fontweight="bold", y=1.02, color="#1a1a1a"
)

plt.savefig(f"{OUT}/Fig9_Permutation_Importance_Redesigned.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: Fig9_Permutation_Importance_Redesigned.png")
