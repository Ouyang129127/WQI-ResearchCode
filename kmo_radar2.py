# -*- coding: utf-8 -*-
"""
第二个KMO雷达图：
① AHP单独 → 各指标KMO
② RF单独 → 各指标KMO
③ RF-CRITIC(constrained)单独 → 各指标KMO
④ AHP+RF-CRITIC Nash融合 → 各指标KMO
"""

import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── Chinese font ──
for fp in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(fp):
        zh = font_manager.FontProperties(fname=fp)
        plt.rcParams['font.family'] = zh.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Font: {fp}")
        break

# ── Paths ──
WEIGHT_DIR = r"D:\WQIPaper\DataAnalytics\权重表"
WQ_PATH = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
OUT_DIR = r"D:\WQIPaper\DataAnalytics"

# ── Load water quality data ──
wq = pd.read_excel(WQ_PATH)
feature_cols = [c for c in wq.columns if c not in ['ID', 'Date']]
X = wq[feature_cols].values.astype(np.float64)
n_samples, n_vars = X.shape
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=1)
X_std[X_std < 1e-12] = 1e-12
Z = (X - X_mean) / X_std
R = np.corrcoef(Z, rowvar=False)

# ── Standard KMO computation ──
R_inv = np.linalg.inv(R)
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
A = D_inv_sqrt @ R_inv @ D_inv_sqrt
P = np.eye(n_vars)
for i in range(n_vars):
    for j in range(n_vars):
        if i != j:
            P[i, j] = -A[i, j] / np.sqrt(A[i, i] * A[j, j])

def weighted_kmo_for_method(w_dict, indicator_order):
    """
    Compute weighted per-indicator KMO using DIRECT weights.
    KMO_i = Σ_{j≠i} w_j * r²_{ij} / (Σ_{j≠i} w_j*(r²_{ij} + p²_{ij}))
    """
    # Map indicator names to data columns
    name_to_col = {}
    for fc in feature_cols:
        for io in indicator_order:
            if io.lower() == fc.lower() or io in fc or fc in io:
                name_to_col[io] = feature_cols.index(fc)
                break
    # Build weight vector
    w_vec = np.ones(n_vars)
    for io_name, io_idx in name_to_col.items():
        w_vec[io_idx] = w_dict.get(io_name, 1.0 / n_vars)
    
    kmo_w = np.zeros(n_vars)
    for i in range(n_vars):
        num = 0.0; den = 0.0
        for j in range(n_vars):
            if i == j: continue
            w_j = w_vec[j]
            num += w_j * R[i, j]**2
            den += w_j * (R[i, j]**2 + P[i, j]**2)
        kmo_w[i] = num / den if den > 1e-12 else 0
    
    # Map back
    kmo_dict = {}
    for io_name in indicator_order:
        idx = name_to_col.get(io_name, 0)
        kmo_dict[io_name] = kmo_w[idx]
    return kmo_dict

# ── Load AHP ──
ahp = pd.read_excel(os.path.join(WEIGHT_DIR, "AHP_Weights.xlsx"), engine='openpyxl')
ahp_order = ahp['Indicator'].tolist()
w_ahp_raw = ahp.set_index('Indicator')['AHP_Global_Weight'].to_dict()
w_ahp_vals = np.array(list(w_ahp_raw.values()))
w_ahp_vals = w_ahp_vals / w_ahp_vals.sum()
w_ahp = dict(zip(w_ahp_raw.keys(), w_ahp_vals))

# ── Load RF ──
rf = pd.read_excel(os.path.join(WEIGHT_DIR, "RF_Weights.xlsx"), engine='openpyxl')
rf_map = dict(zip(rf['Indicators'], rf['Combined_raw'] / 100))
w_rf = {k: rf_map.get(k, 0) for k in ahp_order}

# ── Load RF-CRITIC constr ──
nash = pd.read_excel(os.path.join(WEIGHT_DIR, "Nash_Equilibrium_Weights.xlsx"),
                     sheet_name='Nash_Weights', engine='openpyxl')
rf_critic_map = {}
for _, row in nash.iterrows():
    val = row['ConstrainedQP_Nash_fmt']
    rf_critic_map[row['Indicators']] = float(val.replace(' %', '')) / 100
w_rf_critic = {k: rf_critic_map.get(k, 0) for k in ahp_order}

# ── AHP+RF-CRITIC Nash fusion (already computed) ──
w_subj_arr = np.array([w_ahp[k] for k in ahp_order])
w_obj_arr = np.array([w_rf_critic[k] for k in ahp_order])
w_ahp_rf_critic_nash = (w_subj_arr + w_obj_arr) / 2
w_ahp_rf_critic_nash = w_ahp_rf_critic_nash / w_ahp_rf_critic_nash.sum()
w_ahp_rf_critic_dict = dict(zip(ahp_order, w_ahp_rf_critic_nash))

# ── Compute KMO for each method ──
methods = {
    '① AHP (主观)': w_ahp,
    '② RF (客观)': w_rf,
    '③ RF-CRITIC constr (客观)': w_rf_critic,
    '④ AHP+RF-CRITIC Nash融合': w_ahp_rf_critic_dict,
}

kmo_results = {}
for label, w_dict in methods.items():
    kmo_dict = weighted_kmo_for_method(w_dict, ahp_order)
    kmo_results[label] = kmo_dict
    
    # Map to feature_cols for printing
    mapped = {}
    for fc in feature_cols:
        for an in ahp_order:
            if an.lower() == fc.lower() or an in fc or fc in an:
                mapped[fc] = kmo_dict.get(an, 0)
                break
    
    overall = 0
    # compute overall weighted KMO
    w_vec = np.ones(n_vars)
    for fc in feature_cols:
        for an in ahp_order:
            if an.lower() == fc.lower() or an in fc or fc in an:
                w_vec[feature_cols.index(fc)] = w_dict.get(an, 1/n_vars)
                break
    total_num = 0; total_den = 0
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            w_pair = w_vec[i] * w_vec[j]
            total_num += w_pair * R[i, j]**2
            total_den += w_pair * (R[i, j]**2 + P[i, j]**2)
    overall = total_num / total_den if total_den > 1e-12 else 0
    
    print(f"\n{label}:  Overall KMO = {overall:.4f}")
    for fc in feature_cols:
        print(f"  {fc:40s}: {mapped.get(fc, 0):.4f}")
    print(f"  {'─'*50}")

# ── Radar chart ──
categories = feature_cols
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
colors = ['#333333', '#E63946', '#F4A261', '#2A9D8F']

for idx, (label, kmo_dict) in enumerate(kmo_results.items()):
    # Map to feature_cols order
    values = []
    for fc in feature_cols:
        for an in ahp_order:
            if an.lower() == fc.lower() or an in fc or fc in an:
                values.append(kmo_dict.get(an, 0))
                break
    values += values[:1]  # close
    
    ax.fill(angles, values, alpha=0.05, color=colors[idx])
    ax.plot(angles, values, 'o-', linewidth=2.0, color=colors[idx],
            label=label, markersize=5, markerfacecolor=colors[idx])

ax.set_xticks(angles[:-1])
short = [c[:20] if len(c) > 20 else c for c in categories]
ax.set_xticklabels(short, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_rlabel_position(30)
ax.set_title('AHP/RF/RF-CRITIC及融合方案下各指标KMO对比\n'
             'Weighted KMO per Indicator: Single Methods vs Nash Fusion',
             fontsize=14, pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

out = os.path.join(OUT_DIR, "Figure_KMO_Radar_AHP_RF_RFCRITIC.png")
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nRadar chart saved: {out}")
print("[DONE] 🦐")
