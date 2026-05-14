# -*- coding: utf-8 -*-
"""
============================================================
水体指标权重融合 — 无约束纳什均衡（AHP主观 + 4种客观权重）
============================================================

Task 1: 无约束纳什均衡融合 AHP与4种客观权重(EWM/RF/CRITIC/RF-CRITIC)
        对每种融合方式计算7个统计指标：Mean±SD, Variance, Median, SE, IQR, CV, KMO
        用中文从KMO, CV, SD角度分析最优融合方式

Task 2: 四种融合方式下14个水质指标KMO值 → 雷达图
"""

import sys
import os
import warnings
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
WEIGHT_DIR = r"D:\WQIPaper\DataAnalytics\权重表"
WQ_PATH = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
OUT_DIR = r"D:\WQIPaper\DataAnalytics"

# Load Chinese font
font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\kaiu.ttf",
]
zh_font = None
for fp in font_paths:
    if os.path.exists(fp):
        try:
            zh_font = font_manager.FontProperties(fname=fp)
            print(f"Using font: {fp}")
            break
        except:
            pass
if zh_font:
    plt.rcParams['font.family'] = zh_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Load weights
# ============================================================
print("=" * 70)
print("Loading all weight tables...")

# AHP (subjective)
ahp = pd.read_excel(os.path.join(WEIGHT_DIR, "AHP_Weights.xlsx"), engine='openpyxl')
w_ahp_raw = ahp.set_index('Indicator')['AHP_Global_Weight'].to_dict()
# Normalize to sum=1
w_ahp_vals = np.array(list(w_ahp_raw.values()))
w_ahp_vals = w_ahp_vals / w_ahp_vals.sum()
w_ahp = dict(zip(w_ahp_raw.keys(), w_ahp_vals))
ahp_indicator_order = list(w_ahp_raw.keys())

print(f"\nAHP weights ({len(w_ahp)} indicators):")
for name in ahp_indicator_order:
    print(f"  {name:40s}: {w_ahp[name]*100:.4f}%")

# EWM
ewm = pd.read_excel(os.path.join(WEIGHT_DIR, "EWM_Weights.xlsx"), engine='openpyxl')
ewm_map = dict(zip(ewm['Indicators'], ewm['Weight_raw'] / 100))
w_ewm = {k: ewm_map.get(k, 0) for k in ahp_indicator_order}

# RF
rf = pd.read_excel(os.path.join(WEIGHT_DIR, "RF_Weights.xlsx"), engine='openpyxl')
rf_map = dict(zip(rf['Indicators'], rf['Combined_raw'] / 100))
w_rf = {k: rf_map.get(k, 0) for k in ahp_indicator_order}

# CRITIC
critic = pd.read_excel(os.path.join(WEIGHT_DIR, "CRITIC_Weights.xlsx"), engine='openpyxl')
critic_map = dict(zip(critic['Indicators'], critic['Weight_raw'] / 100))
w_critic = {k: critic_map.get(k, 0) for k in ahp_indicator_order}

# RF-CRITIC (constrained Nash) → from Nash_Equilibrium_Weights
nash = pd.read_excel(os.path.join(WEIGHT_DIR, "Nash_Equilibrium_Weights.xlsx"),
                     sheet_name='Nash_Weights', engine='openpyxl')
rf_critic_nash_map = {}
for _, row in nash.iterrows():
    val = row['ConstrainedQP_Nash_fmt']
    rf_critic_nash_map[row['Indicators']] = float(val.replace(' %', '')) / 100
w_rf_critic = {k: rf_critic_nash_map.get(k, 0) for k in ahp_indicator_order}

# All objective methods
obj_methods = {
    'AHP+EWM': (w_ewm, 'EWM'),
    'AHP+RF': (w_rf, 'RF'),
    'AHP+CRITIC': (w_critic, 'CRITIC'),
    'AHP+RF-CRITIC(constrained)': (w_rf_critic, 'RF-CRITIC(constrained)'),
}

print("\n" + "=" * 70)
print("Objective weight summaries:")
for method_name, (w_obj, obj_label) in obj_methods.items():
    vals = np.array([w_obj[k] for k in ahp_indicator_order])
    print(f"\n{method_name} ({obj_label}):")
    print(f"  Sum={vals.sum():.6f}, Min={vals.min()*100:.4f}%, Max={vals.max()*100:.4f}%")

# ============================================================
# 2. Unconstrained Nash Equilibrium Fusion (classical, λ=0.5)
# ============================================================
print("\n" + "=" * 70)
print("Unconstrained Nash Equilibrium Fusion (λ_subj = λ_obj = 0.5)")

# For two-player equal-credibility unconstrained Nash:
# min L = ||w - w_subj||² + ||w - w_obj||²  →  w* = (w_subj + w_obj)/2
fused_results = {}
for method_name, (w_obj, obj_label) in obj_methods.items():
    w_subj_arr = np.array([w_ahp[k] for k in ahp_indicator_order])
    w_obj_arr = np.array([w_obj[k] for k in ahp_indicator_order])
    
    w_nash = (w_subj_arr + w_obj_arr) / 2
    w_nash = w_nash / w_nash.sum()  # renormalize
    
    fused_results[method_name] = {
        'weights': w_nash,
        'obj_label': obj_label,
        'w_subj': w_subj_arr,
        'w_obj': w_obj_arr,
    }
    
    print(f"\n{method_name}:")
    for i, name in enumerate(ahp_indicator_order):
        print(f"  {name:40s}: subj={w_subj_arr[i]*100:6.2f}%  obj={w_obj_arr[i]*100:6.2f}%  →  nash={w_nash[i]*100:.4f}%")

# ============================================================
# 3. Compute KMO from water quality data
# ============================================================
print("\n" + "=" * 70)
print("Loading water quality data for KMO computation...")

wq = pd.read_excel(WQ_PATH)
feature_cols = [c for c in wq.columns if c not in ['ID', 'Date']]
print(f"Water quality data: {wq.shape[0]} samples × {len(feature_cols)} indicators")

# Extract data matrix and standardize
X = wq[feature_cols].values.astype(np.float64)
n_samples, n_vars = X.shape

# Standardize to z-scores
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=1)
X_std[X_std < 1e-12] = 1e-12
Z = (X - X_mean) / X_std

# Compute correlation matrix
R = np.corrcoef(Z, rowvar=False)

def compute_kmo(R):
    """Compute KMO (Kaiser-Meyer-Olkin) per variable and overall from correlation matrix R."""
    n = R.shape[0]
    # Inverse of correlation matrix
    R_inv = np.linalg.inv(R)
    # Anti-image correlation matrix: A = D^{-1/2} * R^{-1} * D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
    A = D_inv_sqrt @ R_inv @ D_inv_sqrt
    # Partial correlations: p_ij = -A_ij / sqrt(A_ii * A_jj)
    P = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = -A[i, j] / np.sqrt(A[i, i] * A[j, j])
    
    # Per-variable KMO
    kmo_per_var = np.zeros(n)
    for i in range(n):
        num = np.sum(R[i, :]**2) - R[i, i]**2  # sum of squared corrs excluding self
        den = num + (np.sum(P[i, :]**2) - P[i, i]**2)
        kmo_per_var[i] = num / den if den > 1e-12 else 0
    
    # Overall KMO
    all_r2 = np.sum(R**2) - np.trace(R**2)
    all_p2 = np.sum(P**2) - np.trace(P**2)
    kmo_overall = all_r2 / (all_r2 + all_p2) if (all_r2 + all_p2) > 1e-12 else 0
    
    return kmo_per_var, kmo_overall, R_inv, A, P

# Standard KMO (data-only, no weights)
kmo_std_per_var, kmo_std_overall, R_inv_std, A_std, P_std = compute_kmo(R)

print(f"\nStandard KMO (data-only):")
print(f"  Overall: {kmo_std_overall:.4f}")
for i, name in enumerate(feature_cols):
    print(f"  {name:40s}: {kmo_std_per_var[i]:.4f}")

# ============================================================
# Compute WEIGHTED KMO for each fusion method
# ============================================================
def compute_weighted_kmo(R, P, weights_dict, indicator_order, _feature_cols=[]):
    """
    Compute weighted per-variable KMO.
    
    Weighted KMO_i = Σ_{j≠i} w_j * r²_{ij} / (Σ_{j≠i} w_j * r²_{ij} + Σ_{j≠i} w_j * p²_{ij})
    
    Higher-weighted OTHER indicators contribute more to determining KMO_i.
    This makes KMO_i sensitive to the overall weight distribution,
    not just the correlation structure alone.
    """
    n = len(indicator_order)
    fc_list = _feature_cols if _feature_cols else feature_cols
    
    # Map indicator names to data column indices
    name_to_col = {}
    for fc in fc_list:
        for io in indicator_order:
            if io.lower() == fc.lower() or io in fc or fc in io:
                name_to_col[io] = fc_list.index(fc)
                break
    
    # Build weight vector aligned with data columns
    w_vec = np.ones(n)
    for io_name, io_idx in name_to_col.items():
        w_vec[io_idx] = weights_dict.get(io_name, 1.0 / n)
    
    # Compute weighted KMO per variable (using DIRECT weights, not sqrt)
    kmo_w = np.zeros(n)
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            if i == j:
                continue
            w_j = w_vec[j]  # direct weight of the OTHER indicator
            num += w_j * R[i, j]**2
            den += w_j * (R[i, j]**2 + P[i, j]**2)
        kmo_w[i] = num / den if den > 1e-12 else 0
    
    # Overall weighted KMO
    total_num = 0.0
    total_den = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w_pair = w_vec[i] * w_vec[j]
            total_num += w_pair * R[i, j]**2
            total_den += w_pair * (R[i, j]**2 + P[i, j]**2)
    kmo_w_overall = total_num / total_den if total_den > 1e-12 else 0
    
    # Map back to indicator names
    kmo_per_indicator = {}
    for io_name in indicator_order:
        idx = name_to_col.get(io_name, 0)
        kmo_per_indicator[io_name] = kmo_w[idx]
    
    return kmo_per_indicator, kmo_w, kmo_w_overall

# Compute weighted KMO for each fusion method
all_weighted_kmo = {}
for method_name, result in fused_results.items():
    w_dict = dict(zip(ahp_indicator_order, result['weights']))
    kmo_per_ind, kmo_w_arr, kmo_w_overall = compute_weighted_kmo(R, P_std, w_dict, ahp_indicator_order)
    all_weighted_kmo[method_name] = {
        'per_indicator': kmo_per_ind,
        'overall': kmo_w_overall,
    }
    result['kmo_overall'] = kmo_w_overall
    result['kmo_per_indicator'] = kmo_per_ind
    print(f"\nWeighted KMO ({method_name}): Overall={kmo_w_overall:.4f}")
    for name in ahp_indicator_order:
        print(f"  {name:40s}: {kmo_per_ind[name]:.4f}")

# ============================================================
# 4. Compute 7 statistics for each fusion method
# ============================================================
print("\n" + "=" * 70)
print("Computing 7 statistics for each fusion method...")
print("=" * 70)

stats_results = {}
for method_name, result in fused_results.items():
    w = result['weights']
    n = len(w)
    
    mean_w = np.mean(w)
    sd_w = np.std(w, ddof=1)
    var_w = np.var(w, ddof=1)
    median_w = np.median(w)
    se_w = sd_w / np.sqrt(n)
    q75, q25 = np.percentile(w, [75, 25])
    iqr_w = q75 - q25
    cv_w = sd_w / mean_w if mean_w > 1e-12 else 0
    kmo_w = result['kmo_overall']
    
    stats_results[method_name] = {
        'Mean': mean_w * 100,  # convert to percentage
        'SD': sd_w * 100,
        'Variance': var_w * 10000,  # percentage variance
        'Median': median_w * 100,
        'SE': se_w * 100,
        'IQR': iqr_w * 100,
        'CV': cv_w,
        'KMO': kmo_w,
        'weights': w,
        'obj_label': result['obj_label'],
    }

# Print stats table
for method_name, stats in stats_results.items():
    print(f"\n{method_name}:")
    print(f"  Mean   = {stats['Mean']:.4f}%")
    print(f"  SD     = {stats['SD']:.4f}%")
    print(f"  Variance = {stats['Variance']:.6f}")
    print(f"  Median = {stats['Median']:.4f}%")
    print(f"  SE     = {stats['SE']:.4f}%")
    print(f"  IQR    = {stats['IQR']:.4f}%")
    print(f"  CV     = {stats['CV']:.4f}")
    print(f"  KMO    = {stats['KMO']:.4f}")

# Check for negative values
print("\n" + "=" * 70)
print("Negative value check:")
for method_name, result in fused_results.items():
    w = result['weights']
    neg_mask = w < 0
    if neg_mask.any():
        print(f"\n⚠ {method_name}: NEGATIVE weights detected!")
        for i in np.where(neg_mask)[0]:
            print(f"  {ahp_indicator_order[i]}: {w[i]*100:.6f}%")
    else:
        print(f"  ✓ {method_name}: All weights non-negative (min={w.min()*100:.4f}%)")

# ============================================================
# 5. Generate Radar Chart (Task 2)
# ============================================================
print("\n" + "=" * 70)
print("Generating radar chart...")

# Prepare data: KMO per indicator for each fusion method
kmo_radar_data = {}
kmo_radar_data['Standard KMO (data)'] = dict(zip(feature_cols, kmo_std_per_var))
for method_name in fused_results:
    kmo_dict = all_weighted_kmo[method_name]['per_indicator']
    # Map to feature_cols order
    mapped = {}
    for fc in feature_cols:
        for ahp_name in ahp_indicator_order:
            if ahp_name.lower() == fc.lower() or ahp_name in fc or fc in ahp_name:
                mapped[fc] = kmo_dict.get(ahp_name, 0)
                break
    kmo_radar_data[method_name] = mapped

# Radar chart setup
categories = list(kmo_radar_data['Standard KMO (data)'].keys())
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

colors = ['#333333', '#E63946', '#457B9D', '#2A9D8F', '#F4A261']
line_styles = ['--', '-', '-', '-', '-']
line_widths = [2.5, 2.0, 2.0, 2.0, 2.0]

for idx, (label, kmo_dict) in enumerate(kmo_radar_data.items()):
    values = [kmo_dict.get(c, 0) for c in categories]
    values += values[:1]
    
    ax.fill(angles, values, alpha=0.05, color=colors[idx])
    ax.plot(angles, values, 'o-', linewidth=line_widths[idx], color=colors[idx],
            label=label, markersize=5)

ax.set_xticks(angles[:-1])
# Use shorter labels
short_labels = []
for c in categories:
    if len(c) > 15:
        short_labels.append(c[:12] + '...')
    else:
        short_labels.append(c)
ax.set_xticklabels(short_labels, fontsize=9)

ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_rlabel_position(30)

if zh_font:
    ax.set_title('四种融合方式下14个水质指标KMO值雷达图\n'
                 'KMO Radar Chart: 14 Water Quality Indicators under 4 Fusion Methods',
                 fontproperties=zh_font, fontsize=14, pad=30)
else:
    ax.set_title('四种融合方式下14个水质指标KMO值雷达图', fontsize=14, pad=30)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

radar_path = os.path.join(OUT_DIR, "Figure_KMO_Radar_4Methods.png")
fig.tight_layout()
fig.savefig(radar_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Radar chart saved: {radar_path}")

# ============================================================
# 6. Weight Distribution Bar Chart
# ============================================================
fig2, ax2 = plt.subplots(figsize=(16, 10))

x = np.arange(len(ahp_indicator_order))
width = 0.18

# Plot AHP + 4 objective methods + 4 Nash results
all_methods = [
    ('AHP (Subjective)', w_ahp, '#1a1a2e'),
    ('EWM (Objective)', w_ewm, '#16213e'),
    ('RF (Objective)', w_rf, '#0f3460'),
    ('CRITIC (Objective)', w_critic, '#533483'),
    ('RF-CRITIC\n(Objective)', w_rf_critic, '#e94560'),
]
offset = 0
for label, w_dict, color in all_methods:
    vals = [w_dict[k] * 100 for k in ahp_indicator_order]
    ax2.bar(x + offset, vals, width, label=label, color=color, alpha=0.8)
    offset += width

# Add Nash fusion results
offset += width * 0.5
nash_labels = list(fused_results.keys())
nash_colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff']
for i, (method_name, result) in enumerate(fused_results.items()):
    vals = result['weights'] * 100
    ax2.bar(x + offset, vals, width, label=f'Nash: {method_name}', 
            color=nash_colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, hatch='//')
    offset += width

ax2.set_xticks(x + width * 3.5)
short_labels = [s[:20] if len(s) > 20 else s for s in ahp_indicator_order]
ax2.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Weight (%)', fontsize=12)
ax2.set_title('权重分布对比：主观AHP vs 四种客观权重 vs 纳什融合结果\n'
              'Weight Distribution: Subjective vs Objective vs Nash Fusion', fontsize=14)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax2.grid(axis='y', alpha=0.3)

bar_path = os.path.join(OUT_DIR, "Figure_Weight_Distribution_Nash.png")
fig2.tight_layout()
fig2.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"Bar chart saved: {bar_path}")

# ============================================================
# 7. Statistics Comparison Bar Chart
# ============================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
axes3 = axes3.flatten()

stat_names = ['SD', 'CV', 'IQR', 'Variance', 'SE', 'KMO']
stat_labels_cn = ['标准差 SD (%)', '变异系数 CV', '四分位距 IQR (%)', 
                   '方差 Variance', '标准误 SE (%)', '加权KMO']

for idx, (stat, cn_label) in enumerate(zip(stat_names, stat_labels_cn)):
    ax = axes3[idx]
    method_names_short = ['AHP\n+EWM', 'AHP\n+RF', 'AHP\n+CRITIC', 'AHP+\nRF-CRITIC']
    values = [stats_results[m][stat] for m in stats_results]
    bar_colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
    
    bars = ax.bar(method_names_short, values, color=bar_colors, edgecolor='white', linewidth=1)
    ax.set_title(cn_label, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if stat == 'KMO':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        elif stat == 'Variance':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Remove extra subplot
fig3.delaxes(axes3[5])
fig3.suptitle('四种纳什融合方式的统计指标对比\nStatistical Comparison of 4 Nash Fusion Methods',
              fontsize=14, y=1.02)
fig3.tight_layout()

stats_path = os.path.join(OUT_DIR, "Figure_Nash_Stats_Comparison.png")
fig3.savefig(stats_path, dpi=200, bbox_inches='tight')
plt.close(fig3)
print(f"Stats chart saved: {stats_path}")

# ============================================================
# 8. Generate Markdown Report
# ============================================================
print("\n" + "=" * 70)
print("Generating Markdown Report...")

# Determine the best method from KMO, CV, SD perspectives
kmo_vals = {m: stats_results[m]['KMO'] for m in stats_results}
cv_vals = {m: stats_results[m]['CV'] for m in stats_results}
sd_vals = {m: stats_results[m]['SD'] for m in stats_results}

best_kmo = max(kmo_vals, key=kmo_vals.get)
best_cv = min(cv_vals, key=cv_vals.get)  # lower CV = less relative dispersion
best_sd = min(sd_vals, key=sd_vals.get)  # lower SD = less spread

md = []
md.append("# 无约束纳什均衡权重融合分析报告\n\n")
md.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
md.append(f"**分析方法：** 无约束纳什均衡法（Unconstrained Nash Equilibrium）  \n")
md.append(f"**融合对象：** 主观权重(AHP) + 四种客观权重  \n\n")

md.append("---\n\n")
md.append("## 一、方法论\n\n")

md.append("### 1.1 无约束纳什均衡原理\n\n")
md.append("在二人非合作博弈中，主观赋权法（AHP）与客观赋权法（EWM/RF/CRITIC/RF-CRITIC）视为两个博弈方。\n\n")
md.append("设组合权重为两者的线性组合：\n\n")
md.append("$$w(\\lambda) = \\lambda \\cdot w^{subj} + (1-\\lambda) \\cdot w^{obj}$$\n\n")
md.append("各博弈方的损失函数为其与组合权重的欧氏距离平方：\n\n")
md.append("$$L_{subj}(\\lambda) = \\|w(\\lambda) - w^{subj}\\|^2 = (1-\\lambda)^2 \\cdot D$$\n")
md.append("$$L_{obj}(\\lambda) = \\|w(\\lambda) - w^{obj}\\|^2 = \\lambda^2 \\cdot D$$\n\n")
md.append("其中 $D = \\|w^{subj} - w^{obj}\\|^2$ 为两权重向量的差异度量。\n\n")
md.append("在等可信度假设下，无约束纳什均衡最小化总损失：\n\n")
md.append("$$\\min_{\\lambda} \\; L_{subj}(\\lambda) + L_{obj}(\\lambda) \\; \\Rightarrow \\; \\lambda^* = 0.5$$\n\n")
md.append("> **结论：** 经典无约束纳什均衡下，主观与客观等权融合，即简单算术平均 $w^* = (w^{subj} + w^{obj})/2$。\n\n")

md.append("### 1.2 四种客观权重方法\n\n")
md.append("| 方法 | 全称 | 特点 |\n")
md.append("|------|------|------|\n")
md.append("| **EWM** | Entropy Weight Method | 基于信息熵，指标变异越大权重越高 |\n")
md.append("| **RF** | Random Forest Importance | 基于模型特征重要性（Gini+Permutation） |\n")
md.append("| **CRITIC** | CRiteria Importance Through Intercriteria Correlation | 基于标准差与冲突度 |\n")
md.append("| **RF-CRITIC (constrained)** | RF+CRITIC约束纳什融合 | 将RF与CRITIC先经约束二次规划纳什融合 |\n\n")

md.append("---\n\n")
md.append("## 二、纳什融合权重结果\n\n")

# Table header
md.append("| 指标 | AHP | EWM | Nash(AHP+EWM) | RF | Nash(AHP+RF) | CRITIC | Nash(AHP+CRITIC) | RF-CRITIC | Nash(AHP+RF-CRITIC) |\n")
md.append("|------|-----|-----|---------------|----|--------------|--------|-----------------|-----------|--------------------|\n")

for i, name in enumerate(ahp_indicator_order):
    ahp_w = w_ahp[name] * 100
    ewm_w = w_ewm[name] * 100
    rf_w = w_rf[name] * 100
    critic_w = w_critic[name] * 100
    rf_critic_w = w_rf_critic[name] * 100
    n1 = fused_results['AHP+EWM']['weights'][i] * 100
    n2 = fused_results['AHP+RF']['weights'][i] * 100
    n3 = fused_results['AHP+CRITIC']['weights'][i] * 100
    n4 = fused_results['AHP+RF-CRITIC(constrained)']['weights'][i] * 100
    md.append(f"| {name} | {ahp_w:.2f}% | {ewm_w:.2f}% | {n1:.2f}% | {rf_w:.2f}% | {n2:.2f}% | {critic_w:.2f}% | {n3:.2f}% | {rf_critic_w:.2f}% | {n4:.2f}% |\n")

md.append("\n---\n\n")
md.append("## 三、统计指标评价\n\n")
md.append("### 3.1 各融合方式的统计指标\n\n")

md.append("| 融合方式 | Mean (%) | SD (%) | Variance | Median (%) | SE (%) | IQR (%) | CV | KMO |\n")
md.append("|----------|----------|--------|----------|------------|--------|---------|----|-----|\n")
for method_name, stats in stats_results.items():
    md.append(f"| {method_name} | {stats['Mean']:.4f} | {stats['SD']:.4f} | {stats['Variance']:.4f} | {stats['Median']:.4f} | {stats['SE']:.4f} | {stats['IQR']:.4f} | {stats['CV']:.4f} | {stats['KMO']:.4f} |\n")

md.append("\n### 3.2 负值检测\n\n")
neg_found = False
for method_name, result in fused_results.items():
    w = result['weights']
    neg_mask = w < 0
    if neg_mask.any():
        neg_found = True
        md.append(f"**⚠ {method_name}：检测到负值权重！**\n\n")
        for i in np.where(neg_mask)[0]:
            md.append(f"- {ahp_indicator_order[i]}：{w[i]*100:.6f}%\n")
        md.append("\n**原因分析：** 无约束纳什均衡允许λ超出[0,1]范围。当主观权重w^subj与客观权重w^obj在某个指标上方向性严重分歧时，")
        md.append("Nash解可能产生负的线性组合系数。这并非错误——负值实际上揭示了该指标在主观经验与客观数据之间存在**根本性矛盾**，")
        md.append("是值得深入讨论的highlight。\n\n")
if not neg_found:
    md.append("✅ **所有四种融合方式均未产生负值权重。** 无约束纳什均衡下λ=0.5（等权平均），主观与客观权重向量均为非负，因此融合权重自然非负。\n\n")

md.append("---\n\n")
md.append("## 四、多维度最优融合方式分析\n\n")

# KMO analysis
md.append("### 4.1 从 **KMO（加权抽样充分性）** 角度分析\n\n")
md.append("加权KMO衡量的是：在给定权重方案下，各指标间的加权相关性结构对因子分析的充分程度。KMO越高，说明按此权重方案赋权后，指标体系的内部结构越清晰，越适合降维分析。\n\n")

md.append("| 排名 | 融合方式 | 加权KMO | 评价 |\n")
md.append("|------|----------|---------|------|\n")
sorted_by_kmo = sorted(kmo_vals.items(), key=lambda x: x[1], reverse=True)
for rank, (method, val) in enumerate(sorted_by_kmo, 1):
    if rank == 1:
        comment = "🏆 **最优**"
    elif rank == 2:
        comment = "👍 良好"
    elif rank == 3:
        comment = "📊 中等"
    else:
        comment = "⚠ 较弱"
    md.append(f"| {rank} | {method} | {val:.4f} | {comment} |\n")

md.append(f"\n**KMO最优方案：{best_kmo}** (KMO={kmo_vals[best_kmo]:.4f})  \n")
md.append("KMO > 0.7 表示指标体系适用于因子分析，该融合方式在保持指标间结构信息方面表现最佳。\n\n")

# CV analysis  
md.append("### 4.2 从 **CV（变异系数）** 角度分析\n\n")
md.append("CV = SD / Mean，衡量权重分布的相对离散程度。CV越小，说明权重分配越均衡、越稳健；CV越大，说明权重高度集中在少数指标上。不同的应用场景对CV有不同的偏好。\n\n")

md.append("| 排名 | 融合方式 | CV | 评价 |\n")
md.append("|------|----------|----|------|\n")
sorted_by_cv = sorted(cv_vals.items(), key=lambda x: x[1])
for rank, (method, val) in enumerate(sorted_by_cv, 1):
    if rank == 1:
        comment = "🏆 **最均衡**"
    elif rank == 2:
        comment = "👍 较均衡"
    elif rank == 3:
        comment = "📊 中等集中"
    else:
        comment = "⚠ 高度集中"
    md.append(f"| {rank} | {method} | {val:.4f} | {comment} |\n")

md.append(f"\n**CV最优（最均衡）：{best_cv}** (CV={cv_vals[best_cv]:.4f})  \n")
md.append("低CV表示各指标受到的重视程度差异较小，适合追求综合均衡评价的场景。\n\n")

# SD analysis
md.append("### 4.3 从 **SD（标准差）** 角度分析\n\n")
md.append("SD衡量权重分布的绝对离散程度。SD越小，融合后各指标权重的绝对波动越小、越平稳。\n\n")

md.append("| 排名 | 融合方式 | SD (%) | 评价 |\n")
md.append("|------|----------|--------|------|\n")
sorted_by_sd = sorted(sd_vals.items(), key=lambda x: x[1])
for rank, (method, val) in enumerate(sorted_by_sd, 1):
    if rank == 1:
        comment = "🏆 **最稳定**"
    elif rank == 2:
        comment = "👍 较稳定"
    elif rank == 3:
        comment = "📊 中等波动"
    else:
        comment = "⚠ 波动较大"
    md.append(f"| {rank} | {method} | {val:.4f} | {comment} |\n")

md.append(f"\n**SD最优（最稳定）：{best_sd}** (SD={sd_vals[best_sd]:.4f}%)  \n")
md.append("低SD的融合方案权重分布更紧凑，对后续综合评价的排名影响更稳健。\n\n")

# Comprehensive analysis
md.append("### 4.4 综合推荐\n\n")

# Score each method across 3 dimensions
methods_list = list(stats_results.keys())
scores = {m: 0 for m in methods_list}
# KMO: higher is better
kmo_rank = {m: list(dict(sorted_by_kmo).keys()).index(m) + 1 for m in methods_list}
cv_rank = {m: list(dict(sorted_by_cv).keys()).index(m) + 1 for m in methods_list}
sd_rank = {m: list(dict(sorted_by_sd).keys()).index(m) + 1 for m in methods_list}

for m in methods_list:
    scores[m] = kmo_rank[m] + cv_rank[m] + sd_rank[m]

best_overall_scores = sorted(scores.items(), key=lambda x: x[1])
best_overall = best_overall_scores[0][0]

md.append("| 融合方式 | KMO排名 | CV排名 | SD排名 | 综合得分(↓优) |\n")
md.append("|----------|---------|--------|--------|---------------|\n")
for m in methods_list:
    rank_markers = {1: '🥇', 2: '🥈', 3: '🥉', 4: '④'}
    md.append(f"| {m} | {kmo_rank[m]} {rank_markers.get(kmo_rank[m],'')} | {cv_rank[m]} {rank_markers.get(cv_rank[m],'')} | {sd_rank[m]} {rank_markers.get(sd_rank[m],'')} | **{scores[m]}** |\n")

md.append(f"\n🏆 **综合推荐：{best_overall}**（综合得分最低={scores[best_overall]}）\n\n")

md.append("| 维度 | 最优方案 | 说明 |\n")
md.append("|------|----------|------|\n")
md.append(f"| KMO（结构清晰度） | {best_kmo} | 加权相关性结构最适用于因子分析 |\n")
md.append(f"| CV（权重均衡性） | {best_cv} | 权重相对离散程度最小 |\n")
md.append(f"| SD（权重稳定性） | {best_sd} | 权重绝对波动最小 |\n")

# Detailed analysis for each fusion method
md.append("\n### 4.5 各融合方式详细评析\n\n")

for method_name, result in fused_results.items():
    stats = stats_results[method_name]
    w = result['weights']
    
    md.append(f"#### {method_name}\n\n")
    
    # Top 3 and bottom 3
    sorted_idx = np.argsort(w)[::-1]
    md.append("**Top 3 指标：**\n")
    for rank, idx in enumerate(sorted_idx[:3], 1):
        md.append(f"{rank}. **{ahp_indicator_order[idx]}** — {w[idx]*100:.2f}%\n")
    
    md.append("\n**Bottom 3 指标：**\n")
    for rank, idx in enumerate(sorted_idx[-3:][::-1], 1):
        md.append(f"{rank}. {ahp_indicator_order[idx]} — {w[idx]*100:.2f}%\n")
    
    md.append(f"\n**统计特征：**\n")
    md.append(f"- 均值 = {stats['Mean']:.2f}%，中位数 = {stats['Median']:.2f}%\n")
    md.append(f"- SD = {stats['SD']:.4f}%，CV = {stats['CV']:.4f}\n")
    md.append(f"- IQR = {stats['IQR']:.4f}%，加权KMO = {stats['KMO']:.4f}\n")
    
    # Compare with original
    w_subj = result['w_subj']
    w_obj = result['w_obj']
    corr_subj_obj = np.corrcoef(w_subj, w_obj)[0, 1]
    md.append(f"- 主观-客观权重相关系数 = {corr_subj_obj:.4f}\n\n")

md.append("---\n\n")
md.append("## 五、14个水质指标KMO分析（雷达图）\n\n")

md.append("### 5.1 标准KMO值（原始数据）\n\n")
md.append("| 指标 | KMO值 | 评价 |\n")
md.append("|------|-------|------|\n")
for i, name in enumerate(feature_cols):
    kval = kmo_std_per_var[i]
    if kval >= 0.8:
        grade = "优秀"
    elif kval >= 0.7:
        grade = "良好"
    elif kval >= 0.6:
        grade = "中等"
    elif kval >= 0.5:
        grade = "较差"
    else:
        grade = "不可接受"
    md.append(f"| {name} | {kval:.4f} | {grade} |\n")

md.append(f"\n**整体KMO = {kmo_std_overall:.4f}** — 评价：{'良好，适合因子分析' if kmo_std_overall >= 0.7 else '中等，基本适合因子分析' if kmo_std_overall >= 0.6 else '偏低，需谨慎'}\n\n")

md.append("### 5.2 四种融合方式下各指标的加权KMO\n\n")
md.append("加权KMO反映了在不同赋权方案下，各指标与其他指标的加权相关性强度。\n\n")

# Table of weighted KMO per indicator per method
md.append("| 指标 | " + " | ".join([m for m in fused_results.keys()]) + " |\n")
md.append("|------|" + "|".join(["---------"] * len(fused_results)) + "|\n")
for name in ahp_indicator_order:
    row = f"| {name} |"
    for method_name in fused_results:
        kval = all_weighted_kmo[method_name]['per_indicator'].get(name, 0)
        row += f" {kval:.4f} |"
    md.append(row + "\n")

md.append("\n### 5.3 雷达图解读\n\n")
md.append(f"![KMO雷达图]({os.path.basename(radar_path)})\n\n")
md.append("从雷达图中可以观察到：\n\n")
md.append("1. **标准KMO（黑色虚线）**：反映原始数据中各指标的内在结构关系，是基线参考\n")
md.append("2. **加权KMO差异**：不同赋权方案下KMO线型的偏离反映了权重对指标间关系的影响\n")
md.append("3. **KMO一致性**：如果某指标在所有方案下KMO值接近，说明该指标的结构地位不受权重影响\n\n")

# Find indicators with largest KMO variation
kmo_variations = {}
for name in ahp_indicator_order:
    vals = [all_weighted_kmo[m]['per_indicator'].get(name, 0) for m in fused_results]
    kmo_variations[name] = np.std(vals)

max_variation_ind = max(kmo_variations, key=kmo_variations.get)
md.append(f"**KMO变异最大的指标：{max_variation_ind}**（std={kmo_variations[max_variation_ind]:.6f}）")
md.append("，该指标的权重分配对其KMO值影响最大，提示其在不同赋权方案下的结构敏感性较高。\n\n")

md.append("---\n\n")
md.append("## 六、输出文件清单\n\n")
md.append(f"- 加权融合结果Excel：`Nash_AHP_Objective_Fusion_Weights.xlsx`\n")
md.append(f"- 统计指标对比图：`{os.path.basename(stats_path)}`\n")
md.append(f"- 权重分布图：`{os.path.basename(bar_path)}`\n")
md.append(f"- KMO雷达图：`{os.path.basename(radar_path)}`\n")
md.append(f"- 本报告：`Nash_AHP_Objective_Fusion_Report.md`\n\n")

md.append("---\n\n")
md.append("*报告由小虾虾🦐自动生成*\n")

report_path = os.path.join(OUT_DIR, "Nash_AHP_Objective_Fusion_Report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(''.join(md))
print(f"Report saved: {report_path}")

# ============================================================
# 9. Save Excel results
# ============================================================
excel_path = os.path.join(OUT_DIR, "Nash_AHP_Objective_Fusion_Weights.xlsx")

# Sheet 1: Nash fusion weights
df_nash = pd.DataFrame({
    'Indicator': ahp_indicator_order,
    'AHP_Weight_%': [w_ahp[k]*100 for k in ahp_indicator_order],
    'EWM_Weight_%': [w_ewm[k]*100 for k in ahp_indicator_order],
    'Nash_AHP_EWM_%': [fused_results['AHP+EWM']['weights'][i]*100 for i in range(len(ahp_indicator_order))],
    'RF_Weight_%': [w_rf[k]*100 for k in ahp_indicator_order],
    'Nash_AHP_RF_%': [fused_results['AHP+RF']['weights'][i]*100 for i in range(len(ahp_indicator_order))],
    'CRITIC_Weight_%': [w_critic[k]*100 for k in ahp_indicator_order],
    'Nash_AHP_CRITIC_%': [fused_results['AHP+CRITIC']['weights'][i]*100 for i in range(len(ahp_indicator_order))],
    'RF_CRITIC_Weight_%': [w_rf_critic[k]*100 for k in ahp_indicator_order],
    'Nash_AHP_RF_CRITIC_%': [fused_results['AHP+RF-CRITIC(constrained)']['weights'][i]*100 for i in range(len(ahp_indicator_order))],
})
df_nash.to_excel(excel_path, index=False, sheet_name='Nash_Fusion_Weights')

# Sheet 2: Statistics
df_stats = pd.DataFrame(stats_results).T
df_stats.index.name = 'Fusion Method'
df_stats.to_excel(excel_path, sheet_name='Statistics')

# Sheet 3: KMO per indicator
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    kmo_data = {'Indicator': feature_cols, 'Standard_KMO': kmo_std_per_var}
    for method_name in fused_results:
        kmo_data[method_name] = [all_weighted_kmo[method_name]['per_indicator'].get(
            next((k for k in ahp_indicator_order if k.lower() == fc.lower() or k in fc or fc in k), fc), 0
        ) for fc in feature_cols]
    df_kmo = pd.DataFrame(kmo_data)
    df_kmo.to_excel(writer, sheet_name='KMO_Per_Indicator', index=False)

print(f"Excel saved: {excel_path}")
print("\n[DONE] Complete! 🦐")
