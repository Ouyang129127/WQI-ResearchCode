# -*- coding: utf-8 -*-
"""
==============================================================================
水体指标权重融合 — 无约束纳什均衡（AHP主观 + 4种客观权重）【画图模块 v2】
==============================================================================

依赖：先运行 nash_ahp_fusion_calc.py 生成 Excel 结果文件

图(a) — 雷达图：四种融合方式 × 14个指标的融合权重
图(b) — 雷达图：四种融合方式 × 7个统计指标 (SD/Variance/Median/SE/IQR/CV/KMO)
图(c) — 柱形图：AHP+RF-CRITIC 方式下，14指标 × (AHP权重 + RF-CRITIC权重 + 融合权重) 并列

SCI 规范：Arial 字体 / 莫兰迪配色 / cm 尺寸 / PNG+TIFF 双导出
计算模块：nash_ahp_fusion_calc.py
"""

import sys
import os
import warnings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════╗
# ║              1. 全局尺寸 & 单位换算                       ║
# ╚══════════════════════════════════════════════════════════╝
CM2INCH = 1 / 2.54

FIG_WIDTH_CM  = 19     # 整页宽度 (cm)
FIG_HEIGHT_CM = 23     # 总高度 (cm)：上排雷达 ~9cm + 下排柱形 ~14cm

# ╔══════════════════════════════════════════════════════════╗
# ║              2. SCI 字体 & 线宽 (全局 rcParams)           ║
# ╚══════════════════════════════════════════════════════════╝
rcParams['font.family'] = 'Arial'
rcParams['font.size']      = 9
rcParams['axes.titlesize']    = 10
rcParams['axes.labelsize']    = 9
rcParams['xtick.labelsize']   = 7
rcParams['ytick.labelsize']   = 7
rcParams['legend.fontsize']   = 8
rcParams['legend.title_fontsize'] = 9

rcParams['axes.linewidth']     = 1.0
rcParams['xtick.major.width']  = 0.8
rcParams['ytick.major.width']  = 0.8
rcParams['xtick.major.size']   = 3
rcParams['ytick.major.size']   = 3
rcParams['xtick.minor.size']   = 1.5
rcParams['ytick.minor.size']   = 1.5

rcParams['figure.dpi']         = 300
rcParams['savefig.dpi']        = 500
rcParams['savefig.bbox']       = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# ╔══════════════════════════════════════════════════════════╗
# ║              3. 莫兰迪配色                                ║
# ╚══════════════════════════════════════════════════════════╝
MORANDI = [
    "#C49A8C",  # 陶土玫瑰
    "#9CAF88",  # 鼠尾草绿
    "#8FA7B7",  # 灰蓝
    "#B8A99A",  # 暖灰
    "#A88C9A",  # 淡紫灰
    "#B0A878",  # 橄榄黄绿
    "#7E8DA2",  # 石板蓝
    "#C1876B",  # 陶土棕
]

def morandi(idx):
    """循环取莫兰迪色"""
    return MORANDI[idx % len(MORANDI)]

# ╔══════════════════════════════════════════════════════════╗
# ║              4. 统一导出函数                              ║
# ╚══════════════════════════════════════════════════════════╝
def save_fig(fig, name, dpi=500):
    """同时输出 PNG (预览) + TIFF LZW (投稿)"""
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{name}.tif", dpi=dpi, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"  Saved: {os.path.basename(name)}.png / .tif")

# ╔══════════════════════════════════════════════════════════╗
# ║              5. Paths & 数据加载                          ║
# ╚══════════════════════════════════════════════════════════╝
OUT_DIR = r"D:\WQIPaper\DataAnalytics"
EXCEL_PATH = os.path.join(OUT_DIR, "Nash_AHP_Objective_Fusion_Weights.xlsx")

if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(
        f"找不到 {EXCEL_PATH}\n"
        f"请先运行 nash_ahp_fusion_calc.py 生成计算结果！"
    )

print("=" * 70)
print("Loading results from Excel...")

# ── Sheet: Fusion_Weights_14Indicators ──
df_fw = pd.read_excel(EXCEL_PATH, sheet_name='Fusion_Weights_14Indicators', engine='openpyxl')
ahp_indicator_order = df_fw['Indicator'].tolist()
N_IND = len(ahp_indicator_order)  # 14

fusion_method_names = [
    'AHP+EWM',
    'AHP+RF',
    'AHP+CRITIC',
    'AHP+RF-CRITIC(constrained)',
]

# 提取四种融合权重 (原始 0~1)
fusion_weights_dict = {}
for method in fusion_method_names:
    col = f'{method}_Weight'
    fusion_weights_dict[method] = df_fw[col].values

# ── Sheet: Nash_Fusion_Weights (获取 AHP/RF-CRITIC 原始权重) ──
df_nash = pd.read_excel(EXCEL_PATH, sheet_name='Nash_Fusion_Weights', engine='openpyxl')
w_ahp_arr = df_nash['AHP_Weight_%'].values / 100
w_rf_critic_arr = df_nash['RF_CRITIC_Weight_%'].values / 100

# ── Sheet: Statistics ──
df_stats = pd.read_excel(EXCEL_PATH, sheet_name='Statistics', engine='openpyxl', index_col=0)
stat_methods = df_stats.index.tolist()
stat_names = ['SD', 'Variance', 'Median', 'SE', 'IQR', 'CV', 'KMO']
stat_labels = ['SD (%)', 'Variance', 'Median (%)', 'SE (%)', 'IQR (%)', 'CV', 'KMO']

# 提取统计值矩阵 (4 methods × 7 stats)
stats_matrix = np.zeros((len(stat_methods), len(stat_names)))
for i, method in enumerate(stat_methods):
    for j, sn in enumerate(stat_names):
        stats_matrix[i, j] = df_stats.loc[method, sn]

# ╔══════════════════════════════════════════════════════════╗
# ║              6. 创建画布 & GridSpec 布局                  ║
# ╚══════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(FIG_WIDTH_CM * CM2INCH, FIG_HEIGHT_CM * CM2INCH))

gs = GridSpec(2, 2, figure=fig,
              height_ratios=[0.42, 0.58],   # 上排雷达 42% / 下排柱形 58%
              hspace=0.35,                   # 垂直间距
              wspace=0.35)                   # 水平间距

ax_a = fig.add_subplot(gs[0, 0], polar=True)   # 图(a) 权重雷达
ax_b = fig.add_subplot(gs[0, 1], polar=True)   # 图(b) 统计雷达
ax_c = fig.add_subplot(gs[1, :])               # 图(c) 柱形图

# ╔══════════════════════════════════════════════════════════╗
# ║              7. 图(a)：权重雷达图                          ║
# ╚══════════════════════════════════════════════════════════╝
print("Plot (a): Weight Radar Chart...")

# ── 指标简写 (雷达轴标签) ──
short_names = [
    'THMs', 'Pb', 'Free Cl', 'NO₃⁻', 'TOC',
    'F⁻', 'COD_Mn', 'Al', 'TDS', 'pH',
    'Hardness', 'SO₄²⁻', 'Temp', 'Cl⁻',
]

N = N_IND
angles_a = [n / float(N) * 2 * np.pi for n in range(N)]
angles_a += angles_a[:1]  # close

# 确定 y 轴范围
all_weights_concat = np.concatenate([fusion_weights_dict[m] for m in fusion_method_names])
max_w = all_weights_concat.max()
ylim_a = np.ceil(max_w * 100) / 100 + 0.02  # 向上取整留余量

for idx, method in enumerate(fusion_method_names):
    values = list(fusion_weights_dict[method]) + [fusion_weights_dict[method][0]]
    color = morandi(idx)
    ax_a.fill(angles_a, values, alpha=0.08, color=color)
    ax_a.plot(angles_a, values, 'o-',
              linewidth=1.2, color=color,
              markersize=3.5, markerfacecolor='white',
              markeredgewidth=1.0, markeredgecolor=color,
              label=method.replace('(constrained)', '\n(constrained)'))

ax_a.set_xticks(angles_a[:-1])
ax_a.set_xticklabels(short_names, fontsize=7)
ax_a.set_ylim(0, ylim_a)
ax_a.set_yticks(np.round(np.linspace(0, ylim_a, 5), 2))
ax_a.set_yticklabels([f'{v:.2f}' for v in np.linspace(0, ylim_a, 5)], fontsize=6)
ax_a.set_rlabel_position(30)
ax_a.grid(True, alpha=0.3, linewidth=0.4)

# —— 子图标签 (a) ——
ax_a.text(-0.12, 1.08, '(a)', transform=ax_a.transAxes,
          fontsize=12, fontweight='bold', va='bottom', ha='left')

# —— 图例 ——
legend_a = ax_a.legend(loc='upper right', bbox_to_anchor=(1.32, 1.05),
                        fontsize=7, frameon=True, fancybox=False,
                        edgecolor='#999999', facecolor='white',
                        framealpha=0.9, ncol=1)
legend_a.set_title('Fusion Method', prop={'size': 8})

ax_a.set_title('Nash Fusion Weights — 14 Indicators',
               fontsize=10, fontweight='bold', pad=18)

# ╔══════════════════════════════════════════════════════════╗
# ║              8. 图(b)：统计指标雷达图                      ║
# ╚══════════════════════════════════════════════════════════╝
print("Plot (b): Statistics Radar Chart...")

# ── Min-Max 归一化到 [0, 1] ──
stats_norm = np.zeros_like(stats_matrix)
for j in range(len(stat_names)):
    col_min = stats_matrix[:, j].min()
    col_max = stats_matrix[:, j].max()
    if col_max - col_min < 1e-12:
        stats_norm[:, j] = 0.5
    else:
        stats_norm[:, j] = (stats_matrix[:, j] - col_min) / (col_max - col_min)

N_stats = len(stat_names)
angles_b = [n / float(N_stats) * 2 * np.pi for n in range(N_stats)]
angles_b += angles_b[:1]

for idx, method in enumerate(stat_methods):
    values = list(stats_norm[idx]) + [stats_norm[idx][0]]
    color = morandi(idx)
    ax_b.fill(angles_b, values, alpha=0.08, color=color)
    ax_b.plot(angles_b, values, 'o-',
              linewidth=1.2, color=color,
              markersize=3.5, markerfacecolor='white',
              markeredgewidth=1.0, markeredgecolor=color,
              label=method.replace('(constrained)', '\n(constrained)'))

ax_b.set_xticks(angles_b[:-1])
ax_b.set_xticklabels(stat_labels, fontsize=7)
ax_b.set_ylim(0, 1.05)
ax_b.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax_b.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'], fontsize=6)
ax_b.set_rlabel_position(30)
ax_b.grid(True, alpha=0.3, linewidth=0.4)

# —— 子图标签 (b) ——
ax_b.text(-0.12, 1.08, '(b)', transform=ax_b.transAxes,
          fontsize=12, fontweight='bold', va='bottom', ha='left')

# —— 图例 ——
legend_b = ax_b.legend(loc='upper right', bbox_to_anchor=(1.32, 1.05),
                        fontsize=7, frameon=True, fancybox=False,
                        edgecolor='#999999', facecolor='white',
                        framealpha=0.9, ncol=1)
legend_b.set_title('Fusion Method', prop={'size': 8})

ax_b.set_title('Statistical Indicators — 7 Metrics (normalized)',
               fontsize=10, fontweight='bold', pad=18)

# —— 在雷达图旁注释实际数值范围 ——
stat_ranges_text = []
for j, sn in enumerate(stat_names):
    lo = stats_matrix[:, j].min()
    hi = stats_matrix[:, j].max()
    stat_ranges_text.append(f"{sn}: [{lo:.3f}, {hi:.3f}]")
range_annotation = "\n".join(stat_ranges_text)
ax_b.text(1.45, 0.5, range_annotation, transform=ax_b.transAxes,
          fontsize=5.5, va='center', ha='left',
          family='monospace',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                    edgecolor='#cccccc', alpha=0.8))

# ╔══════════════════════════════════════════════════════════╗
# ║              9. 图(c)：AHP+RF-CRITIC 三柱并列图           ║
# ╚══════════════════════════════════════════════════════════╝
print("Plot (c): AHP+RF-CRITIC Weight Distribution Bar Chart...")

# 只取 AHP+RF-CRITIC(constrained) 的融合权重
w_nash_rf_critic = fusion_weights_dict['AHP+RF-CRITIC(constrained)']

# 准备数据：每指标三个柱子 [AHP, RF-CRITIC, Nash Fusion]
group_labels = short_names  # 使用简写

x = np.arange(N_IND)
width = 0.25
gap = 0.02  # 同组内柱子间隙

bar_colors_c = [morandi(0), morandi(6), morandi(7)]  # 陶土玫瑰 / 石板蓝 / 陶土棕
bar_labels_c = ['AHP (Subjective)', 'RF-CRITIC (Objective)', 'Nash Fusion']
bar_data = [w_ahp_arr, w_rf_critic_arr, w_nash_rf_critic]

for i, (data, color, label) in enumerate(zip(bar_data, bar_colors_c, bar_labels_c)):
    offset = (i - 1) * (width + gap)
    bars = ax_c.bar(x + offset, data * 100, width,       # *100 → 百分比
                    color=color, alpha=0.85,
                    edgecolor='white', linewidth=0.3,
                    label=label)
    # 数值标注 (仅 > 2% 的标注，避免拥挤)
    for j, (bar, val) in enumerate(zip(bars, data * 100)):
        if val > 2.0:
            ax_c.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.3,
                      f'{val:.1f}',
                      ha='center', va='bottom',
                      fontsize=5, color='#333333')

# —— X 轴 ——
ax_c.set_xticks(x)
ax_c.set_xticklabels(group_labels, fontsize=7, rotation=45, ha='right')
ax_c.set_xlim(-0.8, N_IND - 0.2)

# —— Y 轴 ——
ax_c.set_ylabel('Weight (%)', fontsize=9)
ax_c.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax_c.tick_params(axis='y', labelsize=7)

# —— 网格 ——
ax_c.yaxis.grid(True, linestyle=':', alpha=0.5, linewidth=0.4)
ax_c.set_axisbelow(True)

# —— 四边脊线 (SCI 四方框) ——
ax_c.spines['top'].set_visible(True)
ax_c.spines['right'].set_visible(True)

# —— 子图标签 (c) ——
ax_c.text(-0.04, 1.03, '(c)', transform=ax_c.transAxes,
          fontsize=12, fontweight='bold', va='bottom', ha='left')

# —— 标题 ——
ax_c.set_title('AHP+RF-CRITIC Nash Fusion — Weight Allocation\n'
               '(AHP Subjective × RF-CRITIC Objective × Nash Equilibrium)',
               fontsize=10, fontweight='bold', pad=6)

# —— 图例 ——
legend_c = ax_c.legend(loc='upper right', fontsize=8,
                        frameon=True, fancybox=False,
                        edgecolor='#999999', facecolor='white',
                        framealpha=0.9, ncol=1)
legend_c.set_title('Weight Type', prop={'size': 9})

# ╔══════════════════════════════════════════════════════════╗
# ║             10. 整体边距 & 子图间距                        ║
# ╚══════════════════════════════════════════════════════════╝
plt.subplots_adjust(
    left=0.06,
    right=0.96,
    bottom=0.05,
    top=0.95,
    hspace=0.35,
    wspace=0.40,
)

# ╔══════════════════════════════════════════════════════════╗
# ║             11. 保存 & 输出                               ║
# ╚══════════════════════════════════════════════════════════╝
output_base = os.path.join(OUT_DIR, 'Figure_Nash_Fusion_Overview')
print(f"\nExporting figure ({FIG_WIDTH_CM}×{FIG_HEIGHT_CM} cm, 500 dpi)...")
save_fig(fig, output_base, dpi=500)

# 交互预览 —— 开发时取消注释
# plt.show()

plt.close(fig)
print("\n" + "=" * 70)
print("[DONE] 三合一组合图已生成！🦐")
print(f"  (a) 权重雷达 — 4方法 × 14指标")
print(f"  (b) 统计雷达 — 4方法 × 7指标 (归一化)")
print(f"  (c) 柱形图 — AHP+RF-CRITIC: 14指标 × 3权重")
print("=" * 70)
