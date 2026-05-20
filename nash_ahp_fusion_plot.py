# -*- coding: utf-8 -*-
"""
==============================================================================
水体指标权重融合 — 无约束纳什均衡（AHP主观 + 4种客观权重）【画图模块 v3】
==============================================================================

依赖：先运行 nash_ahp_fusion_calc.py 生成 Excel 结果文件

图(a) — 雷达图（polar）：四种融合方式 × 14个指标的融合权重
图(b) — 雷达图（polar）：四种融合方式 × 7个统计指标 (SD/Variance/Median/SE/IQR/CV/KMO)
图(c) — 柱形图（bar）  ：AHP+RF-CRITIC 方式下，14指标 × 三柱并列(主观/客观/融合)

SCI 规范：Arial 字体 / 莫兰迪配色 / cm 尺寸 / PNG+TIFF 双导出
计算模块：nash_ahp_fusion_calc.py

━━━━━━━━━━━━━━━━━━━━━━━ 微调工作流 ━━━━━━━━━━━━━━━━━━━━━━━
1. 代码中所有【可调】标记处均可按需修改
2. 改完参数后运行脚本 → 查看 PNG 预览
3. 如需交互预览 → 取消底部 # plt.show() 注释
4. 确认无误后，TIFF (500 dpi LZW) 即为投稿用图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 1 节 · 全局尺寸 & 单位换算                                            ║
# ║  — 所有物理尺寸用 cm 定义，通过 CM2INCH 转为英寸                            ║
# ║  — SCI 整页 ≤ 19 cm，单列 ≤ 9 cm                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CM2INCH = 1 / 2.54          # 厘米 → 英寸换算常数，勿动
                             # 用法: any_cm * CM2INCH = inch

# ── 图片物理尺寸 ──────────────────────────────────────────────────
# 【可调】改这两个值控制整幅图宽/高
FIG_WIDTH_CM  = 17.5         # 总宽度 (cm)。当前值适配雷达圆方等宽，上下对齐
                             #   增大→整图变宽，子图间距不变时每个子图更宽
FIG_HEIGHT_CM = 22           # 总高度 (cm)。当前 = 上排雷达 ~9cm + 下排柱形 ~13cm
                             #   增大→整图变高，下排柱形图获得更多纵向空间
                             #   减小→图变矮，柱形图 X 轴标签可能被裁


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 2 节 · SCI 全局字体 & 线宽 (rcParams)                                 ║
# ║  — 所有未显式设定字号的文本继承 font.size=9                                ║
# ║  — 所有未显式设定线宽的元素继承 axes.linewidth=1.0                          ║
# ║  — 字号不得小于 6 pt（印刷可读下限）                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 字体族 ────────────────────────────────────────────────────────
rcParams['font.family'] = 'Arial'       # 【SCI 标准，勿动】可选: Times New Roman, Helvetica
rcParams['font.size']   = 9             # 【可调 8~10】全局基准字号，影响坐标系内文字

# ── 字号分级 ──────────────────────────────────────────────────────
#   以下逐项覆盖全局 font.size，实现分级：
#     axes.titlesize  → 子图标题（如 "Nash Fusion Weights — 14 Indicators"）
#     axes.labelsize  → 坐标轴标签（如 "Weight (%)"）
#     xtick.labelsize → X 轴刻度数字
#     ytick.labelsize → Y 轴刻度数字
#     legend.fontsize → 图例正文
#     legend.title_fontsize → 图例标题（柱形图用，雷达图无标题）
rcParams['axes.titlesize']    = 10      # 【可调 10~12】
rcParams['axes.labelsize']    = 9       # 【可调 8~10】
rcParams['xtick.labelsize']   = 7       # 【可调 6~8】
rcParams['ytick.labelsize']   = 7       # 【可调 6~8】
rcParams['legend.fontsize']   = 8       # 【可调 7~9】
rcParams['legend.title_fontsize'] = 9   # 【可调 8~10】

# ── 线宽 ──────────────────────────────────────────────────────────
#   axes.linewidth     → 坐标轴脊线（四边框线），不宜过粗
#   xtick.major.width  → X 轴主刻度线宽
#   ytick.major.width  → Y 轴主刻度线宽
#   xtick.major.size   → X 轴主刻度伸出长度 (pt)，0=隐藏
#   ytick.major.size   → Y 轴主刻度伸出长度 (pt)
rcParams['axes.linewidth']     = 1.0    # 【可调 0.8~1.2】
rcParams['xtick.major.width']  = 0.8    # 【可调 0.6~1.0】
rcParams['ytick.major.width']  = 0.8
rcParams['xtick.major.size']   = 3      # 【可调 2~4】
rcParams['ytick.major.size']   = 3
rcParams['xtick.minor.size']   = 1.5    # 次刻度，一半长
rcParams['ytick.minor.size']   = 1.5

# ── 分辨率 & 保存 ──────────────────────────────────────────────────
#   figure.dpi  → 屏幕渲染 DPI（开发调试用，不影响导出）
#   savefig.dpi → 导出 DPI。组合图 ≥ 500，纯线条 ≥ 1000
#   savefig.bbox → 'tight' 自动裁掉子图外的白边
#   savefig.pad_inches → tight 模式下额外留白 (英寸)
rcParams['figure.dpi']         = 300    # 【可调】开发时设 150 可加速预览
rcParams['savefig.dpi']        = 500    # 【可调 300~1000】投稿≥500
rcParams['savefig.bbox']       = 'tight'
rcParams['savefig.pad_inches'] = 0.05   # 【可调 0~0.1】增大→四周多留白


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 3 节 · 莫兰迪配色 (Morandi Palette)                                   ║
# ║  — 8 色低饱和柔和调，打印友好，SCI 首选                                     ║
# ║  — 通过 morandi(idx) 循环取色，想换颜色改数组即可                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MORANDI = [
    "#C49A8C",  # [0] 陶土玫瑰  → 雷达图 方法1 / 柱形图 AHP
    "#9CAF88",  # [1] 鼠尾草绿  → 雷达图 方法2
    "#8FA7B7",  # [2] 灰蓝      → 雷达图 方法3
    "#B8A99A",  # [3] 暖灰      → 雷达图 方法4
    "#A88C9A",  # [4] 淡紫灰    → 备用
    "#B0A878",  # [5] 橄榄黄绿  → 备用
    "#7E8DA2",  # [6] 石板蓝    → 柱形图 RF-CRITIC
    "#C1876B",  # [7] 陶土棕    → 柱形图 Nash Fusion
]
# 【可调】增删颜色、调换顺序。原则：5~8 色，相邻色灰度/色相有区分即可

def morandi(idx):
    """循环取莫兰迪色，idx 越界自动折返"""
    return MORANDI[idx % len(MORANDI)]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 4 节 · 统一导出函数                                                    ║
# ║  — PNG 用于预览/文档插入，TIFF LZW 用于投稿                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_fig(fig, name, dpi=500):
    """
    同时输出 PNG (预览) + TIFF LZW (投稿)

    参数
    ----
    fig  : matplotlib Figure 对象
    name : 输出路径（不含扩展名），如 'D:/path/Figure_xxx'
    dpi  : 导出分辨率。【可调】500=组合图, 300=纯彩色, 1000=纯线条
    """
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{name}.tif", dpi=dpi, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"  Saved: {os.path.basename(name)}.png / .tif")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 5 节 · 路径 & 数据加载                                                 ║
# ║  — 所有数据来自 nash_ahp_fusion_calc.py 生成的 Excel                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

OUT_DIR = r"D:\WQIPaper\DataAnalytics"                     # 【可调】输出目录
EXCEL_PATH = os.path.join(OUT_DIR, "Nash_AHP_Objective_Fusion_Weights.xlsx")

if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(
        f"找不到 {EXCEL_PATH}\n"
        f"请先运行 nash_ahp_fusion_calc.py 生成计算结果！"
    )

print("=" * 70)
print("Loading results from Excel...")

# ── Sheet ①: Fusion_Weights_14Indicators ──
#   列 = Indicator | AHP+EWM_Weight | AHP+EWM_Pct | AHP+RF_Weight | ...
#   行 = 14 个水质指标
df_fw = pd.read_excel(EXCEL_PATH, sheet_name='Fusion_Weights_14Indicators', engine='openpyxl')
ahp_indicator_order = df_fw['Indicator'].tolist()     # 14 个指标全名
N_IND = len(ahp_indicator_order)                      # = 14

# 融合方式的显示名（顺序决定雷达图颜色分配）
# 【可调】增删方法、改显示名
fusion_method_names = [
    'AHP+EWM',
    'AHP+RF',
    'AHP+CRITIC',
    'AHP+RF-CRITIC(constrained)',
]

# 提取四种融合权重（原始 0~1 数值，用于雷达图(a)）
fusion_weights_dict = {}
for method in fusion_method_names:
    col = f'{method}_Weight'
    fusion_weights_dict[method] = df_fw[col].values   # shape (14,)

# ── Sheet ②: Nash_Fusion_Weights ──
#   获取 AHP 原始权重 和 RF-CRITIC 原始权重（用于柱形图(c)的前两根柱子）
df_nash = pd.read_excel(EXCEL_PATH, sheet_name='Nash_Fusion_Weights', engine='openpyxl')
w_ahp_arr       = df_nash['AHP_Weight_%'].values / 100         # → 0~1
w_rf_critic_arr = df_nash['RF_CRITIC_Weight_%'].values / 100   # → 0~1

# ── Sheet ③: Statistics ──
#   行 = 4 融合方式, 列 = SD/Variance/Median/SE/IQR/CV/KMO
df_stats = pd.read_excel(EXCEL_PATH, sheet_name='Statistics', engine='openpyxl', index_col=0)
stat_methods = df_stats.index.tolist()      # 与 fusion_method_names 顺序一致
stat_names   = ['SD', 'Variance', 'Median', 'SE', 'IQR', 'CV', 'KMO']
stat_labels  = ['SD (%)', 'Variance', 'Median (%)', 'SE (%)', 'IQR (%)', 'CV', 'KMO']
# stat_labels 用于雷达图(b)的轴标签，可加单位

# 构建统计值矩阵 (4 methods × 7 stats)
stats_matrix = np.zeros((len(stat_methods), len(stat_names)))
for i, method in enumerate(stat_methods):
    for j, sn in enumerate(stat_names):
        stats_matrix[i, j] = df_stats.loc[method, sn]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 6 节 · 创建画布 & GridSpec 布局                                        ║
# ║  — 2 行 × 2 列网格，上排两个 polar 雷达，下排一个 bar 柱形（跨两列）          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(FIG_WIDTH_CM * CM2INCH, FIG_HEIGHT_CM * CM2INCH))

# GridSpec 定义网格比例
#   height_ratios — 各行高度比。当前上排 42% / 下排 58%，总和可不等于 1
#   hspace        — 行间垂直间距（占子图平均高度的比例）
#   wspace        — 列间水平间距（占子图平均宽度的比例）
# 【可调】改 height_ratios 调整上下排高度分配
#         改 hspace/wspace 控制子图松紧
gs = GridSpec(2, 2, figure=fig,
              height_ratios=[0.42, 0.58],   # 【可调】上排雷达/下排柱形高度比
              hspace=0.30,                   # 【可调 0.20~0.45】垂直间距
              wspace=0.28)                   # 【可调 0.20~0.40】水平间距

# 上排：polar=True 创建极坐标子图（雷达图）
ax_a = fig.add_subplot(gs[0, 0], polar=True)   # 图(a) — 权重雷达
ax_b = fig.add_subplot(gs[0, 1], polar=True)   # 图(b) — 统计雷达

# 下排：普通直角坐标子图，跨两列 (gs[1, :])
ax_c = fig.add_subplot(gs[1, :])               # 图(c) — 三柱并列


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 7 节 · 图(a) — 权重雷达图                                              ║
# ║  — 4 条线 = 四种融合方式，14 个轴 = 14 个水质指标                            ║
# ║  — 每条线: 半透明填充 + 折线 + 白色圆点标记                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("Plot (a): Weight Radar Chart...")

# ── 7a. 指标简写 (雷达轴标签，14个全名太长会重叠) ──
# 【可调】改 short_names 列表调整雷达轴上的标签文字
short_names = [
    'THMs', 'Pb', 'Free Chlorine', 'NO$_{3}^{-}$', 'TOC',
    'F$^{-}$', 'COD$_{Mn}$', 'Al', 'TDS', 'pH',
    'TH', 'SO$_{4}^{2-}$', 'Temp', 'Cl$^{-}$',
]

N = N_IND                                                     # 轴数 = 14
# 计算各轴的角度 (0° 起，逆时针均匀分布)；末尾补首个值以闭合曲线
angles_a = [n / float(N) * 2 * np.pi for n in range(N)]
angles_a += angles_a[:1]                                      # 闭合

# ── 7b. 确定 Y 轴范围 ──
# 取所有融合权重的最大值，向上取整留一点余量
all_weights_concat = np.concatenate([fusion_weights_dict[m] for m in fusion_method_names])
max_w = all_weights_concat.max()
ylim_a = np.ceil(max_w * 100) / 100 + 0.02                    # 【可调】+0.02 是顶部余量

# ── 7c. 逐条画四种融合方式 ──
for idx, method in enumerate(fusion_method_names):
    values = list(fusion_weights_dict[method]) + [fusion_weights_dict[method][0]]
    color = morandi(idx)                                       # 莫兰迪配色

    # 半透明填充 (增强视觉层次)
    ax_a.fill(angles_a, values,
              alpha=0.08,                                      # 【可调 0.05~0.20】填充透明度
              color=color)

    # 折线 + 白色圆点标记
    ax_a.plot(angles_a, values, 'o-',
              linewidth=1.2,                                   # 【可调 0.8~1.5】数据线宽
              color=color,
              markersize=3.5,                                  # 【可调 2~5】圆点大小
              markerfacecolor='white',                         # 圆点填充白色（镂空效果）
              markeredgewidth=1.0,                             # 【可调 0.5~1.5】圆点边框宽
              markeredgecolor=color,
              label=method.replace('(constrained)',
                                   ''))         # 图例换行，避免过长

# ── 7d. 轴标签与刻度 ──
ax_a.set_xticks(angles_a[:-1])
ax_a.set_xticklabels(short_names,
                     fontsize=7)                               # 【可调 6~8】轴标签字号
ax_a.set_ylim(0, ylim_a)

# Y 轴刻度：5 档均匀分布
yticks_a = np.round(np.linspace(0, ylim_a, 5), 2)
ax_a.set_yticks(yticks_a)
ax_a.set_yticklabels([f'{v:.2f}' for v in np.linspace(0, ylim_a, 5)],
                     fontsize=6)                               # 【可调 5~7】环形刻度字号
ax_a.set_rlabel_position(30)                                   # 【可调 0~360】刻度标签角度

# ── 7e. 网格 ──
ax_a.grid(True,
          alpha=0.3,                                           # 【可调 0.1~0.5】网格透明度
          linewidth=0.4)                                       # 【可调 0.3~0.5】网格线宽

# ── 7f. 子图标签 (a) ──
# transAxes 坐标：(0,0)=左下, (1,1)=右上
ax_a.text(-0.12, 1.08, '(a)',
          transform=ax_a.transAxes,
          fontsize=12,                                         # 【可调 10~14】子图标签字号
          fontweight='bold',                                   # 加粗
          va='bottom', ha='left')

# ── 7g. 图例 (右下，无框) ──
# loc='lower right'  → 图例右下角对齐 bbox_to_anchor 位置
# bbox_to_anchor     → 在图内坐标 (0.98, 0.02) = 几乎最右下
# frameon=False      → 无图例外框
legend_a = ax_a.legend(loc='lower right',
                       bbox_to_anchor=(1.25, -0.25),            # 【可调】微调图例位置
                       fontsize=7,                             # 【可调 6~8】图例字号
                       frameon=False,                          # False=无框, True=有框
                       ncol=1)                                 # 【可调】图例列数

# ── 7h. 标题 ──
ax_a.set_title('Nash Fusion Weights — 14 Indicators',
               fontsize=10,                                    # 【可调 10~12】
               fontweight='bold',                                                                        
               pad=18)                                         # 【可调 12~24】标题距子图间距


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 8 节 · 图(b) — 统计指标雷达图                                          ║
# ║  — 4 条线 = 四种融合方式，7 个轴 = 7 个统计指标                             ║
# ║  — 归一化方式: val / max → 每条轴都从 0 画起，保持量级比例                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("Plot (b): Statistics Radar Chart...")

# ── 8a. 归一化：除以各指标的最大值，映射到 [0, 1] ──
#        0 = 该统计量为 0（理论下限）
#        1 = 该统计量达到四种方法中的最大值
#        优点：不会像 min-max 那样把微小差异拉满整个轴
stats_norm = np.zeros_like(stats_matrix)                        # shape (4, 7)
stats_max_vals = []                                             # 记录各指标 max，用于调试
for j in range(len(stat_names)):
    col_max = stats_matrix[:, j].max()
    stats_max_vals.append(col_max)
    if col_max < 1e-12:
        stats_norm[:, j] = 0.0                                  # 全零列兜底
    else:
        stats_norm[:, j] = stats_matrix[:, j] / col_max

N_stats = len(stat_names)                                       # = 7
angles_b = [n / float(N_stats) * 2 * np.pi for n in range(N_stats)]
angles_b += angles_b[:1]                                        # 闭合

# ── 8b. 逐条画四种融合方式 ──
for idx, method in enumerate(stat_methods):
    values = list(stats_norm[idx]) + [stats_norm[idx][0]]
    color = morandi(idx)                                        # 与图(a)配色一一对应

    ax_b.fill(angles_b, values,
              alpha=0.08,                                       # 【可调】填充透明度
              color=color)

    ax_b.plot(angles_b, values, 'o-',
              linewidth=1.2,                                    # 【可调】数据线宽
              color=color,
              markersize=3.5,                                   # 【可调】圆点大小
              markerfacecolor='white',
              markeredgewidth=1.0,                              # 【可调】圆点边框宽
              markeredgecolor=color,
              label=method.replace('(constrained)',
                                   '\n(constrained)'))

# ── 8c. 轴标签与刻度 ──
ax_b.set_xticks(angles_b[:-1])
ax_b.set_xticklabels(stat_labels,
                     fontsize=7)                                # 【可调 6~8】
ax_b.set_ylim(0, 1.05)                                          # y=1.05 留一点顶部呼吸空间
ax_b.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax_b.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'],
                     fontsize=6)                                # 【可调 5~7】
ax_b.set_rlabel_position(30)                                    # 【可调】刻度标签角度

# ── 8d. 网格 ──
ax_b.grid(True,
          alpha=0.3,                                            # 【可调】
          linewidth=0.4)                                        # 【可调】

# ── 8e. 子图标签 (b) ──
ax_b.text(-0.12, 1.08, '(b)',
          transform=ax_b.transAxes,
          fontsize=12,                                          # 【可调 10~14】
          fontweight='bold',
          va='bottom', ha='left')

# ── 8f. 图例 (右下，无框) ──
legend_b = ax_b.legend(loc='lower right',
                       bbox_to_anchor=(1.25, -0.25),             # 【可调】
                       fontsize=7,                              # 【可调 6~8】
                       frameon=False,
                       ncol=1)

# ── 8g. 标题 ──
ax_b.set_title('Statistical Indicators — 7 Metrics (0-based scale)',
               fontsize=10,                                     # 【可调 10~12】
               fontweight='bold',
               pad=18)                                          # 【可调 12~24】


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 9 节 · 图(c) — AHP+RF-CRITIC 三柱并列图                               ║
# ║  — 14 组 = 14 个指标，每组 3 根柱子                                        ║
# ║  — 柱 1: AHP(陶土玫瑰) / 柱 2: RF-CRITIC(石板蓝) / 柱 3: Nash融合(陶土棕)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("Plot (c): AHP+RF-CRITIC Weight Distribution Bar Chart...")

# ── 9a. 数据准备 ──
w_nash_rf_critic = fusion_weights_dict['AHP+RF-CRITIC(constrained)']  # 融合权重 (0~1)

group_labels = short_names                                             # X 轴标签用简写
x = np.arange(N_IND)                                                  # 14 个组中心位置

# 柱子参数
width = 0.25                                                           # 【可调 0.15~0.30】单柱宽度
gap   = 0.02                                                           # 【可调 0~0.05】同组柱间缝隙
#   三柱总占宽 ≈ 3×width + 2×gap，应 < 1.0 避免组间重叠

# 配色：三根柱子分别用莫兰迪色系中不同色相
#   [0] 陶土玫瑰 → AHP (主观)
#   [6] 石板蓝   → RF-CRITIC (客观)
#   [7] 陶土棕   → Nash Fusion (融合)
bar_colors_c = [morandi(0), morandi(6), morandi(7)]                    # 【可调】
bar_labels_c = ['AHP (Subjective)', 'RF-CRITIC (Objective)', 'Nash Fusion']
bar_data     = [w_ahp_arr, w_rf_critic_arr, w_nash_rf_critic]

# ── 9b. 逐组画柱 ──
for i, (data, color, label) in enumerate(zip(bar_data, bar_colors_c, bar_labels_c)):
    # 偏移量: 中柱居中(i=1时offset=0)，左柱左移，右柱右移
    offset = (i - 1) * (width + gap)

    bars = ax_c.bar(x + offset, data * 100, width,                     # *100 → 百分比
                    color=color,
                    alpha=0.85,                                        # 【可调 0.70~0.95】柱填充透明度
                    edgecolor='white',                                 # 白色边线让柱子清晰分离
                    linewidth=0.3,                                     # 【可调 0~0.5】柱子边框宽
                    label=label,
                    zorder=2)                                          # 置于网格上方

    # ── 数值标注 ──
    # 只在权重 > 2% 的柱子上方标注，避免小柱子挤成一团
    # 【可调】改阈值 2.0 控制标注密度，改 fontsize 调整字号
    for j, (bar, val) in enumerate(zip(bars, data * 100)):
        if val > 2.0:                                                  # 【可调】标注阈值 (%)
            ax_c.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.3,                          # 【可调】标注距柱顶偏移
                      f'{val:.1f}',                                    # 保留 1 位小数
                      ha='center', va='bottom',
                      fontsize=5,                                      # 【可调 4.5~6】标注字号
                      color='#333333',                                 # 【可调】标注颜色
                      zorder=10)

# ── 9c. X 轴 ──
ax_c.set_xticks(x)
ax_c.set_xticklabels(group_labels,
                     fontsize=7,                                       # 【可调 6~8】
                     rotation=45,                                      # 【可调 0~90】标签旋转角度
                     ha='right')                                       # 'right' 配合 rotation 用
ax_c.set_xlim(-0.8, N_IND - 0.2)                                      # 【可调】左右留白

# ── 9d. Y 轴 ──
ax_c.set_ylabel('Weight (%)',
                fontsize=9)                                            # 【可调 8~10】
ax_c.yaxis.set_major_locator(MaxNLocator(nbins=6))                    # 【可调】Y 轴刻度档数
ax_c.tick_params(axis='y', labelsize=7)                                # 【可调 6~8】

# ── 9e. 网格 (仅 Y 方向，浅色点线) ──
ax_c.yaxis.grid(True,
                linestyle=':',                                         # 【可调】':'=点线, '--'=虚线, '-'=实线
                alpha=0.5,                                             # 【可调 0.3~0.7】
                linewidth=0.4)                                         # 【可调 0.3~0.5】
ax_c.set_axisbelow(True)                                               # 网格置于数据下方

# ── 9f. 四边脊线 (SCI 风格方框，非仅下左两轴) ──
ax_c.spines['top'].set_visible(True)
ax_c.spines['right'].set_visible(True)
# 右轴不留刻度（只留框线）
ax_c.tick_params(axis='y', which='both', right=False, labelright=False)

# ── 9g. 子图标签 (c) ──
ax_c.text(-0.04, 1.03, '(c)',
          transform=ax_c.transAxes,
          fontsize=12,                                                # 【可调 10~14】
          fontweight='bold',
          va='bottom', ha='left')

# ── 9h. 标题 ──
ax_c.set_title('AHP+RF-CRITIC Nash Fusion — Weight Allocation\n'
               '(AHP Subjective  ×  RF-CRITIC Objective  ×  Nash Equilibrium)',
               fontsize=10,                                           # 【可调 10~12】
               fontweight='bold',
               pad=6)                                                 # 【可调 4~10】

# ── 9i. 图例 (右上，有框，区别于雷达图) ──
legend_c = ax_c.legend(loc='upper right',
                       fontsize=8,                                    # 【可调 7~9】
                       frameon=True,                                  # 【可调】True=有框, False=无框
                       fancybox=False,
                       edgecolor='#999999',                           # 【可调】图例边框颜色
                       facecolor='white',
                       framealpha=0.9,                                # 【可调】图例背景透明度
                       ncol=1)                                        # 【可调】列数
legend_c.set_title('Weight Type', prop={'size': 9})                   # 【可调】图例标题


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 10 节 · 整体边距 & 子图间距 (subplots_adjust)                         ║
# ║  — 所有值为 figure 比例 (0~1)，控制子图集合在画布中的位置                     ║
# ║  — 微调思路：                                                             ║
# ║    Y轴标签被裁 → 增大 left                                               ║
# ║    X轴标签被裁 → 增大 bottom                                             ║
# ║    标题被裁     → 增大 top 或减小 hspace                                  ║
# ║    子图间太挤   → 增大 hspace / wspace                                   ║
# ║    子图间太散   → 减小 hspace / wspace                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

plt.subplots_adjust(
    left   = 0.05,       # 【可调 0.04~0.10】左边距
    right  = 0.97,       # 【可调 0.93~0.98】右边距
    bottom = 0.06,       # 【可调 0.04~0.12】底边距（柱形图X轴标签需要空间）
    top    = 0.95,       # 【可调 0.90~0.97】顶边距
    hspace = 0.30,       # 【可调 0.20~0.45】行间垂直间距
    wspace = 0.28,       # 【可调 0.20~0.40】列间水平间距
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 11 节 · 保存 & 输出                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

output_base = os.path.join(OUT_DIR, 'Figure_Nash_Fusion_Overview')
print(f"\nExporting figure ({FIG_WIDTH_CM}×{FIG_HEIGHT_CM} cm, 500 dpi)...")
save_fig(fig, output_base, dpi=500)         # 【可调】改 dpi: 300=快览, 500=投稿, 1000=线条

# ── 交互预览开关 ─────────────────────────────────────────────────────
# 【用法】开发调试时取消下面这行的注释，弹出预览窗口
#         确认无误后在行首加回 # 注释掉，再运行批量出图
# plt.show()

plt.close(fig)
print("\n" + "=" * 70)
print("[DONE] 三合一组合图已生成！🦐")
print(f"  (a) 权重雷达 — 4方法 × 14指标")
print(f"  (b) 统计雷达 — 4方法 × 7指标 (0-based 归一化)")
print(f"  (c) 柱形图   — AHP+RF-CRITIC: 14指标 × 3权重")
print("=" * 70)
