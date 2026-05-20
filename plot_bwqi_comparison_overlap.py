"""
Figure variant: BWQI comparison — OVERLAPPING scatter version
═══════════════════════════════════════════════════════════════════════
内容：三模型 BWQI 对比组合图（重叠散点版）
  (a) 左：时间序列重叠散点 + 平滑趋势线 + 内嵌小提琴图
  (b) 右上：直方图 + KDE 核密度曲线
  (c) 右下：累积分布函数 CDF
重叠效果：同年三模型散点完全叠在同一 jitter 位置，
          半透明颜色叠加产生颜色混合（红+蓝→紫、三色全叠→深灰）
═══════════════════════════════════════════════════════════════════════
依赖：pandas, numpy, matplotlib, scipy, openpyxl
数据：BWQI_Results.xlsx + Extracted_WaterQuality.xlsx
输出：Figure_BWQI_Comparison_Overlap.png + .tif (LZW)
═══════════════════════════════════════════════════════════════════════
SCI 规范版本：v2 (2026-05-20)
参考：memory/sci-figure-standards.md §1-§10
开发工作流：开发时取消注释底部的 plt.show() 预览；
             确认后重新注释掉 plt.show()，运行脚本批量出图。
═══════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde

# ══════════════════════════════════════════════════════════════════════
# 1. 全局尺寸 & 单位换算
# ══════════════════════════════════════════════════════════════════════
CM2INCH = 1 / 2.54  # cm → inch 换算系数（inch = cm × CM2INCH）

# ── 画布物理尺寸 ──
FIG_WIDTH_CM  = 19.0   # 【可调 15~19】总宽度 (cm)，整页组合图 ≤19 cm
FIG_HEIGHT_CM = 16.0   # 【可调 12~22】总高度 (cm)，按内容比例调整
FIG_WIDTH_IN  = FIG_WIDTH_CM  * CM2INCH  # → 英寸
FIG_HEIGHT_IN = FIG_HEIGHT_CM * CM2INCH  # → 英寸

# ── 导出 DPI ──
EXPORT_DPI = 500  # 【SCI 标准】组合图（线条+半色调）≥500 dpi

print(f"[画布] {FIG_WIDTH_CM:.1f}×{FIG_HEIGHT_CM:.1f} cm  →  {FIG_WIDTH_IN:.2f}×{FIG_HEIGHT_IN:.2f} inch")
print(f"[导出] DPI={EXPORT_DPI}  →  {int(FIG_WIDTH_IN*EXPORT_DPI)}×{int(FIG_HEIGHT_IN*EXPORT_DPI)} px")

# ══════════════════════════════════════════════════════════════════════
# 2. 路径配置
# ══════════════════════════════════════════════════════════════════════
BWQI_PATH      = r'D:\WQIPaper\DataAnalytics\BWQI_Results.xlsx'
ORIG_DATA_PATH = r'D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx'
OUTPUT_PATH    = r'D:\WQIPaper\DataAnalytics\Figure_BWQI_Comparison_Overlap.png'
OUTPUT_TIF     = r'D:\WQIPaper\DataAnalytics\Figure_BWQI_Comparison_Overlap.tif'

# ══════════════════════════════════════════════════════════════════════
# 3. 配色方案
# ══════════════════════════════════════════════════════════════════════
# 注：此处使用语义化配色（红/蓝/绿）而非莫兰迪，因为三模型需强区分度。
#     后续如需改为莫兰迪色系，替换下方 HEX 值即可。
COLORS = {
    'WQM':  {'main': '#D62728',   # 【可调】WQM 主色 — 深红
             'light': '#F4A8A8',  # 【可调】WQM 浅色（直方图填充）
             'fill': '#FDE8E8',   # 【可调】WQM 极浅（小提琴填充）
             'label': 'DLIR + WQM'},
    'LQM':  {'main': '#1F77B4',   # 【可调】LQM 主色 — 深蓝
             'light': '#A8C8E8',  # 【可调】LQM 浅色
             'fill': '#E8F0F8',   # 【可调】LQM 极浅
             'label': 'EF + LQM'},
    'SWM':  {'main': '#2CA02C',   # 【可调】SWM 主色 — 深绿
             'light': '#A8D8A8',  # 【可调】SWM 浅色
             'fill': '#E8F8E8',   # 【可调】SWM 极浅
             'label': 'ISF + SWM'},
}
MODELS    = ['WQM', 'LQM', 'SWM']
BWQI_COLS = {m: f'BWQI_{m}' for m in MODELS}

# ══════════════════════════════════════════════════════════════════════
# 4. SCI 字体 & 线宽配置 (rcParams)
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    # ── 字体 ──
    'font.family':  'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size':       9,     # 【SCI 标准】基础字号 8–10 pt
    'axes.titlesize':  10,    # 【SCI 标准】子图标题 10–12 pt
    'axes.labelsize':  9,     # 【SCI 标准】轴标签 8–10 pt
    'xtick.labelsize': 7,     # 【SCI 标准】X 刻度 6–8 pt
    'ytick.labelsize': 7,     # 【SCI 标准】Y 刻度 6–8 pt
    'legend.fontsize': 8,     # 【SCI 标准】图例 7–9 pt
    'legend.title_fontsize': 8,

    # ── 线宽 ──
    'axes.linewidth':  1.0,   # 【SCI 标准】坐标轴脊线 0.8–1.2 pt
    'grid.linewidth':  0.4,   # 【SCI 标准】网格线 0.3–0.5 pt
    'lines.linewidth': 1.5,   # 【SCI 标准】默认数据线 1–2 pt

    # ── 图面 ──
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',  # 轴脊颜色
    'axes.grid':        True,
    'grid.alpha':       0.3,        # 【可调 0.2~0.4】网格透明度
    'grid.linestyle':   '--',       # 网格线型
    'grid.color':       '#CCCCCC',  # 网格颜色
})

# ══════════════════════════════════════════════════════════════════════
# 5. 统一导出函数
# ══════════════════════════════════════════════════════════════════════
def save_fig(fig, png_path, tif_path, dpi=EXPORT_DPI):
    """
    统一保存 PNG（预览）+ TIFF（投稿，LZW 无损压缩）
    参数：
      fig      — matplotlib Figure 对象
      png_path — PNG 输出路径
      tif_path — TIFF 输出路径
      dpi      — 导出分辨率（默认 500）
    """
    fig.savefig(png_path, dpi=dpi, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    fig.savefig(tif_path, dpi=dpi, facecolor='white', edgecolor='none',
                bbox_inches='tight', format='tiff',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"\n[DONE] 图片已保存：")
    print(f"  PNG:  {png_path}")
    print(f"  TIFF: {tif_path}")

# ══════════════════════════════════════════════════════════════════════
# 6. 加载数据
# ══════════════════════════════════════════════════════════════════════
bwqi = pd.read_excel(BWQI_PATH, sheet_name='BWQI')
orig = pd.read_excel(ORIG_DATA_PATH)
bwqi['Date'] = pd.to_datetime(orig['Date'])   # 从原始数据取 Date 列
bwqi['Year'] = bwqi['Date'].dt.year           # 提取年份，用于逐年分组

print(f"[数据] 共 {len(bwqi)} 个样本，{bwqi['Year'].nunique()} 个年份 "
      f"({bwqi['Year'].min()}-{bwqi['Year'].max()})")

# ══════════════════════════════════════════════════════════════════════
# 7. 创建画布 & GridSpec 布局
# ══════════════════════════════════════════════════════════════════════
# 布局示意：
#   ┌──────────────────────┬─────────────┐
#   │                      │   (b) 直方图  │
#   │   (a) 重叠散点+趋势    │   + KDE     │
#   │   + 内嵌小提琴         ├─────────────┤
#   │                      │   (c) CDF    │
#   └──────────────────────┴─────────────┘
#
fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                 dpi=EXPORT_DPI, facecolor='white')

# GridSpec 参数说明：
#   width_ratios=[1.3, 1]  → 左列占 56.5%，右列占 43.5%
#   height_ratios=[1, 1]   → 上下行等高
#   hspace                 → 上下子图垂直间距
#   wspace                 → 左右子图水平间距
#   left/right/bottom/top  → 画布四边留白比例（0~1）
gs = GridSpec(2, 2, figure=fig,
              width_ratios=[1.3, 1],      # 【可调】左列比例
              height_ratios=[1, 1],        # 【可调】上行比例
              hspace=0.32,                 # 【可调 0.25~0.40】垂直间距，增大→拉高白边
              wspace=0.30,                 # 【可调 0.25~0.35】水平间距，增大→拉宽白边
              left=0.07,                   # 【可调 0.05~0.10】左边距，增大→防Y轴标签被裁
              right=0.97,                  # 【可调 0.93~0.98】右边距
              top=0.94,                    # 【可调 0.90~0.96】上边距
              bottom=0.08)                 # 【可调 0.06~0.12】下边距，增大→防X轴标签被裁

ax_a = fig.add_subplot(gs[:, 0])   # 左列，占据两行（图 a）
ax_b = fig.add_subplot(gs[0, 1])   # 右上（图 b）
ax_c = fig.add_subplot(gs[1, 1])   # 右下（图 c）

# ══════════════════════════════════════════════════════════════════════
# 8. 通用辅助：显示上/右脊线（无刻度）
# ══════════════════════════════════════════════════════════════════════
def show_all_spines(ax):
    """
    让子图显示四条脊线（上、下、左、右），其中右和上不设刻度。
    SCI 论文常见做法：四框完整，刻度仅在下/左。
    """
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    # 刻度仅在下/左显示
    ax.tick_params(top=False, labeltop=False, right=False, labelright=False)

# ══════════════════════════════════════════════════════════════════════
# 9. 子图 (a) — 重叠散点 + 平滑趋势线 + 内嵌小提琴图
# ══════════════════════════════════════════════════════════════════════

# ── 9a. 坐标轴范围与标签 ──
ax_a.set_xlim(2019.5, 2025.5)   # 【可调】X轴范围，留半年的边距
ax_a.set_ylim(48, 102)          # 【可调】Y轴范围，留 2pt 上下边距
ax_a.set_xlabel('Year')
ax_a.set_ylabel('Drinking Water Quality Index (BWQI)')
ax_a.set_xticks(range(2020, 2026))   # 每年一个主刻度
ax_a.set_yticks(range(50, 101, 10))  # 每 10 分一个刻度

# ── 9b. 子图标签 (a) — 左上角 ──
ax_a.text(0.02, 0.98, '(a)', transform=ax_a.transAxes, fontsize=10,
          fontweight='bold', va='top', ha='left')

# ── 9c. 显示四条脊线 ──
show_all_spines(ax_a)

# ── 9d. 重叠散点（三模型 Jitter 完全一致 → 叠出颜色混合） ──
SCATTER_SIZE   = 45     # 【可调 25~80】散点面积(s=点直径²)，越大越容易重叠
SCATTER_ALPHA  = 0.30   # 【可调 0.15~0.45】散点透明度，越小叠色越淡
JITTER_WIDTH   = 0.18   # 【可调 0.10~0.25】水平抖动幅度（年份宽度比例），越大越散

np.random.seed(42)  # 固定随机种子 → 每次出图 jitter 位置一致
for yr in range(2020, 2026):
    mask = bwqi['Year'] == yr
    n = mask.sum()
    if n == 0:
        continue
    # 关键：三个模型共用一个 jitter 数组 → 同年散点在 X 方向完全重叠
    jitter = np.random.uniform(-JITTER_WIDTH, JITTER_WIDTH, n)
    for model in MODELS:
        col = BWQI_COLS[model]
        # 只在第一年打 label（避免图例重复三次）
        label = COLORS[model]['label'] if yr == 2020 else None
        ax_a.scatter(yr + jitter, bwqi.loc[mask, col],
                     alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                     c=COLORS[model]['main'],
                     edgecolors='none',  # 无描边→叠加更自然
                     zorder=2,
                     label=label)

# ── 9e. 平滑趋势线（spline 插值各模型年均值） ──
years_array = np.arange(2020, 2026)
linestyles = {'WQM': '-', 'LQM': '--', 'SWM': '-.'}  # 【可调】线型区分三模型
TREND_LW = 2.0              # 【可调 1.5~2.5】趋势线宽度 (pt)
SPLINE_K  = 3               # 【可调 2~3】样条阶数，数据点<3 时自动降阶

for model in MODELS:
    col = BWQI_COLS[model]
    yearly_means = bwqi.groupby('Year')[col].mean()
    means_by_year = [yearly_means.get(y, np.nan) for y in years_array]
    valid_idx = ~np.isnan(means_by_year)
    valid_years = years_array[valid_idx]
    valid_means = np.array(means_by_year)[valid_idx]

    if len(valid_years) >= 3:
        # 样条插值平滑曲线
        x_smooth = np.linspace(valid_years[0], valid_years[-1], 200)  # 【可调 100~300】平滑点数
        k_order = min(SPLINE_K, len(valid_years) - 1)
        spl = make_interp_spline(valid_years, valid_means, k=k_order)
        y_smooth = spl(x_smooth)
        ax_a.plot(x_smooth, y_smooth,
                  color=COLORS[model]['main'],
                  linestyle=linestyles[model], linewidth=TREND_LW, zorder=5)
    else:
        # 数据点不足3个 → 折线连接（带圆点标记）
        ax_a.plot(valid_years, valid_means,
                  color=COLORS[model]['main'],
                  linestyle=linestyles[model], linewidth=TREND_LW,
                  marker='o', markersize=6, zorder=5)

# ── 9f. 图例 — (a) 左下角 ──
LEGEND_ALPHA_BG = 0.9  # 【可调 0.7~1.0】图例背景不透明度
ax_a.legend(loc='lower left', frameon=True, framealpha=LEGEND_ALPHA_BG,
            edgecolor='#CCCCCC', fontsize=8)

# ── 9g. 内嵌小提琴图（inset） ──
# inset_axes 参数说明：
#   width="43%"   → 【可调 35~50%】小图宽度占父图的百分比
#   height="38%"  → 【可调 30~45%】小图高度占父图的百分比
#   loc           → 放置位置：'lower right'
#   bbox_to_anchor → 微调偏移 (dx, dy, w, h)，单位是父图轴比例
INSET_WIDTH  = "43%"   # 【可调】
INSET_HEIGHT = "38%"   # 【可调】
ax_inset = inset_axes(ax_a, width=INSET_WIDTH, height=INSET_HEIGHT,
                       loc='lower right',
                       bbox_to_anchor=(0.02, 0.04, 1, 1),
                       bbox_transform=ax_a.transAxes)
ax_inset.set_facecolor('white')

# ── 小提琴子图标题 ──
ax_inset.set_title('Violin plots for DWQI under three models',
                    fontsize=7.5, fontweight='bold', pad=3)

# ── 小提琴子图也显示四条脊线 ──
show_all_spines(ax_inset)

# ── 箱型图（内嵌于小提琴内部） ──
violin_data = [bwqi[BWQI_COLS[m]].values for m in MODELS]
positions   = [1, 2, 3]
BOX_WIDTH = 0.15  # 【可调 0.10~0.20】箱型图宽度

bp = ax_inset.boxplot(violin_data, positions=positions, widths=BOX_WIDTH,
                       patch_artist=True,
                       boxprops=dict(facecolor='white', edgecolor='black',
                                     linewidth=0.8),
                       whiskerprops=dict(color='black', linewidth=0.8),
                       capprops=dict(color='black', linewidth=0.8),
                       medianprops=dict(color='black', linewidth=1.2),
                       flierprops=dict(marker='o', markersize=2,
                                       markerfacecolor='black',
                                       markeredgecolor='none', alpha=0.4))

# ── 小提琴主体 ──
VIOLIN_WIDTH = 0.65     # 【可调 0.50~0.80】小提琴宽度
VIOLIN_ALPHA = 0.85     # 【可调 0.70~1.00】小提琴填充透明度
vp = ax_inset.violinplot(violin_data, positions=positions,
                          showmeans=False, showmedians=False,
                          showextrema=False, widths=VIOLIN_WIDTH)
for i, model in enumerate(MODELS):
    vp['bodies'][i].set_facecolor(COLORS[model]['fill'])
    vp['bodies'][i].set_edgecolor(COLORS[model]['main'])
    vp['bodies'][i].set_linewidth(0.8)
    vp['bodies'][i].set_alpha(VIOLIN_ALPHA)

# ── 小提琴轴标签 ──
ax_inset.set_xticks(positions)
ax_inset.set_xticklabels(['WQM', 'LQM', 'SWM'], fontsize=7)
ax_inset.set_ylabel('BWQI (100-pt)', fontsize=7)
ax_inset.set_ylim(68, 100)   # 【可调】适配数据范围
ax_inset.tick_params(axis='y', labelsize=6.5)
ax_inset.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.4)

# ══════════════════════════════════════════════════════════════════════
# 10. 子图 (b) — 直方图 + KDE 核密度曲线
# ══════════════════════════════════════════════════════════════════════

# ── 坐标轴设置 ──
ax_b.set_xlim(68, 100)       # 【可调】与 (a)(c) 保持 X 轴一致
ax_b.set_ylim(0, 0.25)       # 【可调 0~0.15~0.30】概率密度上限
ax_b.set_xlabel('Drinking Water Quality Index (BWQI)')
ax_b.set_ylabel('Probability Density')
ax_b.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])

# ── 子图标签 (b) — 左上角 ──
ax_b.text(0.02, 0.98, '(b)', transform=ax_b.transAxes, fontsize=10,
          fontweight='bold', va='top', ha='left')

# ── 四条脊线 ──
show_all_spines(ax_b)

# ── 直方图 + KDE ──
HIST_BINS    = 18        # 【可调 14~24】直方图箱数，越多柱越细
HIST_ALPHA   = 0.35      # 【可调 0.20~0.45】直方图透明度
KDE_LW       = 1.8       # 【可调 1.2~2.2】KDE 曲线宽度 (pt)
KDE_POINTS   = 300       # 【可调 200~500】KDE 曲线采样点数
KDE_BW       = 'scott'   # 【可调 'scott'|'silverman'|数值】带宽方法，scott=较光滑

for model in MODELS:
    col = BWQI_COLS[model]
    data = bwqi[col].dropna().values
    # 直方图
    ax_b.hist(data, bins=HIST_BINS, density=True, alpha=HIST_ALPHA,
              color=COLORS[model]['light'], edgecolor=COLORS[model]['main'],
              linewidth=0.5, zorder=2)
    # KDE 核密度估计
    kde = gaussian_kde(data, bw_method=KDE_BW)
    x_kde = np.linspace(data.min() - 1, data.max() + 1, KDE_POINTS)
    ax_b.plot(x_kde, kde(x_kde), color=COLORS[model]['main'],
              linewidth=KDE_LW, zorder=5)

# ── 图例 — (b) 左上角，Patch 色块样式 ──
legend_b = [Patch(facecolor=COLORS[m]['light'], edgecolor=COLORS[m]['main'],
                   label=COLORS[m]['label'], linewidth=0.8) for m in MODELS]
ax_b.legend(handles=legend_b, loc='upper left', frameon=True,
            framealpha=LEGEND_ALPHA_BG, edgecolor='#CCCCCC', fontsize=8)

# ══════════════════════════════════════════════════════════════════════
# 11. 子图 (c) — 累积分布函数 CDF
# ══════════════════════════════════════════════════════════════════════

# ── 坐标轴设置 ──
ax_c.set_xlim(68, 100)       # 【可调】与其他子图 X 轴范围一致
ax_c.set_ylim(0, 1.02)       # CDF 固定 [0, 1]，留 0.02 上边距
ax_c.set_xlabel('Drinking Water Quality Index (BWQI)')
ax_c.set_ylabel('Cumulative Distribution')
ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])  # 四分位刻度

# ── 子图标签 (c) — 左上角 ──
ax_c.text(0.02, 0.98, '(c)', transform=ax_c.transAxes, fontsize=10,
          fontweight='bold', va='top', ha='left')

# ── 四条脊线 ──
show_all_spines(ax_c)

# ── CDF 曲线 ──
CDF_LW = 2.0  # 【可调 1.5~2.5】CDF 曲线宽度 (pt)

for model in MODELS:
    col = BWQI_COLS[model]
    data_sorted = np.sort(bwqi[col].dropna().values)
    cdf_y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax_c.plot(data_sorted, cdf_y, color=COLORS[model]['main'],
              linewidth=CDF_LW, zorder=3)

# ── 辅助参考线（四分位线） ──
QUARTILE_STYLE_MAIN = {'color': '#999999', 'linestyle': ':',   # 【可调】中位线样式
                        'linewidth': 0.8, 'alpha': 0.7}
QUARTILE_STYLE_MINOR = {'color': '#CCCCCC', 'linestyle': ':',  # 【可调】25/75分位线样式
                         'linewidth': 0.5, 'alpha': 0.5}

ax_c.axhline(y=0.50, **QUARTILE_STYLE_MAIN)    # 中位线
ax_c.axhline(y=0.25, **QUARTILE_STYLE_MINOR)   # 25 分位
ax_c.axhline(y=0.75, **QUARTILE_STYLE_MINOR)   # 75 分位

# ── 图例 — (c) 左上角 ──
ax_c.legend([COLORS[m]['label'] for m in MODELS],
            loc='upper left', frameon=True, framealpha=LEGEND_ALPHA_BG,
            edgecolor='#CCCCCC', fontsize=8)

# ══════════════════════════════════════════════════════════════════════
# 12. 微调边距（最终）
# ══════════════════════════════════════════════════════════════════════
# 注：subplots_adjust 是 GridSpec 边距的"最终覆盖"，可在此细调。
#     如果 GridSpec 的 left/right 等已经足够，此处可不调用。
#     如需微调，取消注释并修改下面的值即可。
#
# fig.subplots_adjust(
#     left=0.07,      # 【可调】左边界，增大→防Y轴标签被裁
#     right=0.97,     # 【可调】右边界
#     bottom=0.08,    # 【可调】下边界，增大→防X轴标签被裁
#     top=0.94,       # 【可调】上边界
#     hspace=0.32,    # 【可调】垂直间距
#     wspace=0.30,    # 【可调】水平间距
# )

# ══════════════════════════════════════════════════════════════════════
# 13. 保存 & 预览
# ══════════════════════════════════════════════════════════════════════

# ── 保存图片 ──
save_fig(fig, OUTPUT_PATH, OUTPUT_TIF, dpi=EXPORT_DPI)

# ── 交互式预览（开发时取消下行注释；确认后重新注释再运行） ──
# plt.show()  # ← 取消注释以预览

plt.close(fig)
print("[DONE] 脚本执行完毕。")
