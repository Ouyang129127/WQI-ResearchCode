#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制14个敏感参数的年际箱型图 (2020-2025)
Elsevier SCI 期刊标准格式: 4列×4行，最后一行2个图，右侧放图例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# ── Elsevier SCI 风格设置 ─────────────────────────────────
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['xtick.minor.size'] = 1.5
rcParams['ytick.minor.size'] = 1.5
rcParams['axes.titlesize'] = 9
rcParams['axes.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# ── 数据读取 ─────────────────────────────────────────────
data_path = r'D:\WQIPaper\basicData\SuzhouYuejian-预处理完毕.xlsx'
df = pd.read_excel(data_path)

# 解析日期，提取年份
df['采样日期'] = pd.to_datetime(df['采样日期'])
df['Year'] = df['采样日期'].dt.year

# 仅保留2020-2025
df = df[(df['Year'] >= 2020) & (df['Year'] <= 2025)]
years_list = list(range(2020, 2026))

print(f"Total records (2020-2025): {len(df)}")
print(f"Records per year:\n{df['Year'].value_counts().sort_index()}")

# ── 14个参数定义 ─────────────────────────────────────────
# 格式: (Excel中文列名, 英文标签(含单位), Y轴标签, 标准限值)
# 限值为 None 表示无标准; 为 tuple 表示范围(上下限); 为 float 表示单值上限
params = [
    ('铅',         'Lead (Pb)',                              'Concentration (mg/L)',  0.01),
    ('溶解性总固体', 'Total Dissolved Solids (TDS)',         'Concentration (mg/L)',  1000),
    ('硝酸盐',      'Nitrate',                               'Concentration (mg/L)',  10),
    ('总硬度',      'Total Hardness',                        'Concentration (mg/L)',  450),
    ('硫酸盐',      'Sulfate',                               'Concentration (mg/L)',  250),
    ('高锰酸盐指数', 'Permanganate Index (COD$_{Mn}$)',       'Concentration (mg/L)',  3),
    ('氟化物',      'Fluoride',                              'Concentration (mg/L)',  1),
    ('水温',        'Water Temperature',                     'Temperature (°C)',      None),
    ('pH值',        'pH',                                    'pH value',              (6.5, 8.5)),
    ('三卤甲烷',    'Trihalomethanes (THMs)',                'Dimensionless ratio',   1),
    ('氯化物',      'Chloride',                              'Concentration (mg/L)',  250),
    ('铝',          'Aluminum (Al)',                         'Concentration (mg/L)',  0.2),
    ('游离氯',      'Free Chlorine',                         'Concentration (mg/L)',  (0.3, 4)),
    ('总有机碳',    'Total Organic Carbon (TOC)',            'Concentration (mg/L)',  None),
]

# ── 创建图形: 4行×4列 ────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
# 隐藏最后两个空位（用于图例）
axes[3, 2].set_visible(False)
axes[3, 3].set_visible(False)

# 颜色方案
box_facecolor = '#B3D9E8'       # 柔和蓝
box_edgecolor = '#4A90B4'
median_color = '#C0392B'        # 中位线深红
mean_color = '#D62728'          # 年均值线红色
limit_color = '#1F77B4'         # 标准限值线蓝色
flier_color = '#7F8C8D'         # 异常值灰色

# ── 逐参数绘制 ───────────────────────────────────────────
for idx, (col_name, eng_label, y_label, limit) in enumerate(params):
    row, col = divmod(idx, 4)
    ax = axes[row, col]
    
    # 按年份分组数据
    data_by_year = []
    valid_years = []
    for y in years_list:
        vals = df[df['Year'] == y][col_name].dropna().values
        if len(vals) > 0:
            data_by_year.append(vals)
            valid_years.append(y)
    
    if not data_by_year:
        print(f"WARNING: No data for {eng_label}")
        continue
    
    # 箱型图
    bp = ax.boxplot(
        data_by_year,
        positions=valid_years,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=box_facecolor, edgecolor=box_edgecolor, linewidth=0.8),
        medianprops=dict(color=median_color, linewidth=1.3),
        whiskerprops=dict(color=box_edgecolor, linewidth=0.8),
        capprops=dict(color=box_edgecolor, linewidth=0.8),
        flierprops=dict(marker='o', markerfacecolor=flier_color,
                        markersize=2.5, alpha=0.5, linestyle='none')
    )
    
    # 年均值红色实线（带圆点）
    annual_means = []
    for i in range(len(data_by_year)):
        m = np.mean(data_by_year[i])
        annual_means.append(m)
    ax.plot(valid_years, annual_means, 'o-', color=mean_color, linewidth=1.8,
            markersize=5, markerfacecolor='white', markeredgewidth=1.5,
            markeredgecolor=mean_color, zorder=5, label='Annual mean')
    
    # 标准限值蓝色虚线
    if limit is not None:
        if isinstance(limit, tuple):
            lo, hi = limit
            ax.axhline(y=lo, color=limit_color, linestyle='--', linewidth=1.2, alpha=0.8)
            ax.axhline(y=hi, color=limit_color, linestyle='--', linewidth=1.2, alpha=0.8,
                       label=f'Standard limit ({lo}–{hi})')
        else:
            ax.axhline(y=limit, color=limit_color, linestyle='--', linewidth=1.2, alpha=0.8,
                       label=f'Standard limit ({limit})')
    
    # 子图字母标注 (a)~(n)
    letter = chr(ord('a') + idx)
    ax.text(-0.08, 1.02, f'({letter})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # 子图标题（英文指标名）
    ax.set_title(eng_label, fontsize=9, fontweight='bold', pad=8)
    
    # Y轴标签
    ax.set_ylabel(y_label, fontsize=8)
    
    # X轴: 只显示有数据的年份（2020-2025）
    ax.set_xticks(years_list)
    ax.set_xticklabels([str(y) for y in years_list], fontsize=7.5)
    ax.set_xlim(years_list[0] - 0.8, years_list[-1] + 0.8)
    
    # 刻度
    ax.tick_params(axis='both', labelsize=7.5)
    
    # 轻微网格（仅Y轴主刻度）
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Y轴科学计数（如需要）
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))
    
    # 顶部和右侧脊线去掉（SCI风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── 图例：放在右下角空位 ─────────────────────────────────
# 创建自定义图例句柄
legend_elements = [
    Patch(facecolor=box_facecolor, edgecolor=box_edgecolor, linewidth=0.8,
          label='Box plot (IQR + median)'),
    Line2D([0], [0], color=mean_color, linewidth=1.8, marker='o', markersize=5,
           markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=mean_color,
           label='Annual mean'),
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=1.2,
           label='Standard limit'),
]

# 在右下角区域内放置图例
# 使用一个隐藏轴的位置做图例容器
legend_ax = axes[3, 2]  # 虽然 visible=False, 但可以用其位置
# 改用 fig.legend 定位到右下角
fig.legend(
    handles=legend_elements,
    loc='lower right',
    bbox_to_anchor=(0.98, 0.08),
    bbox_transform=fig.transFigure,
    frameon=True,
    fancybox=False,
    edgecolor='#333333',
    facecolor='white',
    framealpha=0.95,
    ncol=1,
    fontsize=8.5,
    title='Legend',
    title_fontsize=9,
)

# ── 整体调整 ─────────────────────────────────────────────
plt.subplots_adjust(
    left=0.06, right=0.97,
    bottom=0.05, top=0.96,
    hspace=0.35, wspace=0.30
)

# ── 保存 ─────────────────────────────────────────────────
output_dir = r'D:\WQIPaper\DataAnalytics'
os.makedirs(output_dir, exist_ok=True)

# 保存为高质量 TIFF (Elsevier 首选) 和 PNG
fig_path_tif = os.path.join(output_dir, 'Figure_14params_boxplot.tif')
fig_path_png = os.path.join(output_dir, 'Figure_14params_boxplot.png')

fig.savefig(fig_path_tif, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(fig_path_png, format='png', dpi=600)
print(f"\nFigure saved to:\n  {fig_path_tif}\n  {fig_path_png}")

plt.close(fig)
print("Done!")
