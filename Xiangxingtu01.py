#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制14个敏感参数的年际箱型图 (2020-2025)

布局: 4列×4行网格
  - 第一行 (0,0)(0,1) → 图例, (0,2)(0,3) → 图 a,b
  - 第二/三/四行 → 图 c–n (每行4个)
  - 仅最后一排显示 X 刻度 (竖写左倾30°)

SCI 规范版本 (2026-05-18):
  - 尺寸: 19×20 cm 整页组合图
  - DPI: 300 (开发) / 500 (导出)
  - 字体: Arial, 7-10 pt
  - 线宽: 0.8-1.2 pt
  - 配色: 标准箱型图配色 (中位线红 / 均值红 / 限值蓝 / 异常值灰)
  - 导出: PNG (预览) + TIFF LZW (投稿)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os


# ╔══════════════════════════════════════════════════════════╗
# ║              1. 全局尺寸 & 单位换算                       ║
# ╚══════════════════════════════════════════════════════════╝

CM2INCH = 1 / 2.54
# ↑ 厘米→英寸换算常数，所有物理尺寸都用 cm 定义

# ── 图片物理尺寸 ─────────────────────────────────────────
# 【可调】改这两个值控制整幅图的宽/高
FIG_WIDTH_CM  = 19     # 总宽度 (cm)。SCI 整页≤19, 单列≤9
FIG_HEIGHT_CM = 20     # 总高度 (cm)。根据行数、hspace、子图大小调整
#   经验值: 4行箱型图 → 16~22 cm 适中, 太高会浪费页面空间


# ╔══════════════════════════════════════════════════════════╗
# ║              2. 子图布局 (位置映射)                       ║
# ╚══════════════════════════════════════════════════════════╝

# 【可调】修改这个列表可以重新排列子图顺序
# 每个元组 = (行号, 列号)，对应网格中的格子 (4行×4列, 从0开始)
# 当前: 14个参数按 idx=0..13 顺序填入以下14个位置
POSITION_ORDER = [
    (0, 0), (0, 1),                     # 第一行 2 个 → 对应 params[0],params[1]
    (1, 0), (1, 1), (1, 2), (1, 3),     # 第二行 4 个 → params[2]...params[5]
    (2, 0), (2, 1), (2, 2), (2, 3),     # 第三行 4 个 → params[6]...params[9]
    (3, 0), (3, 1), (3, 2), (3, 3),     # 第四行 4 个 → params[10]...params[13]
]
# 注: (0,0) 和 (0,1) 未使用，留给图例


# ╔══════════════════════════════════════════════════════════╗
# ║              3. SCI 字体 & 线宽 (全局 rcParams)           ║
# ╚══════════════════════════════════════════════════════════╝

# ── 字体 ─────────────────────────────────────────────────
rcParams['font.family'] = 'Arial'        # 【SCI 标准，勿动】Arial/Helvetica/Times New Roman
rcParams['font.size']      = 9           # 【可调】全局基准字号。范围 8~10 pt

# ── 字号分级 ─────────────────────────────────────────────
#   font.size       = 9   基础字号（影响未显式设定大小的文本）
#   axes.titlesize  = 10  子图标题。范围 10~12 pt  【可调】
#   axes.labelsize  = 9   坐标轴标签（如 "Concentration (mg/L)"）。范围 8~10 pt  【可调】
#   xtick.labelsize = 7   X 轴刻度数字。范围 6~8 pt，不低于 6 pt  【可调】
#   ytick.labelsize = 7   Y 轴刻度数字。同上
#   legend.fontsize = 8   图例文字。【可调 7~9 pt】
rcParams['axes.titlesize']    = 10
rcParams['axes.labelsize']    = 9
rcParams['xtick.labelsize']   = 7
rcParams['ytick.labelsize']   = 7
rcParams['legend.fontsize']   = 8
rcParams['legend.title_fontsize'] = 9

# ── 线宽 ─────────────────────────────────────────────────
#   axes.linewidth      = 1.0  坐标轴脊线。范围 0.8~1.2 pt  【可调】
#   xtick.major.width   = 0.8  主刻度线宽
#   xtick.major.size    = 3    主刻度长度 (pt)。【可调 2~4】
#   xtick.minor.size    = 1.5  次刻度长度 (pt)
rcParams['axes.linewidth']     = 1.0
rcParams['xtick.major.width']  = 0.8
rcParams['ytick.major.width']  = 0.8
rcParams['xtick.major.size']   = 3
rcParams['ytick.major.size']   = 3
rcParams['xtick.minor.size']   = 1.5
rcParams['ytick.minor.size']   = 1.5

# ── DPI & 保存 ───────────────────────────────────────────
rcParams['figure.dpi']         = 300   # 【SCI 标准】屏幕/开发用 DPI。≥300
rcParams['savefig.dpi']        = 500   # 【SCI 标准】导出用 DPI。组合图≥500, 彩色≥300
rcParams['savefig.bbox']       = 'tight'      # 自动裁剪白边
rcParams['savefig.pad_inches'] = 0.05         # tight 模式下保留 0.05 英寸边距


# ╔══════════════════════════════════════════════════════════╗
# ║              4. 统一导出函数                              ║
# ╚══════════════════════════════════════════════════════════╝

def save_fig(fig, name, dpi=500):
    """
    同时输出 PNG (预览) + TIFF LZW (投稿) 

    参数:
        fig  : matplotlib Figure 对象
        name : 输出路径（不含扩展名），如 'D:/path/Figure_xxx'
        dpi  : 导出分辨率。【可调】500=组合图, 300=纯彩色, 1000=纯线条
    """
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{name}.tif", dpi=dpi, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f"  Saved: {name}.png / .tif")


# ╔══════════════════════════════════════════════════════════╗
# ║              5. 数据读取 & 预处理                         ║
# ╚══════════════════════════════════════════════════════════╝

data_path = r'D:\WQIPaper\basicData\SuzhouYuejian-预处理完毕.xlsx'
df = pd.read_excel(data_path)

# 日期解析 → 提取年份
df['采样日期'] = pd.to_datetime(df['采样日期'])
df['Year'] = df['采样日期'].dt.year

# 【可调】修改年份范围，例如只看 2021-2024
df = df[(df['Year'] >= 2020) & (df['Year'] <= 2025)]
years_list = list(range(2020, 2026))

print(f"Total records (2020-2025): {len(df)}")
print(f"Records per year:\n{df['Year'].value_counts().sort_index()}")


# ╔══════════════════════════════════════════════════════════╗
# ║              6. 参数定义                                  ║
# ╚══════════════════════════════════════════════════════════╝

# 格式: (Excel 中文列名, 英文标签(含单位), Y轴标签, 标准限值)
#
# 标准限值规则:
#   None        → 不画限值线（如水温、TOC 无国标限值）
#   float       → 画单条蓝色虚线（如 铅≤0.01 mg/L）
#   (lo, hi)    → 画上下限两条虚线（如 pH 6.5~8.5）
#
# 【可调】增删参数、修改限值、改英文标签、改 Y 轴标签
params = [
    ('铅',         'Pb',                              'Pb (mg/L)',  0.01),
    ('溶解性总固体', 'TDS',         'TDS (mg/L)',  1000),
    ('硝酸盐',      'Nitrate',                               'Nitrate (mg/L)',  10),
    ('总硬度',      'Total Hardness',                        'Total Hardness (mg/L)',  450),
    ('硫酸盐',      'Sulfate',                               r'SO$_{4}^{2-}$ (mg/L)',  250),
    ('高锰酸盐指数', 'COD$_{Mn}$',       'COD$_{Mn}$ (mg/L)',  3),
    ('氟化物',      'Fluoride',                              'Fluoride (mg/L)',  1),
    ('水温',        'Temperature',                     'Temperature (°C)',      None),
    ('pH值',        'pH',                                    'pH value',              (6.5, 8.5)),
    ('三卤甲烷',    'THMs',                'THMs',   1),
    ('氯化物',      'Chloride',                              'Chloride (mg/L)',  250),
    ('铝',          'Al',                         'Al (mg/L)',  0.2),
    ('游离氯',      'Free Chlorine',                         'Free Chlorine (mg/L)',  (0.3, 4)),
    ('总有机碳',    'TOC',            'TOC (mg/L)',  None),
]


# ╔══════════════════════════════════════════════════════════╗
# ║              7. 创建画布 & 隐藏空位                        ║
# ╚══════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(4, 4,
                         figsize=(FIG_WIDTH_CM * CM2INCH, FIG_HEIGHT_CM * CM2INCH))

# 左上角 (0,0) 和 (0,1) 留给图例
axes[0, 2].set_visible(False)
axes[0, 3].set_visible(False)


# ╔══════════════════════════════════════════════════════════╗
# ║              8. 配色方案                                  ║
# ║  【可调】所有颜色用十六进制 (HEX) 或 matplotlib 颜色名      ║
# ╚══════════════════════════════════════════════════════════╝

box_facecolor = '#B3D9E8'       # 箱体填充色（柔和浅蓝）。改成 '#E8D5B3'=米黄 等
box_edgecolor = '#4A90B4'       # 箱体边框色（中蓝）
median_color  = '#C0392B'       # 中位线颜色（深红）。改 '#2C3E50'=炭灰更低调
mean_color    = '#D62728'       # 年均值线+点（红）。改 '#E67E22'=橙
limit_color   = '#1F77B4'       # 标准限值虚线（蓝）。改 '#2ECC71'=绿
flier_color   = '#7F8C8D'       # 异常值散点（灰）。改 None 可隐藏异常值


# ╔══════════════════════════════════════════════════════════╗
# ║              9. 逐子图循环绘制                             ║
# ╚══════════════════════════════════════════════════════════╝

for idx, (col_name, eng_label, y_label, limit) in enumerate(params):
    row, col = POSITION_ORDER[idx]   # 从布局映射取行列
    ax = axes[row, col]

    # ── 9a. 按年份分组数据 ─────────────────────────────────
    data_by_year = []
    valid_years  = []
    for y in years_list:
        vals = df[df['Year'] == y][col_name].dropna().values
        if len(vals) > 0:
            data_by_year.append(vals)
            valid_years.append(y)

    if not data_by_year:
        print(f"WARNING: No data for {eng_label}")
        continue

    # ── 9b. 箱型图主体 ─────────────────────────────────────
    bp = ax.boxplot(
        data_by_year,
        positions=valid_years,
        widths=0.5,                         # 【可调】箱体宽度 (0.3~0.8)。年份多→缩窄
        patch_artist=True,                  # 允许填充箱体颜色
        showfliers=True,                    # 【可调】True=显示异常值, False=隐藏
        # --- 各部件样式 (均可逐项改颜色/线宽/透明度) ---
        boxprops=dict(
            facecolor=box_facecolor,        # 箱体填充色
            edgecolor=box_edgecolor,        # 箱体边框色
            linewidth=0.8),                 # 【可调】边框线宽
        medianprops=dict(
            color=median_color,             # 中位线颜色
            linewidth=1.3),                 # 【可调】中位线宽度
        whiskerprops=dict(
            color=box_edgecolor,            # 须线颜色
            linewidth=0.8),
        capprops=dict(
            color=box_edgecolor,            # 须端横线颜色
            linewidth=0.8),
        flierprops=dict(
            marker='o',                     # 【可调】异常值形状: 'o','x','+','d'
            markerfacecolor=flier_color,
            markersize=2.5,                 # 【可调】异常值点大小
            alpha=0.5,                      # 【可调】透明度 0~1
            linestyle='none'),              # 异常值无连线
    )

    # ── 9c. 年均值连线 (红色圆点+实线) ────────────────────
    annual_means = [np.mean(d) for d in data_by_year]
    ax.plot(valid_years, annual_means,
            'o-',                           # 圆点+实线
            color=mean_color,               # 【可调】线色
            linewidth=1.8,                  # 【可调】线宽
            markersize=2,                   # 【可调】圆点大小
            markerfacecolor='white',        # 圆点填充白色（镂空效果）
            markeredgewidth=1.5,            # 圆点边框宽度
            markeredgecolor=mean_color,     # 圆点边框颜色
            zorder=5,                       # 置于箱型图上层
            label='Annual mean')

    # ── 9d. 标准限值线 (蓝色虚线) ──────────────────────────
    if limit is not None:
        if isinstance(limit, tuple):
            lo, hi = limit
            ax.axhline(y=lo,
                       color=limit_color, linestyle='--',   # 【可调】改 ':' ' -.' 等
                       linewidth=1.2, alpha=0.8)            # 【可调】线宽、透明度
            ax.axhline(y=hi,
                       color=limit_color, linestyle='--',   # 上限同理
                       linewidth=1.2, alpha=0.8,
                       label=f'Standard limit ({lo}–{hi})')
        else:
            ax.axhline(y=limit,
                       color=limit_color, linestyle='--',
                       linewidth=1.2, alpha=0.8,
                       label=f'Standard limit ({limit})')

    # ── 9e. 子图字母标注 (a)~(n) ───────────────────────────
    letter = chr(ord('a') + idx)
    ax.text(-0.08, 1.05, f'({letter})',
            transform=ax.transAxes,         # 相对于子图坐标 (0,0)=左下 (1,1)=右上
            fontsize=12,                     # 【可调】字母大小
            fontweight='bold',
            va='bottom', ha='left')
    # 位置微调: x=-0.08 向左偏, y=1.02 略高于顶部 → 改这两个值移位

    # ── 9f. 标题 & Y轴标签 ─────────────────────────────────
    ax.set_title(eng_label,
                 fontsize=10,               # 【可调】标题字号 10~12 pt
                 fontweight='bold',
                 pad=6)                     # 【可调】标题与图的间距 (pt)
    ax.set_ylabel(y_label,
                  fontsize=9)               # 【可调】Y轴标签字号

    # ── 9g. X 轴 (仅最后一行显示, 竖写左倾30°) ─────────────
    ax.set_xticks(years_list)
    # X 轴范围: 左边留 0.8 年空白, 右边也留 → 【可调】改 0.8 收紧/放宽
    ax.set_xlim(years_list[0] - 0.8, years_list[-1] + 0.8)

    if row < 3:
        # 前三排: 隐藏 X 刻度标签
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
    else:
        # 最后一排: 显示年份, 竖写左倾 30°
        ax.set_xticklabels([str(y) for y in years_list],
                           fontsize=7,          # 【可调】年份字号
                           rotation=60,         # 【可调】旋转角度: 60=左倾30°(从垂直), 0=水平, 90=纯垂直
                           ha='right')          # 对齐方式: 'right'配合 rotation 用

    # ── 9h. 刻度样式 ───────────────────────────────────────
    ax.tick_params(axis='both', labelsize=7.5)  # 【可调】刻度数字大小

    # ── 9i. Y 轴网格线 ─────────────────────────────────────
    ax.yaxis.grid(True,
                  linestyle=':',              # 【可调】'--'=虚线, ':'=点线, '-'=实线
                  alpha=0.7,                  # 【可调】透明度
                  linewidth=0.5)              # 【可调】网格线宽
    ax.set_axisbelow(True)                    # 网格置于数据下方（不遮挡）

    # ── 9j. 科学计数法 ─────────────────────────────────────
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # scilimits: 数值超出 (-2, 4) 范围时自动切换科学计数
    #   即 <10⁻² 或 >10⁴ 时才用科学计数 → 普通数值保持普通格式
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))

    # ── 9k. 四边脊线 (SCI 四方框风格) ──────────────────────
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    # 右轴不留刻度数字，只保留脊线框
    ax.tick_params(axis='y', which='both', right=False, labelright=False)


# ╔══════════════════════════════════════════════════════════╗
# ║             10. 图例                                     ║
# ╚══════════════════════════════════════════════════════════╝

# 【可调】增删/修改图例条目。三个元素对应箱体/均值线/限值线
legend_elements = [
    Patch(facecolor=box_facecolor, edgecolor=box_edgecolor, linewidth=0.8,
          label='Box plot (IQR + median)'),
    Line2D([0], [0], color=mean_color, linewidth=1.8, marker='o', markersize=5,
           markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=mean_color,
           label='Annual mean'),
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=1.2,
           label='Standard limit'),
]

fig.legend(
    handles=legend_elements,
    loc='upper left',                       # 定位锚点: 'upper left'=图例左上角对齐锚点
    bbox_to_anchor=(0.55, 0.97),            # 【可调】锚点在 figure 中的坐标 (x,y)。
                                            #   x=0.07=左边距, y=0.97=接近顶部
                                            #   减小 x→左移, 减小 y→下移
    bbox_transform=fig.transFigure,         # 相对于整个 figure（非子图）
    frameon=False,                           # 【可调】False=去掉图例边框
    fancybox=False,                         # False=直角框, True=圆角框
    edgecolor='#333333',                    # 图例边框颜色
    facecolor='white',                      # 图例背景色
    framealpha=0.95,                        # 图例背景透明度
    ncol=1,                                 # 【可调】列数: 3=水平排列三个条目
    fontsize=12,                             # 【可调】图例文字大小
    title='Legend',
    title_fontsize=14,                       # 【可调】图例标题大小
)


# ╔══════════════════════════════════════════════════════════╗
# ║             11. 整体边距 & 子图间距 (核心微调区!)          ║
# ╚══════════════════════════════════════════════════════════╝

plt.subplots_adjust(
    # ── 四周边距 (0~1, figure 的比例) ─────────────────────
    left   = 0.07,      # 【可调】左边距。增大→Y轴标签不会被裁; 减小→图更宽
    right  = 0.96,      # 【可调】右边距。增大→右侧留白多
    bottom = 0.04,      # 【可调】底边距。增大→给 X 轴标签腾空间
    top    = 0.96,      # 【可调】顶边距。增大→给标题/图例腾空间

    # ── 子图间距 (hspace/wspace = 子图高/宽的倍数) ───────
    hspace = 0.30,      # 【可调】垂直间距。0.2=紧凑, 0.4=稀疏, >0.5=很松散
                        #   值越大→子图越矮（因为空间分给间距）
    wspace = 0.45,      # 【可调】水平间距。0.25=紧凑, 0.5=宽松
                        #   值越大→子图越窄
)
# 微调思路:
#   1. 如果子图太挤 → 增大 hspace/wspace 或减小 FIG_WIDTH/HEIGHT
#   2. 如果 Y 轴标签被切 → 增大 left
#   3. 如果底部年份被切 → 增大 bottom
#   4. 如果子图太扁 → 增大 FIG_HEIGHT 或减小 hspace


# ╔══════════════════════════════════════════════════════════╗
# ║             12. 保存 & 预览                               ║
# ╚══════════════════════════════════════════════════════════╝

output_dir = r'D:\WQIPaper\DataAnalytics'
os.makedirs(output_dir, exist_ok=True)

fig_path_base = os.path.join(output_dir, 'Figure_14params_boxplot')
print("\nExporting figure...")
save_fig(fig, fig_path_base, dpi=500)     # 【可调】改 dpi 控制导出分辨率

# ── 交互预览开关 ──────────────────────────────────────────
# 【用法】取消下面这行的注释来弹出预览窗口，确认后在行首加回 #
# plt.show()

plt.close(fig)
print("Done!")
