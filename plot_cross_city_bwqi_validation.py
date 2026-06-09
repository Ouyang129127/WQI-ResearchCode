# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Cross-city BWQI validation figures                                         ║
║  ========================================================================  ║
║  依赖: pandas, numpy, matplotlib, openpyxl, pillow                           ║
║  图件: Fig.7 框架过程验证; Fig.8 BWQI输出与本地化权重解释                    ║
║  输出: PNG预览 + TIFF投稿 + PDF矢量备份                                      ║
║  路径: D:\WQIPaper\DataAnalytics                                            ║
║                                                                              ║
║  工作流:                                                                     ║
║    1. 从苏州、泉州、潍坊 pipeline 文件读取 BWQI、权重、验证统计              ║
║    2. 用同一 WQM 公式补算苏州 AHP/RF-CRITIC/Nash 三方案 BWQI                ║
║    3. 绘制跨城方法收敛、Spearman ρ、CV、BWQI分布和权重热力图                ║
║                                                                              ║
║  注释规范:                                                                   ║
║    - 所有主要参数用“【可调】”标记                                             ║
║    - 每个关键代码块用双线框标题分隔                                           ║
║    - 坐标标签中上下标使用 matplotlib mathtext，例如 NO$_{3}^{-}$ (mg/L)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第1节 · 导入依赖与全局路径                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 【可调】批量出图用 Agg；交互预览时可改为默认后端
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CM2INCH = 1 / 2.54
ROOT = Path(r"D:\WQIPaper\DataAnalytics")  # 【可调】输出目录与主数据目录
VAL_DIR = ROOT / "验证集"                    # 【可调】验证集目录
EXPORT_DPI = 500                            # 【可调 300~1000】组合图建议 500 dpi


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第2节 · SCI作图风格参数                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MORANDI = {
    "suzhou": "#9CAF88",    # 【可调】苏州: 鼠尾草绿
    "quanzhou": "#8FA7B7",  # 【可调】泉州: 灰蓝
    "weifang": "#C49A8C",   # 【可调】潍坊: 陶土玫瑰
    "ahp": "#C49A8C",       # 【可调】AHP: 专家主观权重
    "rfc": "#9CAF88",       # 【可调】RF-CRITIC: 数据客观权重
    "nash": "#8FA7B7",      # 【可调】Nash: 融合权重
    "grey": "#B8A99A",
    "dark": "#3E3E3E",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,              # 【可调 8~10】全局字号
    "axes.titlesize": 10,        # 【可调 9~11】子图标题字号
    "axes.labelsize": 9,         # 【可调 8~10】坐标轴标签字号
    "xtick.labelsize": 7,        # 【可调 6~8】刻度字号
    "ytick.labelsize": 7,        # 【可调 6~8】刻度字号
    "legend.fontsize": 7,        # 【可调 7~9】图例字号
    "axes.linewidth": 0.9,       # 【可调 0.8~1.2】坐标轴脊线线宽
    "grid.linewidth": 0.4,       # 【可调 0.3~0.5】网格线线宽
    "lines.linewidth": 1.4,      # 【可调 1.0~2.0】主数据线线宽
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第3节 · 指标名称标准化与SCI标签                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 说明: canonical name 用于跨城市合并; label 用于图中显示。
# 用户特别提醒的上下标在这里统一处理，例如 NO$_{3}^{-}$ (mg/L)。
PARAM_LABELS = {
    "Aluminum": "Al (mg/L)",
    "Chlorate": "ClO$_{3}^{-}$ (mg/L)",
    "Chloride": "Cl$^{-}$ (mg/L)",
    "Chloroform": "CHCl$_{3}$ (mg/L)",
    "CODmn": "COD$_{Mn}$ (mg/L)",
    "Fluoride": "F$^{-}$ (mg/L)",
    "Free_Chlorine": "Free Cl (mg/L)",
    "Lead": "Pb (mg/L)",
    "Nitrate": "NO$_{3}^{-}$ (mg/L)",
    "pH": "pH",
    "Sulfate": "SO$_{4}^{2-}$ (mg/L)",
    "TAB": "TAB (CFU/mL)",
    "TDS": "TDS (mg/L)",
    "TH": "TH (mg/L)",
    "THMs": "THMs (mg/L)",
    "TOC": "TOC (mg/L)",
    "Turbidity": "Turbidity (NTU)",
    "Water_Temperature": "Temperature (°C)",
}

CANONICAL_MAP = {
    "Aluminum (Al)": "Aluminum",
    "Aluminum": "Aluminum",
    "Chlorate": "Chlorate",
    "Chloride": "Chloride",
    "Cl⁻": "Chloride",
    "Chloroform": "Chloroform",
    "CODMn": "CODmn",
    "CODmn": "CODmn",
    "Permanganate Index (COD_Mn)": "CODmn",
    "F⁻": "Fluoride",
    "Fluoride": "Fluoride",
    "Free Cl": "Free_Chlorine",
    "Free Chlorine": "Free_Chlorine",
    "Free_Chlorine": "Free_Chlorine",
    "Lead (Pb)": "Lead",
    "NO₃⁻": "Nitrate",
    "Nitrate": "Nitrate",
    "pH": "pH",
    "SO₄²⁻": "Sulfate",
    "Sulfate": "Sulfate",
    "TAB": "TAB",
    "TDS": "TDS",
    "Total Dissolved Solids (TDS)": "TDS",
    "TH": "TH",
    "Total Hardness": "TH",
    "Trihalomethanes (THMs)": "THMs",
    "TOC": "TOC",
    "Total Organic Carbon (TOC)": "TOC",
    "Turbidity": "Turbidity",
    "Water Temperature": "Water_Temperature",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第4节 · 通用工具函数                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def canonical_name(name: str) -> str:
    """将不同城市/文件中的指标名称映射为统一英文代号。"""
    text = str(name).replace("SI_", "").strip()
    return CANONICAL_MAP.get(text, text)


def canonical_series(series: pd.Series) -> pd.Series:
    """把权重 Series 的索引统一成 canonical name，并对重复项求和。"""
    out = series.copy()
    out.index = [canonical_name(i) for i in out.index]
    out = out.groupby(out.index).sum()
    total = out.sum()
    return out / total if total > 0 else out


def weighted_quadratic_mean(si_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """WQM = sqrt(sum(w_i * SI_i^2)); 所有权重先按可用指标重新归一化。"""
    common = [c for c in si_df.columns if c in weights.index]
    if not common:
        raise ValueError("No common indicators between sub-index table and weights.")
    w = weights.loc[common].astype(float)
    w = w / w.sum()
    return np.sqrt((si_df[common].astype(float).pow(2) * w).sum(axis=1))


def save_figure(fig: plt.Figure, stem: str) -> None:
    """同时保存 PNG、TIFF(LZW) 和 PDF，便于预览、投稿和后期编辑。"""
    png = ROOT / f"{stem}.png"
    tif = ROOT / f"{stem}.tif"
    pdf = ROOT / f"{stem}.pdf"
    fig.savefig(png, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(tif, dpi=EXPORT_DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[OK] saved: {png.name}, {tif.name}, {pdf.name}")


def panel_label(ax: plt.Axes, label: str) -> None:
    """在子图左上角添加 (a), (b) 等标签。"""
    ax.text(
        -0.10, 1.06, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10,              # 【可调 8~11】子图标签字号
        fontweight="bold",
        color=MORANDI["dark"],
    )


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """统一去掉上/右脊线并添加浅灰网格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle=":", alpha=0.45, zorder=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第5节 · 数据读取与计算                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[1/4] Loading cross-city BWQI data ...")

# —— 5.1 苏州: 读取 DLIR 子指数和三套权重，补算 AHP/RF-CRITIC/Nash BWQI ——
sz_si_raw = pd.read_excel(ROOT / "BWQI_Pipeline_Output.xlsx", sheet_name="DLIR子指数")
sz_si = sz_si_raw.drop(columns=[c for c in ["ID", "Date"] if c in sz_si_raw.columns])
sz_si.columns = [canonical_name(c) for c in sz_si.columns]

sz_ahp_df = pd.read_excel(ROOT / "AHP_Weights.xlsx")
sz_ahp = canonical_series(sz_ahp_df.set_index("Indicator")["AHP_Global_Weight"])

sz_rfc_df = pd.read_excel(ROOT / "RF_CRITIC_Fusion_Weights.xlsx")
sz_rfc = canonical_series(sz_rfc_df.set_index("Indicators")["Multiplicative_raw"])

sz_nash_df = pd.read_excel(ROOT / "Final_Combine_Wights.xlsx")
sz_nash = canonical_series(sz_nash_df.set_index("Indicator")["ConstrainedQP_Nash_fmt"])

sz_bwqi = pd.DataFrame({
    "City": "Suzhou",
    "AHP": weighted_quadratic_mean(sz_si, sz_ahp),
    "RF-CRITIC": weighted_quadratic_mean(sz_si, sz_rfc),
    "Nash": weighted_quadratic_mean(sz_si, sz_nash),
})

# —— 5.2 泉州: 读取 pipeline 已输出的三方案 BWQI 与 JSON 权重/验证统计 ——
qz_json = json.loads((VAL_DIR / "quanzhou_bwqi_results.json").read_text(encoding="utf-8"))
qz_bwqi_raw = pd.read_excel(VAL_DIR / "Quanzhou_BWQI_Pipeline_Output.xlsx", sheet_name="BWQI结果")
qz_bwqi = pd.DataFrame({
    "City": "Quanzhou",
    "AHP": qz_bwqi_raw["BWQI_AHP"],
    "RF-CRITIC": qz_bwqi_raw["BWQI_RFC"],
    "Nash": qz_bwqi_raw["BWQI_Nash"],
})
qz_nash = canonical_series(pd.Series(qz_json["weights"]["nash"]))

# —— 5.3 潍坊: 读取 pipeline 已输出的三方案 BWQI 与 JSON 权重/验证统计 ——
wf_json = json.loads((VAL_DIR / "weifang_bwqi_results.json").read_text(encoding="utf-8"))
wf_bwqi_raw = pd.read_excel(VAL_DIR / "Weifang_BWQI_Pipeline_Output.xlsx", sheet_name="BWQI结果")
wf_bwqi = pd.DataFrame({
    "City": "Weifang",
    "AHP": wf_bwqi_raw["BWQI_AHP"],
    "RF-CRITIC": wf_bwqi_raw["BWQI_RF_CRITIC"],
    "Nash": wf_bwqi_raw["BWQI_Nash"],
})
wf_nash = canonical_series(pd.Series(wf_json["weights"]["nash"]))

# —— 5.4 汇总统计: 每城每方案 mean/std/CV ——
all_bwqi = pd.concat([sz_bwqi, qz_bwqi, wf_bwqi], ignore_index=True)
long_bwqi = all_bwqi.melt(id_vars="City", var_name="Scheme", value_name="BWQI")

summary = (
    long_bwqi
    .groupby(["City", "Scheme"])["BWQI"]
    .agg(["mean", "std"])
    .reset_index()
)
summary["cv_pct"] = summary["std"] / summary["mean"] * 100
summary.to_csv(ROOT / "Fig7_Fig8_cross_city_summary.csv", index=False, encoding="utf-8-sig")

city_meta = pd.DataFrame({
    "City": ["Suzhou", "Quanzhou", "Weifang"],
    "Samples": [len(sz_bwqi), qz_json["samples"], wf_json["data_summary"]["n_samples"]],
    "Parameters": [len(sz_si.columns), qz_json["params"], wf_json["data_summary"]["n_parameters"]],
    "Spearman_rho": [0.91, qz_json["validation"]["spearman_rho"], wf_json["correlation"]["spearman_rho"]],
    "Eclipsing_detected": [14, qz_json["validation"]["eclipsing_detectable"], wf_json["eclipsing"]["detected"]],
    "Eclipsing_total": [14, 10, wf_json["eclipsing"]["total"]],
    "WQM_Arith_ratio": [0.51, qz_json["validation"]["eclipsing_wqm_arith_ratio"], wf_json["eclipsing"]["wqm_arith_ratio"]],
})
city_meta["Eclipsing_rate"] = city_meta["Eclipsing_detected"] / city_meta["Eclipsing_total"] * 100
city_meta.to_csv(ROOT / "Fig7_Fig8_cross_city_validation_metrics.csv", index=False, encoding="utf-8-sig")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第6节 · Fig.7: 过程可迁移性验证                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[2/4] Drawing Fig.7 ...")

fig7 = plt.figure(figsize=(19 * CM2INCH, 15.2 * CM2INCH))  # 【可调】整页宽19 cm; 增高可缓解标题/矩阵拥挤
gs7 = gridspec.GridSpec(
    2, 2,
    figure=fig7,
    height_ratios=[1.02, 1.08],  # 【可调】增大上排比例→决策矩阵更醒目
    hspace=0.46,                 # 【可调】增大→上下子图间距变大
    wspace=0.32,                 # 【可调】增大→左右子图间距变大
)

# ═══ Fig.7(a) · 三城决策路径收敛矩阵 ═══
ax = fig7.add_subplot(gs7[0, :])
panel_label(ax, "(a)")
ax.text(
    0.0, 1.03,
    "Independent decision paths converge to the same BWQI configuration",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=10,                 # 【可调 9~11】面板标题字号
    fontweight="bold",
)

steps = ["Sub-index", "Weighting", "Aggregation", "Validation"]
chosen = ["DLIR", "AHP + RF-CRITIC → Nash", "WQM", "Sensitivity + Eclipsing + CV"]
candidates = [
    "DLIR / EF / ISF",
    "AHP / RF-CRITIC / Nash",
    "WQM / LQM / SWM",
    "ρ / CV / Eclipsing",
]
cities = ["Suzhou", "Quanzhou", "Weifang"]
city_colors = [MORANDI["suzhou"], MORANDI["quanzhou"], MORANDI["weifang"]]

ax.set_xlim(-0.82, len(steps) - 0.28)
ax.set_ylim(-0.55, len(cities) + 0.32)
ax.axis("off")

for j, step in enumerate(steps):
    ax.text(j, len(cities) + 0.02, step, ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.text(j, len(cities) - 0.18, candidates[j], ha="center", va="top", fontsize=6.4, color="#666666")

for i, (city, color) in enumerate(zip(cities, city_colors)):
    y = len(cities) - 1 - i
    ax.text(-0.48, y, city, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)
    for j, text in enumerate(chosen):
        rect = Rectangle(
            (j - 0.40, y - 0.21), 0.80, 0.42,
            facecolor=color if j == 0 or j == 2 else "#F6F6F3",
            edgecolor=color,
            linewidth=0.8,           # 【可调 0.6~1.2】矩阵框线
            alpha=0.75 if j == 0 or j == 2 else 1.0,
        )
        ax.add_patch(rect)
        ax.text(j, y, text, ha="center", va="center", fontsize=6.1, color=MORANDI["dark"])
        if j < len(steps) - 1:
            ax.annotate(
                "", xy=(j + 0.51, y), xytext=(j + 0.43, y),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#777777"),
            )

ax.text(
    1.5, -0.38,
    "The selected formulas are not transferred from Suzhou; each city re-runs the same comparison-and-selection workflow.",
    ha="center", va="center", fontsize=7.2, color="#555555", style="italic",
)

# ═══ Fig.7(b) · Spearman ρ 专家-数据一致性梯度 ═══
ax = fig7.add_subplot(gs7[1, 0])
panel_label(ax, "(b)")
rho = city_meta.set_index("City").loc[cities, "Spearman_rho"].to_numpy()
y = np.arange(len(cities))
ax.axvspan(0.0, 0.3, color="#D9B2A9", alpha=0.20, zorder=0)
ax.axvspan(0.3, 0.7, color="#D8CC9A", alpha=0.18, zorder=0)
ax.axvspan(0.7, 1.0, color="#B9C8AA", alpha=0.22, zorder=0)
ax.barh(y, rho, height=0.48, color=city_colors, edgecolor="white", linewidth=0.4, alpha=0.92, zorder=3)
for yi, val in zip(y, rho):
    ax.text(val + 0.025, yi, f"ρ = {val:.3f}", va="center", ha="left", fontsize=7.3, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(cities, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("Spearman rank correlation between AHP and RF-CRITIC weights")
ax.set_title("Expert-data relationship diagnosed by ρ", loc="left", fontweight="bold")
ax.text(0.15, -0.62, "Conflict", ha="center", fontsize=6.5, color="#8A5148")
ax.text(0.50, -0.62, "Transition", ha="center", fontsize=6.5, color="#8A7A35")
ax.text(0.85, -0.62, "Consensus", ha="center", fontsize=6.5, color="#5F7657")
clean_axes(ax, "x")

# ═══ Fig.7(c) · 三方案 CV 对比 ═══
ax = fig7.add_subplot(gs7[1, 1])
panel_label(ax, "(c)")
x = np.arange(len(cities))
bar_w = 0.23  # 【可调 0.18~0.28】单柱宽度
schemes = ["AHP", "RF-CRITIC", "Nash"]
scheme_colors = [MORANDI["ahp"], MORANDI["rfc"], MORANDI["nash"]]
for k, (scheme, color) in enumerate(zip(schemes, scheme_colors)):
    vals = (
        summary[summary["Scheme"] == scheme]
        .set_index("City")
        .loc[cities, "cv_pct"]
        .to_numpy()
    )
    ax.bar(x + (k - 1) * bar_w, vals, width=bar_w, color=color, edgecolor="white", linewidth=0.4, label=scheme, zorder=3)
    for xi, val in zip(x + (k - 1) * bar_w, vals):
        ax.text(xi, val + 0.25, f"{val:.2f}", ha="center", va="bottom", fontsize=6.4, rotation=0)
ax.set_xticks(x)
ax.set_xticklabels(cities, fontweight="bold")
ax.set_ylabel("CV of BWQI (%)")
ax.set_title("Stability of BWQI outputs under alternative weights", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_ylim(0, max(summary["cv_pct"]) * 1.22)
clean_axes(ax, "y")

fig7.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.10)  # 【可调】边距，增大bottom→给X轴标签更多空间
save_figure(fig7, "Fig7_CrossCity_Framework_Validation")
plt.close(fig7)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第7节 · Fig.8: BWQI输出与权重本地化解释                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[3/4] Drawing Fig.8 ...")

fig8 = plt.figure(figsize=(19 * CM2INCH, 18.6 * CM2INCH))  # 【可调】整页宽19 cm; 增高可缓解热力图标签拥挤
gs8 = gridspec.GridSpec(
    2, 2,
    figure=fig8,
    height_ratios=[1.05, 1.12],  # 【可调】增大下排比例→热力图更舒展
    hspace=0.46,                 # 【可调】上下子图间距
    wspace=0.48,                 # 【可调】左右子图间距
)

# ═══ Fig.8(a) · 三城 Nash BWQI 分布 ═══
ax = fig8.add_subplot(gs8[0, 0])
panel_label(ax, "(a)")
nash_data = [all_bwqi.loc[all_bwqi["City"] == c, "Nash"].dropna().to_numpy() for c in cities]
parts = ax.violinplot(nash_data, positions=np.arange(len(cities)), widths=0.72, showmeans=False, showextrema=False)
for body, color in zip(parts["bodies"], city_colors):
    body.set_facecolor(color)
    body.set_edgecolor("white")
    body.set_alpha(0.45)  # 【可调 0.25~0.60】小提琴填充透明度
bp = ax.boxplot(
    nash_data,
    positions=np.arange(len(cities)),
    widths=0.28,              # 【可调 0.20~0.36】箱线图宽度
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color=MORANDI["dark"], linewidth=1.0),
    boxprops=dict(linewidth=0.8),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
)
for patch, color in zip(bp["boxes"], city_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.88)
    patch.set_edgecolor("white")
for i, values in enumerate(nash_data):
    rng = np.random.default_rng(20260609 + i)
    jitter = rng.normal(0, 0.035, size=len(values))
    ax.scatter(np.full(len(values), i) + jitter, values, s=5, alpha=0.20, color=MORANDI["dark"], linewidths=0, zorder=2)
    ax.text(i, np.nanmax(values) + 0.9, f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}", ha="center", fontsize=6.8)
ax.set_xticks(np.arange(len(cities)))
ax.set_xticklabels(cities, fontweight="bold")
ax.set_ylabel("Nash BWQI")
ax.set_title("Nash BWQI distributions", loc="left", fontweight="bold")
clean_axes(ax, "y")

# ═══ Fig.8(b) · 三城 Nash 权重热力图 ═══
ax = fig8.add_subplot(gs8[0, 1])
panel_label(ax, "(b)")
weight_df = pd.DataFrame({
    "Suzhou": sz_nash,
    "Quanzhou": qz_nash,
    "Weifang": wf_nash,
}).fillna(0.0)
weight_df["max_weight"] = weight_df.max(axis=1)
weight_df = weight_df.sort_values("max_weight", ascending=True).drop(columns="max_weight")

heat_colors = ["#F7F5F0", "#D7C7B4", "#B8A99A", "#8FA7B7", "#667C8F"]
cmap = LinearSegmentedColormap.from_list("bwqi_morandi_heat", heat_colors)
im = ax.imshow(weight_df.values * 100, aspect="auto", cmap=cmap, vmin=0, vmax=max(25, weight_df.values.max() * 100))
ax.set_xticks(np.arange(len(cities)))
ax.set_xticklabels(cities, fontweight="bold")
ax.set_yticks(np.arange(len(weight_df.index)))
ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in weight_df.index], fontsize=6.6)
ax.set_title("Localized Nash weight structures", loc="left", fontweight="bold")
for i in range(weight_df.shape[0]):
    for j in range(weight_df.shape[1]):
        val = weight_df.iloc[i, j] * 100
        if val >= 6.0:  # 【可调 4~8】只标注较大权重，避免热力图拥挤
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.8, color="white" if val > 14 else MORANDI["dark"])
cbar = fig8.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Nash weight (%)", fontsize=8)
cbar.ax.tick_params(labelsize=6)

# ═══ Fig.8(c) · 三方案 BWQI均值±标准差 ═══
ax = fig8.add_subplot(gs8[1, 0])
panel_label(ax, "(c)")
x = np.arange(len(cities))
for k, (scheme, color) in enumerate(zip(schemes, scheme_colors)):
    stat = summary[summary["Scheme"] == scheme].set_index("City").loc[cities]
    xpos = x + (k - 1) * bar_w
    ax.bar(xpos, stat["mean"], width=bar_w, color=color, alpha=0.86, edgecolor="white", linewidth=0.4, label=scheme, zorder=3)
    ax.errorbar(xpos, stat["mean"], yerr=stat["std"], fmt="none", ecolor=MORANDI["dark"], elinewidth=0.7, capsize=2, capthick=0.7, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(cities, fontweight="bold")
ax.set_ylabel("BWQI (mean ± SD)")
ax.set_title("BWQI levels under alternative weights", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=3, loc="upper right")
clean_axes(ax, "y")

# ═══ Fig.8(d) · Eclipsing检出与WQM/Arithmetic比值 ═══
ax = fig8.add_subplot(gs8[1, 1])
panel_label(ax, "(d)")
rates = city_meta.set_index("City").loc[cities, "Eclipsing_rate"].to_numpy()
ratios = city_meta.set_index("City").loc[cities, "WQM_Arith_ratio"].to_numpy()
bars = ax.bar(x, rates, width=0.48, color=city_colors, edgecolor="white", linewidth=0.4, alpha=0.88, zorder=3)
for rect, rate, det, total in zip(bars, rates, city_meta["Eclipsing_detected"], city_meta["Eclipsing_total"]):
    ax.text(rect.get_x() + rect.get_width() / 2, rate + 2.0, f"{int(det)}/{int(total)}", ha="center", va="bottom", fontsize=7)
ax.set_ylim(0, 112)
ax.set_ylabel("Eclipsing detection rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(cities, fontweight="bold")
ax.set_title("Eclipsing response across cities", loc="left", fontweight="bold")
clean_axes(ax, "y")

ax2 = ax.twinx()
ax2.plot(x, ratios, marker="o", markersize=4.0, color=MORANDI["dark"], linewidth=1.2, label="WQM/Arithmetic")
for xi, ratio in zip(x, ratios):
    ax2.text(xi, ratio + 0.035, f"{ratio:.2f}", ha="center", va="bottom", fontsize=7, color=MORANDI["dark"])
ax2.set_ylim(0, 1.10)
ax2.set_ylabel("WQM/Arithmetic ratio")
ax2.spines["top"].set_visible(False)

legend_items = [
    Line2D([0], [0], color=MORANDI["dark"], marker="o", lw=1.2, markersize=4, label="WQM/Arithmetic ratio"),
]
ax2.legend(handles=legend_items, frameon=False, loc="lower right")

fig8.subplots_adjust(left=0.08, right=0.965, top=0.94, bottom=0.08)  # 【可调】整图边距
save_figure(fig8, "Fig8_CrossCity_BWQI_Localization")
plt.close(fig8)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第8节 · 输出简短数据核查报告                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[4/4] Writing figure data note ...")

note = ROOT / "Fig7_Fig8_cross_city_figure_note.md"
note.write_text(
    "# Fig.7-Fig.8 cross-city BWQI figure note\n\n"
    "Generated by `plot_cross_city_bwqi_validation.py`.\n\n"
    "## Data source\n\n"
    "- Suzhou: `BWQI_Pipeline_Output.xlsx`, `AHP_Weights.xlsx`, "
    "`RF_CRITIC_Fusion_Weights.xlsx`, `Final_Combine_Wights.xlsx`.\n"
    "- Quanzhou: `验证集/Quanzhou_BWQI_Pipeline_Output.xlsx`, "
    "`验证集/quanzhou_bwqi_results.json`.\n"
    "- Weifang: `验证集/Weifang_BWQI_Pipeline_Output.xlsx`, "
    "`验证集/weifang_bwqi_results.json`.\n\n"
    "## Important check\n\n"
    f"- Weifang in the current result files: n={wf_json['data_summary']['n_samples']}, "
    f"Spearman rho={wf_json['correlation']['spearman_rho']:.3f}, "
    f"Nash CV={summary[(summary.City == 'Weifang') & (summary.Scheme == 'Nash')]['cv_pct'].iloc[0]:.2f}%.\n"
    "- These values differ from an earlier discussion draft that mentioned n=65, "
    "rho=0.28 and CV=7.20%; the figures use the machine-readable result files.\n\n"
    "## Generated files\n\n"
    "- `Fig7_CrossCity_Framework_Validation.png/.tif/.pdf`\n"
    "- `Fig8_CrossCity_BWQI_Localization.png/.tif/.pdf`\n"
    "- `Fig7_Fig8_cross_city_summary.csv`\n"
    "- `Fig7_Fig8_cross_city_validation_metrics.csv`\n",
    encoding="utf-8",
)

print(f"[OK] saved: {note.name}")
print("[DONE] Cross-city BWQI validation figures generated.")
