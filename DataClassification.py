# -*- coding: utf-8 -*-
"""
============================================================
出厂水水质监测数据 — 层次聚类分析（Ward法 + 欧式距离）
============================================================

功能说明
--------
1. 读取预处理后的出厂水水质监测 Excel 数据
2. 自动检测并剔除方差为 0 的无效列（常数列）
3. 数据标准化（Z-score）
4. 层次聚类：Ward 连接法 + 欧氏距离
5. 最优聚类数判定（肘部法则 + 轮廓系数 + Calinski-Harabasz）
6. 生成谱系图 & 聚类散点图（PCA 降维可视化）
7. 导出 JSON 结果 + Markdown 分析报告
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# 强制 UTF-8 输出，解决 Windows GBK 终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免弹窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
INPUT_EXCEL = r"D:\WQIPaper\basicData\SuzhouYuejian-预处理完毕.xlsx"
OUTPUT_DIR = Path(r"D:\WQIPaper\DataAnalytics\AnalyticResults")
SCRIPT_DIR = Path(__file__).resolve().parent

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("出厂水水质监测数据 — 层次聚类分析")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1/7] 加载数据 ...")
df_raw = pd.read_excel(INPUT_EXCEL)
print(f"  原始数据维度: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
print(f"  列名示例: {df_raw.columns[:5].tolist()} ...")

# 分离 ID 列与指标列
id_col = df_raw.columns[0]
ids = df_raw[id_col].astype(str).tolist()
df_params = df_raw.drop(columns=[id_col])
param_names = df_params.columns.tolist()
print(f"  ID 列: '{id_col}'")
print(f"  指标列 ({len(param_names)} 个): {param_names[:5]} ...")


# ============================================================
# 2. 数据预处理 —— 剔除无效列 & 处理缺失值
# ============================================================
print("\n[2/7] 数据预处理 ...")

# 检测方差为 0（或接近 0）的列
variances = df_params.var(numeric_only=True)
zero_var_cols = variances[variances < 1e-12].index.tolist()
if zero_var_cols:
    print(f"  ⚠ 剔除 {len(zero_var_cols)} 个常数/近零方差列: {zero_var_cols}")
    df_params = df_params.drop(columns=zero_var_cols)
else:
    print("  ✓ 无常数/近零方差列")

# 检测缺失值
missing = df_params.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"  ⚠ 发现缺失值，使用列均值填充:")
    for col, cnt in missing_cols.items():
        print(f"      {col}: {cnt} 个缺失")
    df_params = df_params.fillna(df_params.mean(numeric_only=True))
else:
    print("  ✓ 无缺失值")

# 仅保留数值列
df_params = df_params.select_dtypes(include=[np.number])
active_params = df_params.columns.tolist()
n_params = len(active_params)
n_samples = len(ids)
print(f"  有效数据维度: {n_samples} 行 × {n_params} 列")


# ============================================================
# 3. 数据标准化
# ============================================================
print("\n[3/7] 数据标准化（Z-score）...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_params)
print(f"  标准化完成，shape: {X_scaled.shape}")


# ============================================================
# 4. 层次聚类（Ward 法 + 欧氏距离）
# ============================================================
print("\n[4/7] 层次聚类（Ward 连接 + 欧氏距离）...")

# 计算距离矩阵（欧氏距离）并执行 Ward 连接聚类
Z = linkage(X_scaled, method="ward", metric="euclidean")
print(f"  聚类链接矩阵 shape: {Z.shape}")
print(f"  样本间最小距离: {Z[0, 2]:.4f}")
print(f"  样本间最大距离: {Z[-1, 2]:.4f}")


# ============================================================
# 5. 最优聚类数判定
# ============================================================
print("\n[5/7] 最优聚类数判定 ...")

# 搜索范围: 2 ~ min(15, n_samples-1)
max_k = min(15, n_samples - 1)
k_range = list(range(2, max_k + 1))

silhouette_scores = []
ch_scores = []
inertias = []

for k in k_range:
    labels = fcluster(Z, k, criterion="maxclust")
    # 轮廓系数
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = -1
    silhouette_scores.append(sil)
    # Calinski-Harabasz 指数
    if len(set(labels)) > 1:
        ch = calinski_harabasz_score(X_scaled, labels)
    else:
        ch = -1
    ch_scores.append(ch)
    # 簇内离差平方和 (inertia / within-cluster sum of squares)
    inertia = 0.0
    for lbl in set(labels):
        cluster_points = X_scaled[labels == lbl]
        centroid = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    inertias.append(inertia)

# --- 综合判定 ---
# 规则 1: 轮廓系数最大对应的 k
best_k_sil = k_range[np.argmax(silhouette_scores)]
# 规则 2: CH 指数最大对应的 k
best_k_ch = k_range[np.argmax(ch_scores)]
# 规则 3: 肘部法则 — 计算二阶差分拐点
inertia_deltas = np.diff(inertias)
inertia_delta2 = np.diff(inertia_deltas)
# 找二阶差分最大处（曲率最大，即肘部）
if len(inertia_delta2) > 0:
    elbow_idx = np.argmax(np.abs(inertia_delta2)) + 1  # +1 因为 diff 减小了维度
    best_k_elbow = k_range[elbow_idx]
else:
    best_k_elbow = k_range[0]

# 综合投票
k_candidates = {}
for k in [best_k_sil, best_k_ch, best_k_elbow]:
    k_candidates[k] = k_candidates.get(k, 0) + 1
# 平局时优先选较小的 k（更可解释）
optimal_k = max(k_candidates, key=lambda x: (k_candidates[x], -x))

print(f"  轮廓系数最佳 k = {best_k_sil}  (silhouette = {silhouette_scores[k_range.index(best_k_sil)]:.4f})")
print(f"  CH 指数最佳 k   = {best_k_ch}  (CH = {ch_scores[k_range.index(best_k_ch)]:.2f})")
print(f"  肘部法则最佳 k   = {best_k_elbow}")
print(f"  >>> 综合判定最优聚类数: k = {optimal_k}")

# 生成最终聚类标签
final_labels = fcluster(Z, optimal_k, criterion="maxclust")
cluster_counts = pd.Series(final_labels).value_counts().sort_index()
print(f"  各簇样本数: {dict(cluster_counts)}")


# ============================================================
# 6. 可视化
# ============================================================
print("\n[6/7] 生成可视化图表 ...")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 图1: 谱系图（Dendrogram）----------
fig1, ax1 = plt.subplots(figsize=(20, 8))
dn = dendrogram(
    Z,
    labels=ids,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold=0.7 * max(Z[:, 2]),
    above_threshold_color="grey",
    ax=ax1,
)
# 在最优 k 处画水平截断线
cut_height = Z[-(optimal_k - 1), 2] if optimal_k > 1 else Z[-1, 2]
ax1.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.5,
            label=f"Optimal cut (k={optimal_k})")
ax1.set_title(f"Hierarchical Clustering Dendrogram (Ward + Euclidean)\nOptimal k = {optimal_k}",
              fontsize=14, fontweight="bold")
ax1.set_xlabel("Sample ID")
ax1.set_ylabel("Distance")
ax1.legend()
fig1.tight_layout()
path_dendro = OUTPUT_DIR / "dendrogram.png"
fig1.savefig(path_dendro, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  ✓ 谱系图已保存: {path_dendro}")

# ---------- 图2: 聚类评估指标 ----------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Silhouette
axes2[0].plot(k_range, silhouette_scores, "o-", color="#2c7bb6", markersize=6)
axes2[0].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7)
axes2[0].set_title("Silhouette Score", fontsize=12, fontweight="bold")
axes2[0].set_xlabel("Number of clusters (k)")
axes2[0].set_ylabel("Silhouette Score")
axes2[0].grid(True, alpha=0.3)

# CH Index
axes2[1].plot(k_range, ch_scores, "s-", color="#d7191c", markersize=6)
axes2[1].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7)
axes2[1].set_title("Calinski-Harabasz Index", fontsize=12, fontweight="bold")
axes2[1].set_xlabel("Number of clusters (k)")
axes2[1].set_ylabel("CH Score")
axes2[1].grid(True, alpha=0.3)

# Elbow
axes2[2].plot(k_range, inertias, "D-", color="#fdae61", markersize=6)
axes2[2].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7)
axes2[2].set_title("Elbow Method (WCSS)", fontsize=12, fontweight="bold")
axes2[2].set_xlabel("Number of clusters (k)")
axes2[2].set_ylabel("Within-Cluster Sum of Squares")
axes2[2].grid(True, alpha=0.3)

fig2.suptitle(f"Clustering Evaluation Metrics  →  Optimal k = {optimal_k}",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
path_eval = OUTPUT_DIR / "clustering_evaluation.png"
fig2.savefig(path_eval, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  ✓ 评估指标图已保存: {path_eval}")

# ---------- 图3: PCA 降维聚类散点图 ----------
# 降到 2 维或 3 维做可视化
pca_viz = PCA(n_components=2)
X_pca = pca_viz.fit_transform(X_scaled)

fig3, ax3 = plt.subplots(figsize=(12, 9))
cmap = plt.cm.get_cmap("tab10", optimal_k)
scatter = ax3.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=final_labels, cmap=cmap, s=60, alpha=0.8,
    edgecolors="black", linewidth=0.5,
)
# 标注每个点
for i, sid in enumerate(ids):
    ax3.annotate(
        sid, (X_pca[i, 0], X_pca[i, 1]),
        fontsize=6, alpha=0.7,
        textcoords="offset points", xytext=(3, 3),
    )
ax3.set_xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)")
ax3.set_title(
    f"PCA Scatter Plot (k={optimal_k}) — Ward + Euclidean",
    fontsize=14, fontweight="bold",
)
cbar = plt.colorbar(scatter, ax=ax3, ticks=range(1, optimal_k + 1))
cbar.set_label("Cluster", fontsize=11)
ax3.grid(True, alpha=0.2)
fig3.tight_layout()
path_pca = OUTPUT_DIR / "pca_clusters.png"
fig3.savefig(path_pca, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  ✓ PCA 聚类散点图已保存: {path_pca}")

# ---------- 图4: 各簇指标均值热力图 ----------
fig4, ax4 = plt.subplots(figsize=(16, max(6, optimal_k * 0.8)))
cluster_means = pd.DataFrame(index=range(1, optimal_k + 1), columns=active_params)
for c in range(1, optimal_k + 1):
    cluster_means.loc[c] = df_params.iloc[np.where(final_labels == c)[0]].mean()

# 对均值做归一化方便热力图展示
from sklearn.preprocessing import MinMaxScaler as MMS
cluster_means_norm = pd.DataFrame(
    MMS().fit_transform(cluster_means.T).T,
    index=cluster_means.index,
    columns=cluster_means.columns,
)
im = ax4.imshow(cluster_means_norm.values, aspect="auto", cmap="RdYlBu_r")
ax4.set_xticks(range(len(active_params)))
ax4.set_xticklabels(active_params, rotation=45, ha="right", fontsize=8)
ax4.set_yticks(range(optimal_k))
ax4.set_yticklabels([f"Cluster {i}" for i in range(1, optimal_k + 1)], fontsize=10)
ax4.set_title("Cluster Characteristic Heatmap (Normalized Means)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax4, label="Normalized Value")
fig4.tight_layout()
path_heat = OUTPUT_DIR / "cluster_heatmap.png"
fig4.savefig(path_heat, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  ✓ 簇特征热力图已保存: {path_heat}")


# ============================================================
# 7. 导出结果
# ============================================================
print("\n[7/7] 导出结果文件 ...")

# --- 7a. 聚类结果 DataFrame ---
df_result = df_raw.copy()
df_result["Cluster"] = final_labels
df_result = df_result.sort_values(["Cluster", id_col])

# 各簇统计信息
cluster_stats = {}
for c in sorted(set(final_labels)):
    cluster_ids = [ids[i] for i in range(n_samples) if final_labels[i] == c]
    param_values = df_result[df_result["Cluster"] == c][active_params]
    cluster_stats[f"Cluster_{c}"] = {
        "size": int(len(cluster_ids)),
        "sample_ids": cluster_ids,
        "parameter_means": param_values.mean().round(6).to_dict(),
        "parameter_stds": param_values.std().round(6).to_dict(),
    }

# --- 7b. JSON 输出 ---
json_output = {
    "analysis_metadata": {
        "title": "出厂水水质监测数据层次聚类分析",
        "method": "Hierarchical Clustering (Agglomerative)",
        "linkage": "Ward",
        "distance_metric": "Euclidean",
        "data_scaling": "Z-score Standardization",
        "input_file": INPUT_EXCEL,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": n_samples,
        "total_parameters": len(param_names),
        "active_parameters": n_params,
        "removed_constant_columns": zero_var_cols,
        "optimal_clusters": optimal_k,
    },
    "evaluation_metrics": {
        "silhouette_scores": {str(k): round(s, 4) for k, s in zip(k_range, silhouette_scores)},
        "calinski_harabasz_scores": {str(k): round(s, 2) for k, s in zip(k_range, ch_scores)},
        "within_cluster_ss": {str(k): round(s, 4) for k, s in zip(k_range, inertias)},
        "best_k_by_silhouette": best_k_sil,
        "best_k_by_ch": best_k_ch,
        "best_k_by_elbow": best_k_elbow,
    },
    "cluster_statistics": cluster_stats,
    "sample_assignments": {
        str(row[id_col]): int(row["Cluster"])
        for _, row in df_result.iterrows()
    },
    "pca_explained_variance": {
        "PC1": round(pca_viz.explained_variance_ratio_[0], 4),
        "PC2": round(pca_viz.explained_variance_ratio_[1], 4),
    },
}

path_json = OUTPUT_DIR / "clustering_results.json"
with open(path_json, "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)
print(f"  ✓ JSON 结果已保存: {path_json}")

# --- 7c. Markdown 报告 ---
md_report = f"""# 出厂水水质监测数据 — 层次聚类分析报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 分析方法

| 项目 | 说明 |
|------|------|
| **聚类方法** | 层次聚类（Agglomerative Hierarchical Clustering） |
| **连接法** | Ward 法（最小化簇内方差增量） |
| **距离度量** | 欧氏距离（Euclidean Distance） |
| **数据标准化** | Z-score 标准化（均值为 0，标准差为 1） |
| **最优 k 判定** | 轮廓系数 + Calinski-Harabasz 指数 + 肘部法则综合投票 |

---

## 2. 数据概况

| 项目 | 数值 |
|------|------|
| 输入文件 | `{INPUT_EXCEL}` |
| 样本总数 | {n_samples} |
| 原始指标数 | {len(param_names)} |
| 有效指标数 | {n_params} |
| 剔除常数列 | {len(zero_var_cols)} 个{'（' + ', '.join(str(c) for c in zero_var_cols) + '）' if zero_var_cols else ''} |

**使用的指标：** {', '.join(str(c) for c in active_params)}

---

## 3. 最优聚类数判定

### 评估指标汇总

| k | Silhouette | CH Index | WCSS |
|---|-----------|----------|------|
"""
for i, k in enumerate(k_range):
    md_report += f"| {k} | {silhouette_scores[i]:.4f} | {ch_scores[i]:.2f} | {inertias[i]:.4f} |\n"

md_report += f"""
### 各方法最佳 k

| 方法 | 最佳 k | 指标值 |
|------|--------|--------|
| 轮廓系数（Silhouette） | {best_k_sil} | {silhouette_scores[k_range.index(best_k_sil)]:.4f} |
| Calinski-Harabasz 指数 | {best_k_ch} | {ch_scores[k_range.index(best_k_ch)]:.2f} |
| 肘部法则（Elbow） | {best_k_elbow} | — |

> **最终选定聚类数：k = {optimal_k}**（综合投票结果）

---

## 4. 聚类结果

### 各簇基本信息

| 簇编号 | 样本数 | 占比 |
|--------|--------|------|
"""
for c in sorted(set(final_labels)):
    cnt = int((final_labels == c).sum())
    md_report += f"| Cluster {c} | {cnt} | {cnt/n_samples*100:.1f}% |\n"

md_report += """
### 各簇指标均值（原始尺度）

"""
# 指标均值表 - 转置为 指标 × 簇
for c in sorted(set(final_labels)):
    mean_vals = cluster_stats[f"Cluster_{c}"]["parameter_means"]
    md_report += f"\n#### Cluster {c} ({cluster_stats[f'Cluster_{c}']['size']} 个样本)\n\n"
    md_report += "| 指标 | 均值 |\n|------|------|\n"
    for param, val in mean_vals.items():
        md_report += f"| {param} | {val} |\n"

md_report += f"""
---

## 5. PCA 降维信息

| 主成分 | 解释方差比 | 累计 |
|--------|-----------|------|
| PC1 | {pca_viz.explained_variance_ratio_[0]*100:.1f}% | {pca_viz.explained_variance_ratio_[0]*100:.1f}% |
| PC2 | {pca_viz.explained_variance_ratio_[1]*100:.1f}% | {sum(pca_viz.explained_variance_ratio_[:2])*100:.1f}% |

---

## 6. 输出文件清单

| 文件 | 说明 |
|------|------|
| `clustering_results.json` | 完整聚类结果（含统计指标与样本分配） |
| `dendrogram.png` | 层次聚类谱系图（含最优截断线） |
| `clustering_evaluation.png` | 三种评估指标曲线图 |
| `pca_clusters.png` | PCA 降维聚类散点图 |
| `cluster_heatmap.png` | 各簇指标均值热力图 |
| `analysis_report.md` | 本分析报告 |

---

*报告由 `DataClassification.py` 自动生成*
"""

path_md = OUTPUT_DIR / "analysis_report.md"
with open(path_md, "w", encoding="utf-8") as f:
    f.write(md_report)
print(f"  ✓ Markdown 报告已保存: {path_md}")


# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("聚类分析完成！")
print(f"最优聚类数: k = {optimal_k}")
print(f"结果目录: {OUTPUT_DIR}")
print("=" * 60)
