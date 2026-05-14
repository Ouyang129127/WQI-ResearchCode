# -*- coding: utf-8 -*-
"""
============================================================
出厂水水质监测数据 — 合成数据生成器
============================================================

基于原始数据拟合多元正态分布，生成 2 倍合成样本（342 组总计），
保留原始数据的：
  - 各指标均值、方差
  - 指标间的线性相关性（协方差结构）
  - 常数/近零方差列的定值特性

方法：对非常数列拟合多元正态分布，MCMC 采样，再裁剪到合理范围。
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 路径配置
# ============================================================
INPUT_EXCEL = r"D:\WQIPaper\basicData\SuzhouYuejian-预处理完毕.xlsx"
OUTPUT_DIR = Path(r"D:\WQIPaper\basicData")
OUTPUT_EXCEL = OUTPUT_DIR / "SuzhouAdding.xlsx"

print("=" * 60)
print("出厂水水质监测数据 — 合成数据生成")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# 1. 加载原始数据
# ============================================================
print("\n[1/5] 加载原始数据 ...")
df_orig = pd.read_excel(INPUT_EXCEL)
n_orig = len(df_orig)
id_col = df_orig.columns[0]
print(f"  原始样本数: {n_orig}")
print(f"  总列数: {df_orig.shape[1]}")
print(f"  ID 列: '{id_col}'")

# 分离 ID 和参数
orig_ids = df_orig[id_col].tolist()
df_params = df_orig.drop(columns=[id_col])
# 仅保留数值列
df_params = df_params.select_dtypes(include=[np.number])
all_params = df_params.columns.tolist()
n_params_all = len(all_params)
print(f"  数值参数列: {n_params_all} 个")


# ============================================================
# 2. 分离常数列与变量列
# ============================================================
print("\n[2/5] 识别常数列与变量列 ...")

EPS = 1e-12
constant_cols = []
variable_cols = []
constant_values = {}

for col in all_params:
    std_val = df_params[col].std()
    if std_val < EPS:
        constant_cols.append(col)
        constant_values[col] = df_params[col].iloc[0]
    else:
        variable_cols.append(col)

print(f"  常数列 ({len(constant_cols)}): {constant_cols}")
for c in constant_cols:
    print(f"      {c} = {constant_values[c]}")
print(f"  变量列 ({len(variable_cols)}): {len(variable_cols)} 个")

X_orig = df_params[variable_cols].values
n_vars = len(variable_cols)


# ============================================================
# 3. 拟合多元正态分布并生成合成数据
# ============================================================
print("\n[3/5] 拟合多元正态分布 ...")

mean_vec = X_orig.mean(axis=0)
cov_mat = np.cov(X_orig, rowvar=False)

# 协方差矩阵正则化：添加小量对角扰动保证正定
eigvals = np.linalg.eigvalsh(cov_mat)
min_eig = eigvals.min()
print(f"  协方差矩阵最小特征值: {min_eig:.6f}")

if min_eig < EPS:
    reg = max(1e-10, abs(min_eig) * 1.1)
    cov_mat += np.eye(n_vars) * reg
    print(f"  ⚠ 协方差矩阵近似奇异，已添加正则化 (λ={reg:.2e})")
else:
    print(f"  ✓ 协方差矩阵正定，条件数: {eigvals.max() / eigvals.min():.1f}")

# 生成合成样本
n_synthetic = n_orig  # 再生成 1 倍，总计 2 倍
rng = np.random.RandomState(42)
X_synthetic = rng.multivariate_normal(mean_vec, cov_mat, size=n_synthetic)
print(f"  已生成 {n_synthetic} 组合成样本")

# ============================================================
# 4. 后处理：裁剪到合理范围 + 常数填充
# ============================================================
print("\n[4/5] 后处理：裁剪到合理范围 ...")

# 获取原始数据的 min/max，略微放宽边界（±20% 范围扩展）
orig_min = X_orig.min(axis=0)
orig_max = X_orig.max(axis=0)
# 对于极小值列不做相对扩展，用绝对扩展
range_width = orig_max - orig_min
lower_bound = orig_min - 0.2 * np.maximum(range_width, 1e-10)
upper_bound = orig_max + 0.2 * np.maximum(range_width, 1e-10)
# 保证下界不大于上界
lower_bound = np.minimum(lower_bound, orig_min)
upper_bound = np.maximum(upper_bound, orig_max)

# 裁剪
X_synthetic_clipped = np.clip(X_synthetic, lower_bound, upper_bound)

# 统计裁剪比例
clipped_frac = np.mean(
    (X_synthetic < lower_bound) | (X_synthetic > upper_bound)
)
print(f"  被裁剪的数值比例: {clipped_frac*100:.2f}%")

# 构建合成 DataFrame（变量列）
df_syn_var = pd.DataFrame(X_synthetic_clipped, columns=variable_cols)

# 添加常数列
for col in constant_cols:
    df_syn_var[col] = constant_values[col]

# 按原始列顺序排列
df_synthetic = df_syn_var[all_params].copy()

# 设置新 ID（接着原始 ID 继续编号）
max_orig_id = max(orig_ids) if isinstance(orig_ids[0], (int, float)) else n_orig
new_ids = list(range(int(max_orig_id) + 1, int(max_orig_id) + n_synthetic + 1))
df_synthetic.insert(0, id_col, new_ids)

print(f"  合成数据 shape: {df_synthetic.shape}")


# ============================================================
# 5. 合并并导出
# ============================================================
print("\n[5/5] 合并原始数据与合成数据 ...")

df_combined = pd.concat([df_orig, df_synthetic], ignore_index=True)
print(f"  合并后总样本数: {len(df_combined)}")
print(f"  原始样本: {n_orig}")
print(f"  合成样本: {n_synthetic}")

# 保存 Excel
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_combined.to_excel(OUTPUT_EXCEL, index=False)
print(f"  ✓ 已保存: {OUTPUT_EXCEL}")


# ============================================================
# 统计对比
# ============================================================
print("\n" + "=" * 60)
print("原始数据 vs 合成数据 统计对比")
print("=" * 60)

for col in variable_cols[:10]:  # 仅展示前 10 列
    o_mean, o_std = df_orig[col].mean(), df_orig[col].std()
    s_mean, s_std = df_synthetic[col].mean(), df_synthetic[col].std()
    print(f"  {col:12s}: μ={o_mean:.4f}→{s_mean:.4f}  σ={o_std:.4f}→{s_std:.4f}")

print(f"\n... (共 {n_vars} 个变量列，此处仅展示前 10 列)")

print("\n" + "=" * 60)
print("数据生成完成！")
print(f"输出文件: {OUTPUT_EXCEL}")
print(f"总样本数: {len(df_combined)}")
print("=" * 60)
