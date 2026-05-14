"""
子指数计算脚本 — DLIR / EF / ISF 三种模型
基于 GB 5749-2022 标准限值及论文方法 (Ding et al., 2023)
"""

import pandas as pd
import numpy as np

# ========== 读取数据 ==========
data_path = r'D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx'
df = pd.read_excel(data_path)
print(f'读取水质数据: {df.shape[0]} 样本 × {df.shape[1]} 列')

# ========== 指标参数定义 ==========
# xideal=None 表示 xideal=x1 (越小越好)
indicator_params = {
    'Fluoride':                     {'xideal': None, 'x1': 0,   'x2': 1},
    'Nitrate':                      {'xideal': None, 'x1': 0,   'x2': 10},
    'Lead (Pb)':                    {'xideal': None, 'x1': 0,   'x2': 0.01},
    'Trihalomethanes (THMs)':       {'xideal': None, 'x1': 0,   'x2': 1},
    'Free Chlorine':                {'xideal': 0.7,  'x1': 0.3, 'x2': 4},
    'pH':                           {'xideal': 7,    'x1': 6.5, 'x2': 8.5},
    'Permanganate Index (COD_Mn)':  {'xideal': None, 'x1': 0,   'x2': 3},
    'Total Organic Carbon (TOC)':   {'xideal': None, 'x1': 0,   'x2': df['Total Organic Carbon (TOC)'].max()},  # max=3.98
    'Total Dissolved Solids (TDS)': {'xideal': 300,  'x1': 0,   'x2': 1000},
    'Total Hardness':               {'xideal': 170,  'x1': 0,   'x2': 450},
    'Chloride':                     {'xideal': 75,   'x1': 0,   'x2': 250},
    'Sulfate':                      {'xideal': 75,   'x1': 0,   'x2': 250},
    'Aluminum (Al)':                {'xideal': None, 'x1': 0,   'x2': 0.2},
    'Water Temperature':            {'xideal': 15,   'x1': 3,   'x2': 35},
}

# 对 xideal=None 的指标，令 xideal=x1
for name, p in indicator_params.items():
    if p['xideal'] is None:
        p['xideal'] = p['x1']

# ========== 模型尺度参数 ==========
models = {
    'DLIR': {'S1': 100,   'S2': 0,     'Sm': 40},
    'EF':   {'S1': 99,    'S2': 0,     'Sm': 5.310},
    'ISF':  {'S1': 1.571, 'S2': 0,     'Sm': 0.412},
}

# ========== 计算函数 ==========
def compute_si(xi, xideal, x1, x2, S1, S2, Sm):
    """
    DLIR 子指数计算:
    Eq(2): xi > xideal → SI = S1 - (S1 - S2) * (xi - xideal) / (x2 - xideal)
    Eq(3): xi ≤ xideal → SI = S1 - (S1 - Sm) * (xideal - xi) / (xideal - x1)
    当 xideal=x1 时，Eq(3) 退化 → 直接使用 Eq(2)
    """
    if xi > xideal:
        si = S1 - (S1 - S2) * (xi - xideal) / (x2 - xideal)
    else:
        denom = xideal - x1
        if denom == 0:
            si = S1  # xideal=x1 且 xi≤xideal → 完美情况
        else:
            si = S1 - (S1 - Sm) * (xideal - xi) / denom
    return si

# ========== 生成子指数表格 ==========
output_path = r'D:\WQIPaper\basicData\Sub_Index_Data.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for model_name, scales in models.items():
        S1, S2, Sm = scales['S1'], scales['S2'], scales['Sm']
        si_df = df[['ID', 'Date']].copy()

        for ind_name, p in indicator_params.items():
            xideal, x1, x2 = p['xideal'], p['x1'], p['x2']
            si_df[ind_name] = df[ind_name].apply(
                lambda xi: compute_si(xi, xideal, x1, x2, S1, S2, Sm)
            )

        si_df.to_excel(writer, sheet_name=model_name, index=False)
        print(f'  [{model_name}] S1={S1}, S2={S2}, Sm={Sm} → 完成')

print(f'\n✅ 子指数表格已生成: {output_path}')
print(f'   3 个 Sheet: DLIR / EF / ISF')
print(f'   尺寸: {si_df.shape[0]} 样本 × {len(indicator_params)} 指标')

# ========== 快速统计摘要 ==========
print('\n========== 各模型统计摘要 ==========')
for model_name, scales in models.items():
    S1, S2, Sm = scales['S1'], scales['S2'], scales['Sm']
    si_df = pd.read_excel(output_path, sheet_name=model_name)
    indicator_cols = list(indicator_params.keys())
    stats = si_df[indicator_cols].describe()
    print(f'\n--- {model_name} (S1={S1}, S2={S2}, Sm={Sm}) ---')
    print(f'  各指标均值范围: {stats.loc["mean"].min():.4f} ~ {stats.loc["mean"].max():.4f}')
    print(f'  各指标标准差范围: {stats.loc["std"].min():.4f} ~ {stats.loc["std"].max():.4f}')

print('\n🎉 全部完成!')
