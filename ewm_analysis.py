import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
INPUT_FILE = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
OUT_DIR = r"D:\WQIPaper\DataAnalytics"
OUT_EXCEL = os.path.join(OUT_DIR, "EWM_Weights.xlsx")
OUT_MD = os.path.join(OUT_DIR, "EWM_Analysis_Report.md")

# Indicator types: cost (lower better), optimal (deviation from center)
IND_TYPES = {
    'Lead (Pb)': 'cost',
    'Total Dissolved Solids (TDS)': 'cost',
    'Nitrate': 'cost',
    'Total Hardness': 'cost',
    'Sulfate': 'cost',
    'Permanganate Index (COD_Mn)': 'cost',
    'Fluoride': 'cost',
    'Water Temperature': 'cost',
    'pH': ('optimal', 7.0),
    'Trihalomethanes (THMs)': 'cost',
    'Free Chlorine': 'cost',
    'Aluminum (Al)': 'cost',
    'Total Organic Carbon (TOC)': 'cost',
}

print("=" * 60)
print("Step 1: Load data")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')

# Drop duplicate Free Chlorine cols
fc_cols = [c for c in df.columns if c.startswith('Free Chlorine')]
if len(fc_cols) > 1:
    df = df.drop(columns=fc_cols[1:])

# Extract indicators
meta_cols = ['ID', 'Date']
indicator_cols = [c for c in df.columns if c not in meta_cols]
X = df[indicator_cols].values.astype(np.float64)
n, m = X.shape
names = indicator_cols

print(f"Samples: {n}, Indicators: {m}")
print(f"Indicators: {names}")

# Step 2: Normalize
print("\nStep 2: Normalize (Min-Max)")
R = np.zeros_like(X)
for j in range(m):
    col = X[:, j]
    xmin, xmax = col.min(), col.max()
    it = IND_TYPES.get(names[j], 'cost')
    
    if isinstance(it, tuple) and it[0] == 'optimal':
        # pH: deviation from optimal
        opt = it[1]
        dev = np.abs(col - opt)
        dmin, dmax = dev.min(), dev.max()
        R[:, j] = (dmax - dev) / (dmax - dmin) if dmax > dmin else 1.0
        print(f"  {names[j]:40s} [optimal] min={xmin:.2f} max={xmax:.2f}")
    elif it == 'cost':
        R[:, j] = (xmax - col) / (xmax - xmin) if xmax > xmin else 1.0
        print(f"  {names[j]:40s} [cost]    min={xmin:.4f} max={xmax:.4f}")
    else:  # benefit
        R[:, j] = (col - xmin) / (xmax - xmin) if xmax > xmin else 1.0
        print(f"  {names[j]:40s} [benefit] min={xmin:.4f} max={xmax:.4f}")

# Shift to avoid log(0)
R = R + 1e-4

# Step 3: Proportion
print("\nStep 3: Proportion P")
P = R / R.sum(axis=0)

# Step 4: Entropy
print("Step 4: Information Entropy")
k = 1.0 / np.log(n)
entropy = np.zeros(m)
for j in range(m):
    pj = P[:, j]
    entropy[j] = -k * np.sum(pj * np.log(pj + 1e-12))

# Step 5: Utility & Weight
print("Step 5: Information Utility & Weight")
utility = 1.0 - entropy
weights = utility / utility.sum()
weights_pct = weights * 100

# Step 6: Build results table
results = pd.DataFrame({
    'Indicators': names,
    'Information entropy': entropy.round(4),
    'Information utility': utility.round(4),
    'Weight': [f"{w:.2f} %" for w in weights_pct],
    'Weight_raw': weights_pct
}).sort_values('Weight_raw', ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(results[['Indicators', 'Information entropy', 'Information utility', 'Weight']].to_string(index=False))

# Save Excel
results_excel = results.copy()
results_excel.to_excel(OUT_EXCEL, index=False)
print(f"\nExcel: {OUT_EXCEL}")

# Generate MD report
md = []
md.append("# EWM (Entropy Weight Method) Analysis Report\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append(f"**Samples:** {n} | **Indicators:** {m}  \n\n")

md.append("## Methodology\n\n")
md.append("1. **Normalization** (Min-Max) — cost-type indicators: r = (max-x)/(max-min); pH: deviation from 7.0\n")
md.append("2. **Proportion**: p_ij = r_ij / sum(r_ij)\n")
md.append("3. **Entropy**: e_j = -(1/ln n) * sum(p_ij * ln p_ij)\n")
md.append("4. **Utility**: d_j = 1 - e_j\n")
md.append("5. **Weight**: w_j = d_j / sum(d_j)\n\n")

md.append("## Indicator Types\n\n")
md.append("| Indicator | Type |\n|-----------|------|\n")
for name in names:
    it = IND_TYPES.get(name, 'cost')
    if isinstance(it, tuple):
        md.append(f"| {name} | optimal (center={it[1]}) |\n")
    else:
        md.append(f"| {name} | {it} |\n")

md.append("\n## Results\n\n")
md.append("| Indicators | Information entropy | Information utility | Weight |\n")
md.append("|-----------|-------------------|-------------------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Information entropy']} | {row['Information utility']} | {row['Weight']} |\n")

md.append("\n## Weight Distribution\n\n```\n")
max_name = max(len(n) for n in names)
for _, row in results.iterrows():
    bar = '#' * int(row['Weight_raw'] * 2)
    md.append(f"{row['Indicators']:<{max_name+2}} {bar} {row['Weight']}\n")
md.append("```\n\n")

md.append("## Key Insights\n\n")
top3 = results.head(3)
md.append("**Top 3 indicators (highest information utility):**\n")
for _, row in top3.iterrows():
    md.append(f"- {row['Indicators']} ({row['Weight']})\n")

md.append("\n> Note: EWM weights reflect statistical information content (variation), not health significance. ")
md.append("For WQI construction, consider combining with expert judgment or AHP-based weights.\n")

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"MD report: {OUT_MD}")
print("\n[DONE] EWM Analysis Complete!")
