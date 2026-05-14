import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
INPUT_FILE = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
OUT_DIR = r"D:\WQIPaper\DataAnalytics"
OUT_EXCEL = os.path.join(OUT_DIR, "CRITIC_Weights.xlsx")
OUT_MD = os.path.join(OUT_DIR, "CRITIC_Analysis_Report.md")

# Indicator types
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
    'Chloride': 'cost',
    'Aluminum (Al)': 'cost',
    'Free Chlorine': 'cost',
    'Total Organic Carbon (TOC)': 'cost',
}

print("=" * 60)
print("Step 1: Load data")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')

fc_cols = [c for c in df.columns if c.startswith('Free Chlorine')]
if len(fc_cols) > 1:
    df = df.drop(columns=fc_cols[1:])

meta_cols = ['ID', 'Date']
indicator_cols = [c for c in df.columns if c not in meta_cols]
X = df[indicator_cols].values.astype(np.float64)
n, m = X.shape
names = indicator_cols

print(f"Samples: {n}, Indicators: {m}")

# Step 2: Normalize (Min-Max)
print("\nStep 2: Normalize (Min-Max)")
R = np.zeros_like(X)
for j in range(m):
    col = X[:, j]
    xmin, xmax = col.min(), col.max()
    it = IND_TYPES.get(names[j], 'cost')
    
    if isinstance(it, tuple) and it[0] == 'optimal':
        dev = np.abs(col - it[1])
        dmin, dmax = dev.min(), dev.max()
        R[:, j] = (dmax - dev) / (dmax - dmin) if dmax > dmin else 1.0
    elif it == 'cost':
        R[:, j] = (xmax - col) / (xmax - xmin) if xmax > xmin else 1.0
    else:
        R[:, j] = (col - xmin) / (xmax - xmin) if xmax > xmin else 1.0

print("Normalized range: [{:.4f}, {:.4f}]".format(R.min(), R.max()))

# Step 3: Standard deviation of each indicator (contrast intensity)
print("\nStep 3: Standard deviation (contrast intensity)")
sigma = np.std(R, axis=0, ddof=1)  # sample std

# Step 4: Correlation matrix and conflict degree
print("Step 4: Correlation matrix & conflict degree")
corr_matrix = np.corrcoef(R.T)  # Pearson correlation

# Conflict degree for indicator j: sum over i of (1 - |r_ij|)
# Higher conflict = less correlated with others = more unique info
conflict = np.zeros(m)
for j in range(m):
    conflict[j] = np.sum(1.0 - np.abs(corr_matrix[j, :]))

# Step 5: Information content C_j = sigma_j * conflict_j
print("Step 5: Information content (C = std * conflict)")
info_content = sigma * conflict

# Step 6: Weights w_j = C_j / sum(C_j)
print("Step 6: Weight calculation")
weights = info_content / info_content.sum()
weights_pct = weights * 100

# Results table
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

results = pd.DataFrame({
    'Indicators': names,
    'Std_Deviation': sigma.round(4),
    'Conflict_Degree': conflict.round(4),
    'Information_Content': info_content.round(4),
    'Weight': [f"{w:.2f} %" for w in weights_pct],
    'Weight_raw': weights_pct
}).sort_values('Weight_raw', ascending=False).reset_index(drop=True)

print(f"\n{'Indicators':40s} {'StdDev':>8s} {'Conflict':>9s} {'Info':>9s} {'Weight':>9s}")
print("-" * 85)
for _, row in results.iterrows():
    print(f"{row['Indicators']:40s} {row['Std_Deviation']:>8.4f} {row['Conflict_Degree']:>9.4f} {row['Information_Content']:>9.4f} {row['Weight']:>9s}")

# Save Excel
results_excel = results.copy()
results_excel.to_excel(OUT_EXCEL, index=False)
print(f"\nExcel: {OUT_EXCEL}")

# Generate MD report
md = []
md.append("# CRITIC Method Analysis Report\n\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append(f"**Input file:** `{INPUT_FILE}`  \n")
md.append(f"**Samples:** {n} | **Indicators:** {m}  \n\n")

md.append("---\n## Methodology\n\n")
md.append("CRITIC (**CR**iteria **I**mportance **T**hrough **I**ntercriteria **C**orrelation) ")
md.append("determines objective weights by simultaneously considering:\n\n")
md.append("### 1. Contrast Intensity (Standard Deviation)\n")
md.append("$$\\sigma_j = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(r_{ij} - \\bar{r}_j)^2}$$\n")
md.append("Larger standard deviation = higher contrast = more valuable information.\n\n")

md.append("### 2. Conflict Degree\n")
md.append("$$f_j = \\sum_{i=1}^{m} (1 - |r_{ij}|)$$\n")
md.append("where $r_{ij}$ is the Pearson correlation between indicator $i$ and $j$.  \n")
md.append("Higher conflict = less correlated with other indicators = more unique information.\n\n")

md.append("### 3. Information Content\n")
md.append("$$C_j = \\sigma_j \\times f_j$$\n")
md.append("Combines both contrast intensity and conflict degree.\n\n")

md.append("### 4. Weight Calculation\n")
md.append("$$w_j = \\frac{C_j}{\\sum_{k=1}^{m} C_k}$$\n\n")

md.append("---\n## Results\n\n")

md.append("| Indicators | Std Deviation | Conflict Degree | Information Content | Weight |\n")
md.append("|-----------|-------------|----------------|-------------------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Std_Deviation']} | {row['Conflict_Degree']} | {row['Information_Content']} | {row['Weight']} |\n")

md.append("\n## Standard Deviation Ranking\n\n")
temp = results.sort_values('Std_Deviation', ascending=False)
md.append("| Indicator | Std Deviation |\n|-----------|-------------|\n")
for _, row in temp.iterrows():
    md.append(f"| {row['Indicators']} | {row['Std_Deviation']} |\n")

md.append("\n## Conflict Degree Ranking\n\n")
temp = results.sort_values('Conflict_Degree', ascending=False)
md.append("| Indicator | Conflict Degree |\n|-----------|---------------|\n")
for _, row in temp.iterrows():
    md.append(f"| {row['Indicators']} | {row['Conflict_Degree']} |\n")

md.append("\n## Correlation Matrix (Visual)\n\n```\n")
# Compact correlation matrix
max_name = max(len(n) for n in names)
col_header = " " * (max_name + 2)
for n_short in [name[:5] for name in names]:
    col_header += f"{n_short:>7s}"
md.append(col_header + "\n")
for i, name in enumerate(names):
    line = f"{name:<{max_name+2}}"
    for j in range(m):
        r = corr_matrix[i, j]
        line += f"{r:>7.2f}"
    md.append(line + "\n")
md.append("```\n\n")

md.append("## Weight Distribution\n\n```\n")
for _, row in results.iterrows():
    bar = '#' * int(row['Weight_raw'] * 3)
    md.append(f"{row['Indicators']:<{max_name+2}} {bar} {row['Weight']}\n")
md.append("```\n\n")

md.append("## Key Insights\n\n")
top3 = results.head(3)
md.append("**Top 3 by CRITIC weight:**\n")
for _, row in top3.iterrows():
    md.append(f"- **{row['Indicators']}** ({row['Weight']}) — ")
    if row['Std_Deviation'] > np.median(sigma):
        md.append("high contrast + ")
    if row['Conflict_Degree'] > np.median(conflict):
        md.append("high conflict")
    md.append("\n")

md.append("\n**Interpretation:**\n")
md.append("- **Contrast intensity** (σ): Measures how much an indicator varies across samples. Higher = more discriminative.\n")
md.append("- **Conflict degree** (f): Measures independence from other indicators. Higher = more unique information.\n")
md.append("- Indicators with both high contrast AND high conflict receive the highest CRITIC weights.\n")

# Compare with EWM and RF if available
for method, file_name in [("EWM", "EWM_Weights.xlsx"), ("RF", "RF_Weights.xlsx")]:
    fpath = os.path.join(OUT_DIR, file_name)
    if os.path.exists(fpath):
        md.append(f"\n**Comparison with {method}:** see `{file_name}`\n")

md.append(f"\n> **Note:** CRITIC captures both the variability AND the independence of each indicator. ")
md.append("Unlike EWM (information entropy only), CRITIC rewards indicators that are uncorrelated with others. ")
md.append("Unlike RF, CRITIC does not require a target variable and is fully data-driven.\n")

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"MD report: {OUT_MD}")
print("\n[DONE] CRITIC Analysis Complete!")
