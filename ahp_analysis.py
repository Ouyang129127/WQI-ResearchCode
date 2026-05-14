# -*- coding: utf-8 -*-
"""AHP Subjective Weight Calculation for DWQI (14 indicators, 3 criteria groups)"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

OUT_DIR = r"D:\WQIPaper\DataAnalytics"

# ============================================================
# AHP Utility Functions
# ============================================================

def parse_matrix(text):
    """Parse a text matrix with fractions like '1/3' into numpy array."""
    rows = [r.strip() for r in text.strip().split('\n') if r.strip()]
    data = []
    for row in rows:
        vals = []
        for x in row.split():
            if '/' in x:
                num, den = x.split('/')
                vals.append(float(num) / float(den))
            else:
                vals.append(float(x))
        data.append(vals)
    return np.array(data)

def ahp_weights(matrix):
    """Calculate AHP weights using geometric mean method (Row GM).
    Returns: weights, lambda_max, CI, CR
    """
    n = matrix.shape[0]
    # Geometric mean of each row
    gm = np.prod(matrix, axis=1) ** (1.0 / n)
    w = gm / gm.sum()
    # lambda_max
    Aw = matrix.dot(w)
    lam_max = np.mean(Aw / w)
    # Consistency
    CI = (lam_max - n) / (n - 1)
    # RI values (Saaty)
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
                 13: 1.56, 14: 1.57, 15: 1.59}
    RI = RI_table.get(n, 1.5)
    CR = CI / RI if RI > 0 else 0.0
    return w, lam_max, CI, CR

def print_ahp_results(name, indicators, matrix, w, lam, CI, CR):
    """Pretty-print AHP results."""
    print(f"\n{'='*60}")
    print(f"  {name} (n={len(indicators)})")
    print(f"{'='*60}")
    for ind, wt in zip(indicators, w):
        print(f"  {ind:50s} {wt*100:6.2f}%")
    print(f"  {'-'*56}")
    print(f"  lambda_max = {lam:.4f}")
    print(f"  CI = {CI:.4f},  CR = {CR:.4f}  {'[PASS]' if CR < 0.10 else '[FAIL] CR >= 0.10'}")
    return w

# ============================================================
# Level 1: Criteria Layer
# ============================================================
# REVERSED: original matrices had row/col comparison direction swapped
# Transposed = true intention (Toxicological > Biological > Sensory)
l1_text = """
1 1/3 1/5
3 1 1/3
5 2 1
"""
l1_matrix = parse_matrix(l1_text)
l1_indicators = [
    "A: Sensory & General Chemistry",
    "B: Biological Stability",
    "C: Toxicological"
]
w1, lam1, ci1, cr1 = ahp_weights(l1_matrix)

# ============================================================
# Level 2a: Toxicological (4 indicators)
# ============================================================
# REVERSED: transposed
l2_tox_text = """
1 1/3 1/3 1/5
3 1 1/3 1/3
3 3 1 1/3
5 3 3 1
"""
l2_tox_matrix = parse_matrix(l2_tox_text)
l2_tox_indicators = [
    "Fluoride",
    "Nitrate",
    "Lead (Pb)",
    "Trihalomethanes (THMs)"
]
w2_tox, lam2t, ci2t, cr2t = ahp_weights(l2_tox_matrix)

# ============================================================
# Level 2b: Biological Stability (4 indicators)
# ============================================================
# REVERSED: transposed
l2_bio_text = """
1 1/3 1/3 1/5
3 1 1/3 1/3
3 3 1 1/3
5 3 3 1
"""
l2_bio_matrix = parse_matrix(l2_bio_text)
l2_bio_indicators = [
    "pH",
    "Permanganate Index (COD_Mn)",
    "Total Organic Carbon (TOC)",
    "Free Chlorine"
]
w2_bio, lam2b, ci2b, cr2b = ahp_weights(l2_bio_matrix)

# ============================================================
# Level 2c: Sensory & General Chemistry (6 indicators)
# ============================================================
# REVERSED: transposed
l2_sen_text = """
1 1 1/3 1/3 1/5 1/5
1 1 1/3 1/5 1/5 1/5
3 3 1 1/3 1/5 1/3
3 5 3 1 1/3 1/3
5 5 5 3 1 1
5 5 3 3 1 1
"""
l2_sen_matrix = parse_matrix(l2_sen_text)
l2_sen_indicators = [
    "Water Temperature",
    "Chloride",
    "Sulfate",
    "Total Hardness",
    "Aluminum (Al)",
    "Total Dissolved Solids (TDS)"
]
w2_sen, lam2s, ci2s, cr2s = ahp_weights(l2_sen_matrix)

# ============================================================
# Global AHP Weights
# ============================================================
print("="*60)
print("  AHP SUBJECTIVE WEIGHT CALCULATION")

print_ahp_results("LEVEL 1: Criteria", l1_indicators, l1_matrix, w1, lam1, ci1, cr1)
print_ahp_results("LEVEL 2a: Toxicological", l2_tox_indicators, l2_tox_matrix, w2_tox, lam2t, ci2t, cr2t)
print_ahp_results("LEVEL 2b: Biological Stability", l2_bio_indicators, l2_bio_matrix, w2_bio, lam2b, ci2b, cr2b)
print_ahp_results("LEVEL 2c: Sensory & General Chemistry", l2_sen_indicators, l2_sen_matrix, w2_sen, lam2s, ci2s, cr2s)

# Global weights
global_weights = np.concatenate([
    w2_tox * w1[2],    # Toxicological * C weight
    w2_bio * w1[1],    # Biological * B weight
    w2_sen * w1[0],    # Sensory * A weight
])

global_names = (
    l2_tox_indicators +
    l2_bio_indicators +
    l2_sen_indicators
)

# Sort by weight descending
order = np.argsort(global_weights)[::-1]

print(f"\n{'='*60}")
print(f"  GLOBAL AHP WEIGHTS (14 indicators, sorted)")
print(f"{'='*60}")
for i in order:
    print(f"  {global_names[i]:50s} {global_weights[i]*100:6.2f}%")
print(f"  {'-'*56}")
print(f"  Sum = {global_weights.sum():.6f}")

# ============================================================
# Export to Excel
# ============================================================
data = {
    'Indicator': global_names,
    'AHPL1_Group': ['Toxicological']*4 + ['Biological Stability']*4 + ['Sensory & General Chemistry']*6,
    'AHPL1_Weight': [w1[2]]*4 + [w1[1]]*4 + [w1[0]]*6,
    'AHPL2_WithinGroup_Weight': list(w2_tox) + list(w2_bio) + list(w2_sen),
    'AHP_Global_Weight': global_weights,
}
df_out = pd.DataFrame(data)
df_out = df_out.sort_values('AHP_Global_Weight', ascending=False).reset_index(drop=True)
df_out.index = df_out.index + 1
df_out.index.name = 'Rank'

# Also add percentage column
df_out['AHP_Weight_%'] = df_out['AHP_Global_Weight'] * 100

out_path = os.path.join(OUT_DIR, 'AHP_Weights.xlsx')
df_out.to_excel(out_path, index=True, engine='openpyxl')
print(f"\nExcel saved: {out_path}")

# ============================================================
# Generate MD Report
# ============================================================
md = []
md.append("# AHP Subjective Weight Analysis for DWQI\n\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append("**Method:** Analytic Hierarchy Process (AHP) with geometric mean method  \n")
md.append("**Structure:** 3 criteria groups, 14 indicators total  \n\n")

# Add correction note
md.append("---\n\n## ⚠️ Correction Note: Judgment Matrix Direction\n\n")
md.append("**Initial submission had all judgment matrices written in reversed direction.**\n\n")
md.append("In AHP, the judgment matrix element $a_{ij}$ represents the relative importance of **row $i$ compared to column $j$**. ")
md.append("The original submission placed higher importance on the **row** indicators, but intended the **column** indicators to be more important.\n\n")
md.append("**Example (Level 1):**\n\n")
md.append("| Version | Sensory vs Toxicological | Biological vs Toxicological | Result |\n")
md.append("|---------|-------------------------|---------------------------|--------|\n")
md.append("| ❌ Original (reversed) | $a_{13}=5$ (Sensory 5x Toxicological) | $a_{23}=2$ | Toxicological = 10.8% (weakest) |\n")
md.append("| ✅ Corrected (transposed) | $a_{31}=5$ (Toxicological 5x Sensory) | $a_{32}=3$ | Toxicological = 60.5% (strongest) |\n\n")
md.append("**Correction applied:** All 4 judgment matrices were **transposed** ($a_{ij} \\rightarrow a_{ji}$) to ")
md.append("reflect the true intention: **Toxicological > Biological Stability > Sensory & General Chemistry** in Level 1, ")
md.append("with corresponding transpositions in all Level 2 sub-matrices.\n\n")
md.append("**Affected matrices:** Level 1 (Criteria), Level 2a (Toxicological), Level 2b (Biological Stability), Level 2c (Sensory & General Chemistry)\n\n")

md.append("---\n\n## 1. Hierarchical Structure\n\n")
md.append("```\n")
md.append("Goal: DWQI\n")
md.append("  |-- [C1] Sensory & General Chemistry (6 indicators)\n")
md.append("  |     |-- Water Temperature, Chloride, Sulfate, Total Hardness, Al, TDS\n")
md.append("  |-- [C2] Biological Stability (4 indicators)\n")
md.append("  |     |-- pH, COD_Mn, TOC, Free Chlorine\n")
md.append("  |-- [C3] Toxicological (4 indicators)\n")
md.append("        |-- Fluoride, Nitrate, Lead (Pb), THMs\n")
md.append("```\n\n")

md.append("---\n\n## 2. Level 1: Criteria Weights\n\n")
md.append("| Criteria | Sensory & General | Biological Stability | Toxicological | Weight (%) |\n")
md.append("|----------|-------------------|---------------------|--------------|-----------|\n")
# Use corrected (transposed) matrix values
md.append(f"| Sensory & General Chemistry | 1 | 1/3 | 1/5 | **{w1[0]*100:.1f}%** |\n")
md.append(f"| Biological Stability | 3 | 1 | 1/3 | **{w1[1]*100:.1f}%** |\n")
md.append(f"| Toxicological | 5 | 3 | 1 | **{w1[2]*100:.1f}%** |\n\n")
md.append(f"- **lambda_max** = {lam1:.4f}\n")
md.append(f"- **CI** = {ci1:.4f}\n")
md.append(f"- **CR** = {cr1:.4f} {'✓ Pass' if cr1 < 0.10 else '✗ Fail'}\n\n")

md.append("---\n\n## 3. Level 2a: Toxicological Indicators\n\n")
md.append("| | Fluoride | Nitrate | Lead (Pb) | THMs | Weight (%) |\n")
md.append("|---|---------|---------|-----------|------|----------|\n")
# Corrected transposed matrix
md.append(f"| Fluoride | 1 | 1/3 | 1/3 | 1/5 | **{w2_tox[0]*100:.1f}%** |\n")
md.append(f"| Nitrate | 3 | 1 | 1/3 | 1/3 | **{w2_tox[1]*100:.1f}%** |\n")
md.append(f"| Lead (Pb) | 3 | 3 | 1 | 1/3 | **{w2_tox[2]*100:.1f}%** |\n")
md.append(f"| THMs | 5 | 3 | 3 | 1 | **{w2_tox[3]*100:.1f}%** |\n\n")
md.append(f"- **lambda_max** = {lam2t:.4f}\n")
md.append(f"- **CI** = {ci2t:.4f}\n")
md.append(f"- **CR** = {cr2t:.4f} {'✓ Pass' if cr2t < 0.10 else '✗ Fail'}\n\n")

md.append("---\n\n## 4. Level 2b: Biological Stability Indicators\n\n")
md.append("| | pH | COD_Mn | TOC | Free Chlorine | Weight (%) |\n")
md.append("|---|---|--------|-----|-------------|----------|\n")
# Corrected transposed matrix
md.append(f"| pH | 1 | 1/3 | 1/3 | 1/5 | **{w2_bio[0]*100:.1f}%** |\n")
md.append(f"| COD_Mn | 3 | 1 | 1/3 | 1/3 | **{w2_bio[1]*100:.1f}%** |\n")
md.append(f"| TOC | 3 | 3 | 1 | 1/3 | **{w2_bio[2]*100:.1f}%** |\n")
md.append(f"| Free Chlorine | 5 | 3 | 3 | 1 | **{w2_bio[3]*100:.1f}%** |\n\n")
md.append(f"- **lambda_max** = {lam2b:.4f}\n")
md.append(f"- **CI** = {ci2b:.4f}\n")
md.append(f"- **CR** = {cr2b:.4f} {'✓ Pass' if cr2b < 0.10 else '✗ Fail'}\n\n")

md.append("---\n\n## 5. Level 2c: Sensory & General Chemistry Indicators\n\n")
sen_labels = ["Water Temp", "Chloride", "Sulfate", "Total Hardness", "Aluminum (Al)", "TDS"]
md.append("| | " + " | ".join(sen_labels) + " | Weight (%) |\n")
md.append("|" + "|".join(["---"]*(len(sen_labels)+2)) + "|\n")
# Corrected transposed matrix
row_data = ["1", "1", "1/3", "1/3", "1/5", "1/5"]
md.append(f"| Water Temp | " + " | ".join(row_data) + f" | **{w2_sen[0]*100:.1f}%** |\n")
row_data = ["1", "1", "1/3", "1/5", "1/5", "1/5"]
md.append(f"| Chloride | " + " | ".join(row_data) + f" | **{w2_sen[1]*100:.1f}%** |\n")
row_data = ["3", "3", "1", "1/3", "1/5", "1/3"]
md.append(f"| Sulfate | " + " | ".join(row_data) + f" | **{w2_sen[2]*100:.1f}%** |\n")
row_data = ["3", "5", "3", "1", "1/3", "1/3"]
md.append(f"| Total Hardness | " + " | ".join(row_data) + f" | **{w2_sen[3]*100:.1f}%** |\n")
row_data = ["5", "5", "5", "3", "1", "1"]
md.append(f"| Aluminum (Al) | " + " | ".join(row_data) + f" | **{w2_sen[4]*100:.1f}%** |\n")
row_data = ["5", "5", "3", "3", "1", "1"]
md.append(f"| TDS | " + " | ".join(row_data) + f" | **{w2_sen[5]*100:.1f}%** |\n\n")
md.append(f"- **lambda_max** = {lam2s:.4f}\n")
md.append(f"- **CI** = {ci2s:.4f}\n")
md.append(f"- **CR** = {cr2s:.4f} {'✓ Pass' if cr2s < 0.10 else '✗ Fail'}\n\n")

# Global weights
md.append("---\n\n## 6. Global AHP Weights (14 Indicators)\n\n")
md.append("| Rank | Indicator | Group | L1 Weight (%) | L2 Weight (%) | Global Weight (%) |\n")
md.append("|------|-----------|-------|--------------|--------------|------------------|\n")
for rank, i in enumerate(order, 1):
    grp = df_out.loc[df_out['Indicator'] == global_names[i], 'AHPL1_Group'].values[0]
    l1w = df_out.loc[df_out['Indicator'] == global_names[i], 'AHPL1_Weight'].values[0] * 100
    l2w = df_out.loc[df_out['Indicator'] == global_names[i], 'AHPL2_WithinGroup_Weight'].values[0] * 100
    md.append(f"| {rank} | {global_names[i]} | {grp} | {l1w:.1f} | {l2w:.1f} | **{global_weights[i]*100:.2f}** |\n")

md.append(f"\n**Sum of global weights:** {global_weights.sum():.4f} (should equal 1.0)\n\n")

md.append("---\n\n## 7. Consistency Summary\n\n")
md.append("| Hierarchy Level | n | lambda_max | CI | RI | CR | Verdict |\n")
md.append("|----------------|----|-----------|-----|-----|-----|--------|\n")
RI_table = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41}
n1 = len(l1_indicators)
md.append(f"| Level 1 (Criteria) | {n1} | {lam1:.4f} | {ci1:.4f} | {RI_table.get(n1,1.5):.2f} | {cr1:.4f} | {'PASS' if cr1<0.1 else 'FAIL'} |\n")
md.append(f"| Level 2a (Tox) | 4 | {lam2t:.4f} | {ci2t:.4f} | 0.90 | {cr2t:.4f} | {'PASS' if cr2t<0.1 else 'FAIL'} |\n")
md.append(f"| Level 2b (Bio) | 4 | {lam2b:.4f} | {ci2b:.4f} | 0.90 | {cr2b:.4f} | {'PASS' if cr2b<0.1 else 'FAIL'} |\n")
md.append(f"| Level 2c (Sensory) | 6 | {lam2s:.4f} | {ci2s:.4f} | 1.24 | {cr2s:.4f} | {'PASS' if cr2s<0.1 else 'FAIL'} |\n\n")

all_pass = all(cr < 0.10 for cr in [cr1, cr2t, cr2b, cr2s])
md.append(f"**Overall consistency:** {'ALL PASS ✓' if all_pass else 'SOME FAILURES — review required'}\n\n")

# DONE
out_md = os.path.join(OUT_DIR, 'AHP_Analysis_Report.md')
with open(out_md, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"\nMD Report: {out_md}")
print(f"Length: {len(''.join(md))} chars")
print("\n[DONE] AHP subjective weights calculated.")
