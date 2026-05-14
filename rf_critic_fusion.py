import pandas as pd
import numpy as np
import os
from datetime import datetime

OUT_DIR = r"D:\WQIPaper\DataAnalytics"
OUT_EXCEL = os.path.join(OUT_DIR, "RF_CRITIC_Fusion_Weights.xlsx")
OUT_MD = os.path.join(OUT_DIR, "RF_CRITIC_Fusion_Report.md")

# Load RF weights
rf_file = os.path.join(OUT_DIR, "RF_Weights.xlsx")
df_rf = pd.read_excel(rf_file, engine='openpyxl')

# Load CRITIC weights
critic_file = os.path.join(OUT_DIR, "CRITIC_Weights.xlsx")
df_critic = pd.read_excel(critic_file, engine='openpyxl')

# Merge by indicator name to align
df_merged = pd.merge(
    df_rf[['Indicators', 'Combined_raw']], 
    df_critic[['Indicators', 'Weight_raw']], 
    on='Indicators', how='inner'
)
names = df_merged['Indicators'].tolist()
m = len(names)
rf_raw = df_merged['Combined_raw'].values
critic_raw = df_merged['Weight_raw'].values

# Convert to proportions (0-1)
rf_w = rf_raw / rf_raw.sum()
critic_w = critic_raw / critic_raw.sum()

print("=" * 60)
print("RF + CRITIC Weight Fusion")
print("=" * 60)

print(f"RF sum: {rf_w.sum():.4f}, CRITIC sum: {critic_w.sum():.4f}")

# ================================================================
# Fusion Strategy 1: Multiplicative Synthesis (most common)
# w_j = (w1_j * w2_j) / sum(w1_j * w2_j)
# ================================================================
print("\n--- Fusion 1: Multiplicative Synthesis ---")
mult_raw = rf_w * critic_w
mult_w = mult_raw / mult_raw.sum()

# ================================================================
# Fusion Strategy 2: Additive (Simple Average)
# ================================================================
print("--- Fusion 2: Additive Average ---")
add_raw = (rf_w + critic_w) / 2
add_w = add_raw / add_raw.sum()

# ================================================================
# Fusion Strategy 3: Minimum Relative Entropy (MRE)
# Find w that minimizes: w * ln(w/w_rf) + w * ln(w/w_critic)
# Solution: w_j = sqrt(w_rf_j * w_critic_j) (unnormalized geometric mean)
# ================================================================
print("--- Fusion 3: Geometric Mean (MRE solution) ---")
geo_raw = np.sqrt(rf_w * critic_w)
geo_w = geo_raw / geo_raw.sum()

# ================================================================
# Fusion Strategy 4: Game Theory / Lagrange
# Minimize sum of squared deviations from both
# w_j = (w_rf_j + w_critic_j) / 2  (same as additive)
# So we use the additive as game theory solution
# ================================================================

# ================================================================
# Fusion Strategy 5: TOPSIS-style
# Rank-based: average of ranks, then convert to weights
# ================================================================
print("--- Fusion 5: Rank Aggregation ---")
rf_rank = rf_raw.argsort()[::-1].argsort() + 1  # 1=best
critic_rank = critic_raw.argsort()[::-1].argsort() + 1
avg_rank = (rf_rank + critic_rank) / 2
rank_score = (m + 1 - avg_rank)  # higher score = better rank
rank_raw = rank_score / rank_score.sum()

# ================================================================
# Compile all results
# ================================================================
print("\n" + "=" * 60)
print("FUSION RESULTS")
print("=" * 60)

results = pd.DataFrame({
    'Indicators': names,
    'RF_Weight': [f"{w:.2f} %" for w in rf_w * 100],
    'CRITIC_Weight': [f"{w:.2f} %" for w in critic_w * 100],
    'Multiplicative_raw': mult_w,
    'Multiplicative': [f"{w:.2f} %" for w in mult_w * 100],
    'Geometric_raw': geo_w,
    'Geometric_Mean': [f"{w:.2f} %" for w in geo_w * 100],
    'Additive_raw': add_w,
    'Additive_Avg': [f"{w:.2f} %" for w in add_w * 100],
    'Rank_Fusion_raw': rank_raw,
    'Rank_Fusion': [f"{w:.2f} %" for w in rank_raw * 100],
})

# Use multiplicative as primary, sort by it
results = results.sort_values('Multiplicative_raw', ascending=False).reset_index(drop=True)

# Print all
for _, row in results.iterrows():
    print(f"\n{row['Indicators']}")
    print(f"  RF: {row['RF_Weight']:>8s}  CRITIC: {row['CRITIC_Weight']:>8s}")
    print(f"  Multiplicative: {row['Multiplicative']:>8s}  Geometric: {row['Geometric_Mean']:>8s}  Additive: {row['Additive_Avg']:>8s}  Rank: {row['Rank_Fusion']:>8s}")

# ================================================================
# Determine the recommended fusion
# ================================================================
# Multiplicative synthesis is recommended: it's most conservative (zero if either is zero)
# and widely used in WQI literature
final_w = mult_w
final_name = "Multiplicative Synthesis"

print("\n" + "=" * 60)
print(f"RECOMMENDED: {final_name}")
print("=" * 60)
for _, row in results.iterrows():
    print(f"  {row['Indicators']:40s} {row['Multiplicative']:>8s}")

# ================================================================
# Save Excel
# ================================================================
excel_cols = ['Indicators', 'RF_Weight', 'CRITIC_Weight', 'Multiplicative', 
              'Geometric_Mean', 'Additive_Avg', 'Rank_Fusion']
results_excel = results[excel_cols].copy()
# Add raw values
results_excel['Multiplicative_raw'] = results['Multiplicative_raw']
results_excel['Geometric_raw'] = results['Geometric_raw']
results_excel['Additive_raw'] = results['Additive_raw']
results_excel['Rank_Fusion_raw'] = results['Rank_Fusion_raw']
results_excel.to_excel(OUT_EXCEL, index=False)
print(f"\nExcel: {OUT_EXCEL}")

# ================================================================
# MD Report
# ================================================================
md = []
md.append("# RF + CRITIC Weight Fusion Report\n\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append(f"**Indicators:** {m}  \n\n")

md.append("---\n## Methodology\n\n")
md.append("Four fusion strategies were applied to combine RF and CRITIC weights:\n\n")

md.append("### 1. Multiplicative Synthesis (Recommended)\n")
md.append("$$w_j = \\frac{w_j^{RF} \\cdot w_j^{CRITIC}}{\\sum_{k} w_k^{RF} \\cdot w_k^{CRITIC}}$$\n")
md.append("- Most conservative: an indicator must be important in BOTH methods to get high weight\n")
md.append("- Widely cited in water quality index literature\n")
md.append("- Effectively zeroes out indicators that one method considers unimportant\n\n")

md.append("### 2. Geometric Mean (Minimum Relative Entropy)\n")
md.append("$$w_j \\propto \\sqrt{w_j^{RF} \\cdot w_j^{CRITIC}}$$\n")
md.append("- Compromise solution minimizing KL divergence to both parent distributions\n")
md.append("- Less extreme than multiplicative, more extreme than additive\n\n")

md.append("### 3. Additive Average (Game Theory)\n")
md.append("$$w_j = \\frac{w_j^{RF} + w_j^{CRITIC}}{2}$$\n")
md.append("- Most balanced: gives equal voice to both methods\n")
md.append("- Solution to the cooperative game with equal player weights\n\n")

md.append("### 4. Rank Aggregation\n")
md.append("- Ranks indicators by each method, averages the ranks, converts to weights\n")
md.append("- Robust to magnitude differences between methods\n\n")

md.append("---\n## Source Weights\n\n")
md.append("### RF Combined Weights\n")
md.append("| Indicator | Weight |\n|-----------|--------|\n")
for name, w in zip(names, rf_w):
    md.append(f"| {name} | {w*100:.2f} % |\n")

md.append("\n### CRITIC Weights\n")
md.append("| Indicator | Weight |\n|-----------|--------|\n")
for name, w in zip(names, critic_w):
    md.append(f"| {name} | {w*100:.2f} % |\n")

md.append("\n---\n## Fusion Results\n\n")
md.append("| Indicators | RF | CRITIC | Multiplicative | Geometric | Additive | Rank |\n")
md.append("|-----------|-----|--------|---------------|-----------|----------|------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['RF_Weight']} | {row['CRITIC_Weight']} | {row['Multiplicative']} | {row['Geometric_Mean']} | {row['Additive_Avg']} | {row['Rank_Fusion']} |\n")

md.append("\n---\n## Recommended Weights (Multiplicative Synthesis)\n\n")
md.append("| Indicator | Weight |\n|-----------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Multiplicative']} |\n")

md.append("\n### Distribution\n\n```\n")
max_name = max(len(n) for n in names)
for _, row in results.iterrows():
    bar = '#' * int(row['Multiplicative_raw'] * 200)
    md.append(f"{row['Indicators']:<{max_name+2}} {bar} {row['Multiplicative']}\n")
md.append("```\n\n")

md.append("## Fusion Strategy Comparison\n\n")
md.append("| Strategy | Formula | Best For |\n")
md.append("|----------|---------|----------|\n")
md.append("| Multiplicative | w ∝ w_rf × w_critic | Conservative, agreement-required |\n")
md.append("| Geometric Mean | w ∝ √(w_rf × w_critic) | Balanced compromise |\n")
md.append("| Additive Average | w ∝ w_rf + w_critic | Equal method contribution |\n")
md.append("| Rank Aggregation | w ∝ 1/avg_rank | Robust to magnitude differences |\n")

md.append("\n## Key Insights\n\n")
top3 = results.head(3)
md.append("**Top 3 by multiplicative fusion:**\n")
for _, row in top3.iterrows():
    md.append(f"- **{row['Indicators']}** ({row['Multiplicative']})\n")

md.append("\n**Why multiplicative synthesis?**\n")
md.append("- RF captures **predictive importance** (what helps predict overall water quality)\n")
md.append("- CRITIC captures **information richness** (variability + independence)\n")
md.append("- Multiplicative fusion ensures an indicator is only highly weighted if BOTH methods agree it matters\n")
md.append("- This reduces the risk of over-weighting noisy or method-specific artifacts\n")

# Agreement analysis
md.append("\n**Method agreement analysis:**\n")
from scipy.stats import spearmanr
rho, pval = spearmanr(rf_raw, critic_raw)
md.append(f"- Spearman rank correlation between RF and CRITIC: **ρ = {rho:.3f}** (p = {pval:.4f})\n")
if rho > 0.5:
    md.append("- Moderate-to-strong agreement: the two methods largely converge on which indicators matter\n")
elif rho > 0.2:
    md.append("- Weak agreement: the methods have different perspectives on indicator importance\n")
else:
    md.append("- Very weak agreement: the methods fundamentally disagree on importance rankings\n")

md.append(f"\n> **Note:** All raw data, weights, and fusion results are available in `{os.path.basename(OUT_EXCEL)}`.\n")

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"MD report: {OUT_MD}")
print("\n[DONE] RF+CRITIC Fusion Complete!")
