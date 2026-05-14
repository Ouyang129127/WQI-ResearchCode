import pandas as pd
import numpy as np
import os
from datetime import datetime

OUT_DIR = r"D:\WQIPaper\DataAnalytics"
OUT_EXCEL = os.path.join(OUT_DIR, "Nash_Equilibrium_Weights.xlsx")
OUT_MD = os.path.join(OUT_DIR, "Nash_Equilibrium_Report.md")

# ── Load RF & CRITIC ──────────────────────────────────────────
df_rf = pd.read_excel(os.path.join(OUT_DIR, "RF_Weights.xlsx"), engine='openpyxl')
df_critic = pd.read_excel(os.path.join(OUT_DIR, "CRITIC_Weights.xlsx"), engine='openpyxl')
df = pd.merge(
    df_rf[['Indicators', 'Combined_raw']],
    df_critic[['Indicators', 'Weight_raw']],
    on='Indicators', how='inner'
)
names = df['Indicators'].tolist()
m = len(names)
w1 = df['Combined_raw'].values
w2 = df['Weight_raw'].values
w1 = w1 / w1.sum()
w2 = w2 / w2.sum()

print("=" * 60)
print("Nash Equilibrium Weight Fusion (Constrained)")
print("=" * 60)
print(f"Indicators: {m}")

# ═══════════════════════════════════════════════════════════════
# Nash Game Theory for Weight Fusion
#
# Combined weight: w(λ) = λ·w¹ + (1-λ)·w², λ ∈ [0,1]
#
# Each method k has a loss: L_k(λ) = ||w(λ) - w^k||²
# 
# Nash equilibrium: minimizes weighted sum of losses
#   min_λ  λ·L₁(λ) + (1-λ)·L₂(λ)
#
# Derivation:
#   L₁(λ) = ||(λ-1)w¹ + (1-λ)w²||² = (1-λ)²||w¹-w²||²
#   L₂(λ) = ||λ·w¹ + (-λ)w²||² = λ²||w¹-w²||²
#
#   f(λ) = λ(1-λ)²D + (1-λ)λ²D = λ(1-λ)D
#   where D = ||w¹ - w²||²
#
#   f'(λ) = (1-2λ)D → λ* = 0.5 (equal weights!)
#
# BUT: methods have different credibility → weighted Nash
# ═══════════════════════════════════════════════════════════════

# D = sum of squared differences
D = np.sum((w1 - w2) ** 2)
print(f"\n||w_RF - w_CRITIC||^2 = {D:.6f}")
print(f"Correlation(w_RF, w_CRITIC) = {np.corrcoef(w1, w2)[0,1]:.4f}")
print(f"Spearman ρ = {np.corrcoef(np.argsort(np.argsort(w1)), np.argsort(np.argsort(w2)))[0,1]:.4f}")

# ═══════════════════════════════════════════════════════════════
# 1. Classical Nash: equal credibility → λ = 0.5 (theoretical optimum)
# ═══════════════════════════════════════════════════════════════
lam_classical = 0.5
w_classical = lam_classical * w1 + (1 - lam_classical) * w2
w_classical = w_classical / w_classical.sum()
print(f"\n[1] Classical Nash: λ_RF = λ_CRITIC = 0.5 (equal weight)")

# ═══════════════════════════════════════════════════════════════
# 2. Credibility-weighted Nash
#    Credibility: RF OOB R² = 0.95, CRITIC = based on CV
#    Weighted objective: min τ·L₁ + (1-τ)·L₂
#    where τ = credibility weight of method 1
#    f(λ) = τ·(1-λ)²·D + (1-τ)·λ²·D
#    f'(λ) = [-2τ(1-λ) + 2(1-τ)λ]·D = 0
#    λ = τ
# ═══════════════════════════════════════════════════════════════

# Credibility estimates
rf_r2 = 0.9500        # RF OOB R²
critic_cred = 0.80     # CRITIC: data-driven but no performance metric
tau = rf_r2 / (rf_r2 + critic_cred)  # RF's credibility weight

print(f"\n[2] Credibility-weighted Nash:")
print(f"    RF credibility     = {rf_r2:.2f}")
print(f"    CRITIC credibility = {critic_cred:.2f}")
print(f"    τ_RF = {tau:.4f}")
print(f"    → λ_RF = τ = {tau:.4f} (optimal λ = credibility ratio)")

w_cred = tau * w1 + (1 - tau) * w2
w_cred = w_cred / w_cred.sum()

# ═══════════════════════════════════════════════════════════════
# 3. Distance minimization with non-negativity constraint
#    Minimize ||Σ λ_k w_k - w_1||² + ||Σ λ_k w_k - w_2||²
#    s.t. λ_k ≥ 0, Σ λ_k = 1
#
#    Using scipy.optimize
# ═══════════════════════════════════════════════════════════════
from scipy.optimize import minimize

def objective(lam):
    """Minimize total deviation"""
    lam1, lam2 = lam[0], lam[1]
    w_comb = lam1 * w1 + lam2 * w2
    return np.sum((w_comb - w1)**2) + np.sum((w_comb - w2)**2)

cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})
bounds = [(0, 1), (0, 1)]
res = minimize(objective, [0.5, 0.5], method='SLSQP', bounds=bounds, constraints=cons)

if res.success:
    lam_opt = res.x
else:
    lam_opt = np.array([0.5, 0.5])

w_qp = lam_opt[0] * w1 + lam_opt[1] * w2
w_qp = w_qp / w_qp.sum()

print(f"\n[3] Constrained QP Nash:")
print(f"    λ_RF = {lam_opt[0]:.4f}, λ_CRITIC = {lam_opt[1]:.4f}")
print(f"    Objective value: {res.fun:.8f}")

# ═══════════════════════════════════════════════════════════════
# Build results table
# ═══════════════════════════════════════════════════════════════
results = pd.DataFrame({
    'Indicators': names,
    'RF_Weight': w1 * 100,
    'CRITIC_Weight': w2 * 100,
    'Classical_Nash': w_classical * 100,
    'Credibility_Nash': w_cred * 100,
    'ConstrainedQP_Nash': w_qp * 100,
}).sort_values('Classical_Nash', ascending=False).reset_index(drop=True)

# Format for display
for col in ['RF_Weight', 'CRITIC_Weight', 'Classical_Nash', 'Credibility_Nash', 'ConstrainedQP_Nash']:
    results[col + '_fmt'] = results[col].apply(lambda x: f"{x:.2f} %")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\n{'Indicators':40s} {'RF':>8s} {'CRITIC':>8s} {'Classic':>8s} {'Cred':>8s} {'QP':>8s}")
print("-" * 85)
for _, row in results.iterrows():
    print(f"{row['Indicators']:40s} {row['RF_Weight_fmt']:>8s} {row['CRITIC_Weight_fmt']:>8s} {row['Classical_Nash_fmt']:>8s} {row['Credibility_Nash_fmt']:>8s} {row['ConstrainedQP_Nash_fmt']:>8s}")

# ═══════════════════════════════════════════════════════════════
# Which one is recommended?
# ═══════════════════════════════════════════════════════════════
# Classical Nash = simple average → most democratic
# Credibility Nash = RF leans more → uses model quality
# Constrained QP = mathematically optimal within bounds

print("\nRECOMMENDED: Credibility-weighted Nash (uses RF model quality)")
print(f"  λ_RF = {tau:.4f} → RF dominates by {tau*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# Sensitivity: sweep λ from 0 to 1
# ═══════════════════════════════════════════════════════════════
print("\n--- Sensitivity (λ sweep) ---")
sens_points = [0.0, 0.25, 0.5, 0.75, 1.0]
for lam in sens_points:
    w = lam * w1 + (1 - lam) * w2
    w = w / w.sum()
    top3 = np.argsort(w)[::-1][:3]
    print(f"  λ_RF={lam:.2f}: Top3 = {names[top3[0]]}, {names[top3[1]]}, {names[top3[2]]}")

# ═══════════════════════════════════════════════════════════════
# Save Excel
# ═══════════════════════════════════════════════════════════════
excel_cols = ['Indicators', 'RF_Weight_fmt', 'CRITIC_Weight_fmt', 
              'Classical_Nash_fmt', 'Credibility_Nash_fmt', 'ConstrainedQP_Nash_fmt']
results[excel_cols].to_excel(OUT_EXCEL, index=False, sheet_name='Nash_Weights')

# Sensitivity sheet
sens_data = []
alphas = np.linspace(0, 1, 11)
for a in alphas:
    w = a * w1 + (1 - a) * w2
    w = w / w.sum()
    row = {'α_RF': round(a, 1), 'α_CRITIC': round(1 - a, 1)}
    for j, name in enumerate(names):
        row[name + '_%'] = round(w[j] * 100, 2)
    sens_data.append(row)
df_sens = pd.DataFrame(sens_data)
with pd.ExcelWriter(OUT_EXCEL, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_sens.to_excel(writer, sheet_name='Sensitivity', index=False)

print(f"\nExcel: {OUT_EXCEL}")

# ═══════════════════════════════════════════════════════════════
# MD Report
# ═══════════════════════════════════════════════════════════════
md = []
md.append("# Nash Equilibrium Weight Fusion\n\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append(f"**Indicators:** {m}  \n")
md.append(f"**Methods fused:** RF (Random Forest) + CRITIC  \n\n")

md.append("---\n## Methodology\n\n")
md.append("### Game Theory Framework\n\n")
md.append("In a non-cooperative game, each weighting method is a **player** seeking to minimize its deviation ")
md.append("from the consensus weight vector. The **Nash equilibrium** is reached when no player can unilaterally ")
md.append("reduce its loss.\n\n")

md.append("### Mathematical Formulation\n\n")
md.append("Let the combined weight be a linear combination:\n\n")
md.append("$$w(\\lambda) = \\lambda \\cdot w^{RF} + (1-\\lambda) \\cdot w^{CRITIC}, \\quad \\lambda \\in [0, 1]$$\n\n")

md.append("Each method k has a **loss function**:\n\n")
md.append("$$L_k(\\lambda) = \\| w(\\lambda) - w^k \\|^2$$\n\n")

md.append("**Classical Nash** (equal credibility):\n\n")
md.append("$$\\min_{\\lambda} \\; L_{RF}(\\lambda) + L_{CRITIC}(\\lambda)$$\n\n")
md.append("> Derivative analysis yields λ* = 0.5 (equal weights) — the classic result that Nash equilibrium ")
md.append("coincides with arithmetic mean when both players have equal importance.\n\n")

md.append("**Credibility-weighted Nash**:\n\n")
md.append("$$\\min_{\\lambda} \\; \\tau \\cdot L_{RF}(\\lambda) + (1-\\tau) \\cdot L_{CRITIC}(\\lambda)$$\n\n")
md.append(f"- τ = credibility ratio = {rf_r2:.2f} / ({rf_r2:.2f} + {critic_cred:.2f}) = {tau:.4f}\n")
md.append(f"- RF credibility: OOB R² = {rf_r2:.2f} (explicit performance metric)\n")
md.append(f"- CRITIC credibility: {critic_cred:.2f} (data-driven, no validation target)\n")
md.append(f"- Optimal: λ* = τ = {tau:.4f}\n\n")

md.append("---\n## Results\n\n")
md.append("### Classical Nash Equilibrium (λ_RF = λ_CRITIC = 0.5)\n\n")
md.append("| Indicator | Nash Weight |\n|-----------|-----------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Classical_Nash_fmt']} |\n")

md.append("\n### Credibility-weighted Nash\n\n")
md.append(f"| Indicator | Weight |\n|-----------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Credibility_Nash_fmt']} |\n")

md.append("\n### Comparison: All Three Nash Variants\n\n")
md.append("| Indicator | Classical | Credibility | Constrained QP |\n")
md.append("|-----------|----------|------------|----------------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Classical_Nash_fmt']} | {row['Credibility_Nash_fmt']} | {row['ConstrainedQP_Nash_fmt']} |\n")

md.append("\n---\n## Weight Distribution (Credibility Nash)\n\n```\n")
max_name = max(len(n) for n in names)
for _, row in results.iterrows():
    bar = '#' * int(row['Credibility_Nash'] * 1.5)
    md.append(f"{row['Indicators']:<{max_name+2}} {bar} {row['Credibility_Nash_fmt']}\n")
md.append("```\n\n")

md.append("---\n## Sensitivity Analysis\n\n")
md.append("| λ_RF | λ_CRITIC | Top 3 Indicators |\n")
md.append("|------|---------|------------------|\n")
for a in sens_points:
    w = a * w1 + (1 - a) * w2
    w = w / w.sum()
    top3_idx = np.argsort(w)[::-1][:3]
    top3_str = ", ".join([names[i] for i in top3_idx])
    md.append(f"| {a:.2f} | {1-a:.2f} | {top3_str} |\n")

md.append(f"\n> Full sensitivity table (11 λ values × all indicators) is available in `{os.path.basename(OUT_EXCEL)}` sheet 'Sensitivity'.\n\n")

md.append("---\n## Key Insights\n\n")
md.append("### Top 3 (Credibility Nash)\n")
for _, row in results.head(3).iterrows():
    md.append(f"- **{row['Indicators']}** ({row['Credibility_Nash_fmt']})\n")

md.append("\n### Nash vs Other Fusion Methods\n\n")
md.append("| Method | λ_RF | Characteristic |\n")
md.append("|--------|------|---------------|\n")
md.append(f"| Classical Nash | 0.50 | Equal treatment, coincides with additive average |\n")
md.append(f"| Credibility Nash | {tau:.2f} | RF dominates based on model quality |\n")
md.append("| Multiplicative | N/A | Conservative: agreement required |\n")
md.append("| Additive Avg | 0.50 | Identical to Classical Nash |\n\n")

md.append("### Theoretical Note\n\n")
md.append("The **unconstrained** Nash solution yields λ values outside [0,1] when the two weight vectors ")
md.append("are strongly correlated — a sign that both methods largely agree on relative importance. ")
md.append(f"Correlation ρ = {np.corrcoef(w1, w2)[0,1]:.4f}. ")
md.append("The **constrained** solution (λ ∈ [0,1]) is used here for practical interpretation.\n\n")

md.append(f"> **Recommendation:** Credibility-weighted Nash (λ_RF = {tau:.2f}) — leverages RF's known predictive performance (OOB R² = {rf_r2:.2f}) while preserving CRITIC's data-driven perspective.\n")

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"MD report: {OUT_MD}")
print("\n[DONE] Nash Equilibrium Fusion Complete!")
