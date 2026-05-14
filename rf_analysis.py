import pandas as pd
import numpy as np
import os
from datetime import datetime

# Paths
INPUT_FILE = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
OUT_DIR = r"D:\WQIPaper\DataAnalytics"
OUT_EXCEL = os.path.join(OUT_DIR, "RF_Weights.xlsx")
OUT_MD = os.path.join(OUT_DIR, "RF_Analysis_Report.md")
RANDOM_STATE = 42
N_TREES = 500

print("=" * 60)
print("Step 1: Load & clean data")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')

# Drop dup Free Chlorine
fc_cols = [c for c in df.columns if c.startswith('Free Chlorine')]
if len(fc_cols) > 1:
    df = df.drop(columns=fc_cols[1:])

meta_cols = ['ID', 'Date']
indicator_cols = [c for c in df.columns if c not in meta_cols]
X = df[indicator_cols].values.astype(np.float64)
n, m = X.shape
names = indicator_cols
print(f"Samples: {n}, Indicators: {m}")
print(f"Indicators: {names}")

# Step 2: Build synthetic target via PCA (first component as proxy WQI)
print("\nStep 2: Build synthetic target (PCA 1st component)")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1, random_state=RANDOM_STATE)
target = pca.fit_transform(X_scaled).ravel()
# Normalize target to [0,1] for interpretability
t_min, t_max = target.min(), target.max()
target_norm = (target - t_min) / (t_max - t_min)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")

# Step 3: Train Random Forest
print(f"\nStep 3: Train Random Forest (n_trees={N_TREES})")
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(
    n_estimators=N_TREES,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    oob_score=True
)
rf.fit(X, target)

# Cross-validation
cv_scores = cross_val_score(rf, X, target, cv=5, scoring='r2')
print(f"OOB R2: {rf.oob_score_:.4f}")
print(f"5-fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Step 4: Gini importance
print("\nStep 4: Gini importance (MDI)")
gini_imp = rf.feature_importances_
gini_pct = gini_imp / gini_imp.sum() * 100

# Step 5: Permutation importance
print("Step 5: Permutation importance (MDA)")
perm_result = permutation_importance(
    rf, X, target,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)
perm_imp = perm_result.importances_mean
perm_std = perm_result.importances_std
# Handle negative importance (set to 0)
perm_imp[perm_imp < 0] = 0
perm_pct = perm_imp / perm_imp.sum() * 100 if perm_imp.sum() > 0 else np.zeros_like(perm_imp)

# Step 6: Combined importance (average of Gini + Permutation)
print("Step 6: Combined importance")
combined_pct = (gini_pct + perm_pct) / 2

# Step 7: Build results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

results = pd.DataFrame({
    'Indicators': names,
    'Gini_Importance_raw': gini_imp,
    'Gini_Importance': [f"{v:.2f} %" for v in gini_pct],
    'Permutation_Importance_raw': perm_imp,
    'Permutation_Importance': [f"{v:.2f} %" for v in perm_pct],
    'Permutation_Std': perm_std,
    'Combined_Weight_raw': combined_pct,
    'Combined_Weight': [f"{v:.2f} %" for v in combined_pct],
})

results = results.sort_values('Combined_Weight_raw', ascending=False).reset_index(drop=True)

print("\n--- Gini Importance (MDI) ---")
for _, row in results.iterrows():
    print(f"  {row['Indicators']:40s} {row['Gini_Importance']:>8s}")

print("\n--- Permutation Importance (MDA) ---")
for _, row in results.iterrows():
    print(f"  {row['Indicators']:40s} {row['Permutation_Importance']:>8s}  (+/- {row['Permutation_Std']:.6f})")

print("\n--- Combined Weights ---")
for _, row in results.iterrows():
    print(f"  {row['Indicators']:40s} {row['Combined_Weight']:>8s}")

# Save Excel
col_order = ['Indicators', 'Gini_Importance', 'Permutation_Importance', 
             'Permutation_Std', 'Combined_Weight']
results_excel = results[col_order].copy()
# Also save raw values
results_excel['Gini_raw'] = results['Gini_Importance_raw']
results_excel['Permutation_raw'] = results['Permutation_Importance_raw']
results_excel['Combined_raw'] = results['Combined_Weight_raw']
results_excel.to_excel(OUT_EXCEL, index=False)
print(f"\nExcel: {OUT_EXCEL}")

# MD report
md = []
md.append("# Random Forest Feature Importance Analysis\n\n")
md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
md.append(f"**Input file:** `{INPUT_FILE}`  \n")
md.append(f"**Samples:** {n} | **Indicators:** {m}  \n")
md.append(f"**Model:** Random Forest Regressor (n_estimators={N_TREES}, max_features='sqrt')  \n")
md.append(f"**Target:** PCA 1st component (explained variance: {pca.explained_variance_ratio_[0]:.2%})  \n")
md.append(f"**Performance:** OOB R2={rf.oob_score_:.4f}, 5-CV R2={cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})  \n\n")

md.append("---\n## Methodology\n\n")
md.append("### 1. Target Construction (PCA)\n")
md.append("Since no explicit WQI target exists, the first principal component (PC1) of all standardized ")
md.append("indicators was used as a proxy for 'overall water quality'. PC1 captures the dominant direction ")
md.append(f"of variation in the data, explaining **{pca.explained_variance_ratio_[0]:.2%}** of total variance.\n\n")

md.append("### 2. Random Forest Training\n")
md.append(f"- **Estimators:** {N_TREES}\n")
md.append("- **Max features:** sqrt(m)\n")
md.append("- **Random state:** {}\n".format(RANDOM_STATE))
md.append(f"- **OOB Score (R2):** {rf.oob_score_:.4f}\n")
md.append(f"- **5-fold CV R2:** {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})\n\n")

md.append("### 3. Gini Importance (MDI - Mean Decrease in Impurity)\n")
md.append("Measures how much each feature reduces node impurity (variance) across all trees. ")
md.append("Features that create purer splits receive higher importance.\n\n")

md.append("### 4. Permutation Importance (MDA - Mean Decrease in Accuracy)\n")
md.append("Measures the decrease in model performance when a feature's values are randomly shuffled. ")
md.append("A larger drop indicates higher importance. Uses 30 permutation repeats for stability.\n\n")

md.append("### 5. Combined Weight\n")
md.append("Average of Gini and Permutation importance, providing a balanced view of feature relevance.\n\n")

md.append("---\n## Results\n\n")

md.append("### Gini Importance (MDI)\n\n")
md.append("| Indicators | Weight |\n|-----------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Gini_Importance']} |\n")

md.append("\n### Permutation Importance (MDA)\n\n")
md.append("| Indicators | Weight | Std Dev |\n|-----------|--------|--------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Permutation_Importance']} | {row['Permutation_Std']:.6f} |\n")

md.append("\n### Combined Weights (Final)\n\n")
md.append("| Indicators | Combined Weight |\n|-----------|---------------|\n")
for _, row in results.iterrows():
    md.append(f"| {row['Indicators']} | {row['Combined_Weight']} |\n")

md.append("\n### Weight Distribution\n\n```\n")
max_name = max(len(n) for n in names)
for _, row in results.iterrows():
    bar = '#' * int(row['Combined_Weight_raw'] * 2)
    md.append(f"{row['Indicators']:<{max_name+2}} {bar} {row['Combined_Weight']}\n")
md.append("```\n\n")

md.append("## Key Insights\n\n")

# Top features
top3 = results.head(3)
md.append("**Top 3 by combined weight:**\n")
for _, row in top3.iterrows():
    md.append(f"- **{row['Indicators']}** ({row['Combined_Weight']})\n")

# Compare Gini vs Permutation
md.append("\n**Gini vs Permutation comparison:**\n")
md.append("- **Gini importance** favors features with many split points (continuous, high-variance features)\n")
md.append("- **Permutation importance** reflects actual predictive contribution (more robust to correlated features)\n")

# Compare with EWM if available
ewm_path = os.path.join(OUT_DIR, "EWM_Weights.xlsx")
md.append("\n**Comparison with EWM:**\n")
if os.path.exists(ewm_path):
    md.append("See `EWM_Weights.xlsx` for entropy-based weights. EWM captures statistical information content, ")
    md.append("while RF captures predictive importance for the overall water quality pattern.\n")
    md.append("- EWM: data-driven via information entropy (no target needed)\n")
    md.append("- RF: supervised learning via predictive modeling (synthetic PCA target)\n")
    md.append("- Combining both methods provides a more robust weighting scheme for WQI construction.\n")
else:
    md.append("(EWM results not found - run EWM analysis first for method comparison)\n")

md.append(f"\n> **Note:** RF importance depends on the target variable. Here PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained) ")
md.append("was used as a proxy for overall water quality. Different proxy targets may yield different importance rankings.\n")

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(''.join(md))

print(f"MD report: {OUT_MD}")
print("\n[DONE] RF Analysis Complete!")
