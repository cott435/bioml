import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_parse import get_geoparse
from models import fit_timecourse_ols
from scipy.stats import linregress

expression_df, phenotype_df = get_geoparse()
phenotype_df["hours"] = (
    phenotype_df["title"]
    .str.extract(r"(\d+(?:\.\d+)?)\s*hours", expand=False)
    .astype(float)
)
phenotype_df["ctrl"] = phenotype_df["title"].str.contains(
    "negative control", case=False, na=False
)
phenotype_df = phenotype_df[['hours', 'ctrl']]


# minimal filtering
MIN_EXPR = 4.5  # background cutoff
background_filter = (expression_df > MIN_EXPR).any(axis=1)
exp_filter = expression_df.loc[background_filter]
print(f"Genes after background filter: {exp_filter.shape[0]}")

MIN_VAR = 0.01
var_filter = exp_filter.var(axis=1) > MIN_VAR
exp_filter = exp_filter.loc[var_filter]
print(f"Genes after variance filter: {exp_filter.shape[0]}")

results = fit_timecourse_ols(
     expr_gene_by_sample=exp_filter,
     pheno=phenotype_df,
     time_col="hours",
     type_col="ctrl",
     center_time=True,
     add_timepoint_effects=True,
 )


res = results.copy()
res["abs_beta_treat"] = res["beta_treat"].abs()
res["abs_beta_interaction"] = res["beta_time_c:treat"].abs()
res["log10_fdr_treat"] = -np.log10(res["fdr_treat"] + 1e-300)
res["log10_fdr_interaction"] = -np.log10(res["fdr_interaction"] + 1e-300)

plt.figure(figsize=(6, 5))
plt.scatter(
    res["se_treat"],
    res["beta_treat"],
    c=res["log10_fdr_treat"],
    cmap="viridis",
    s=10
)
plt.axhline(0, color="black", lw=1)
plt.xlabel("Standard Error (SE)")
plt.ylabel("beta_treat (log2 effect)")
plt.title("Effect size vs uncertainty (colored by -log10 FDR)")
plt.colorbar(label="-log10(FDR)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(
    res["beta_treat"],
    res["log10_fdr_treat"],
    s=10,
    alpha=0.6
)
plt.axvline(0, color="black", lw=1)
plt.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
plt.xlabel("beta_treat (log2 effect)")
plt.ylabel("-log10(FDR)")
plt.title("Volcano plot: miR-124 main effect")
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 5))
plt.scatter(
    res["beta_time_c:treat"],
    res["log10_fdr_interaction"],
    s=10,
    alpha=0.6
)
plt.axvline(0, color="black", lw=1)
plt.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
plt.xlabel("beta_time:treat")
plt.ylabel("-log10(FDR)")
plt.title("Time-dependent miR-124 effects")
plt.tight_layout()
plt.show()


hits_main = res[
    (res["fdr_treat"] < 0.05) &
    (res["abs_beta_treat"] > 0.2)
].sort_values("fdr_treat")
hits_time = res[
    (res["fdr_interaction"] < 0.05) &
    (res["abs_beta_interaction"] > 0.01)
].sort_values("fdr_interaction")

early = hits_main[hits_main["effect_treat_at_4h"] < -0.3]
late = hits_main[hits_main["effect_treat_at_16h"] < -0.3]

top_n = 200
candidates = res.nsmallest(top_n, ['fdr_interaction', 'fdr_treat']).index

# Z-score expression per gene
expr_z = exp_filter.loc[candidates].T
expr_z = (expr_z - expr_z.mean()) / expr_z.std()

# Order samples by time, then treatment
sample_order = phenotype_df.sort_values(['ctrl', 'hours']).index
expr_z = expr_z.loc[sample_order]

plt.figure(figsize=(10, 12))
sns.heatmap(expr_z, cmap='RdBu_r', center=0, xticklabels=False, cbar_kws={'label': 'Z-score'})
plt.title(f'Top {top_n} Treatment-Responsive Genes (Z-scored expression)')
plt.ylabel('Genes (ordered by FDR)')
plt.xlabel('Samples (ordered by time then treatment)')
plt.show()


hits = res[(res['fdr_treat'] < 0.05) | (res['fdr_interaction'] < 0.05)].copy()
main_only = res[(res['fdr_treat'] < 0.05) & (res['fdr_interaction'] >= 0.05)]
interaction_only = res[(res['fdr_interaction'] < 0.05) & (res['fdr_treat'] >= 0.05)]
both = res[(res['fdr_treat'] < 0.05) & (res['fdr_interaction'] < 0.05)]



samples = phenotype_df[~phenotype_df['ctrl']].index
time = phenotype_df[~phenotype_df['ctrl']]['hours']
interaction_genes = interaction_only.index
popts=[]
t_norm = time / (time.max() - time.min() + 1e-8)

for gene, y in exp_filter.loc[interaction_genes, samples].iterrows():
    # Design matrix: [1, t, t²]
    X = np.column_stack([np.ones_like(t_norm), t_norm, t_norm ** 2])

    # Fit quadratic model using ordinary least squares
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coeffs  # y ≈ a + b*t + c*t²

    # Check if quadratic term is significant
    # Simple way: compare R² of linear vs quadratic
    # Linear fit
    slope, intercept, r_linear, _, _ = linregress(t_norm, y)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res_linear = np.sum((y - (slope * t_norm + intercept)) ** 2)
    r2_linear = 1 - ss_res_linear / ss_tot

    # Quadratic residuals
    y_pred_quad = X @ coeffs
    ss_res_quad = np.sum((y - y_pred_quad) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot
    improvement = r2_quad - r2_linear
    plt.plot(y)

    if c < 0 and improvement > 0.05:  # significant improvement and concave
        group = "early_response"
    # If quadratic term is large and positive → convex
    elif c > 0 and improvement > 0.05:
        group = "late_response"
    # If quadratic doesn't improve much → linear is sufficient
    elif improvement < 0.03:
        group = "sustained_linear"
    else:
        group = "no_clear_pattern"

    improvement = r2_quad - r2_linear
    popts.append({
        'gene': gene,
        'group': group,
        'y': y
    })

for i in range(15):
    plt.figure()
    plt.plot(popts[i]['y'])
    plt.title(popts[i]['group'])

