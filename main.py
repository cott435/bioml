import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from data_parse import get_geoparse
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

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

# View raw data as is (already log transformed)
expression_df.hist(bins=100)
expression_df.boxplot()

# Ensure same sample order
samples = phenotype_df.index
X = expression_df.T
assert all(X.index == phenotype_df.index)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(
    X_pca,
    index=X.index,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

pca_df = pca_df.join(phenotype_df)
pca_df['hours_cat'] = pca_df['hours'].astype(str)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="hours_cat",
    palette="viridis",
    s=80
)
plt.title("PCA of samples (colored by time)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="ctrl",
    style="ctrl",
    s=80
)
plt.title("PCA of samples (colored by treatment)")
plt.tight_layout()
plt.show()


# minimal filtering
MIN_EXPR = 4.5  # background cutoff
background_filter = (expression_df > MIN_EXPR).any(axis=1)
exp_filter = expression_df.loc[background_filter]
print(f"Genes after background filter: {exp_filter.shape[0]}")

MIN_VAR = 0.01
var_filter = exp_filter.var(axis=1) > MIN_VAR
exp_filter = exp_filter.loc[var_filter]
print(f"Genes after variance filter: {exp_filter.shape[0]}")


plt.hist(expression_df.values.flatten(), bins=100, alpha=0.5, label="All genes")
plt.hist(exp_filter.values.flatten(), bins=100, alpha=0.5, label="Filtered")
plt.legend()
plt.title("Expression distributions")
plt.show()


def fit_timecourse_ols(
    expr_gene_by_sample: pd.DataFrame,
    pheno: pd.DataFrame,
    time_col: str = "time",
    type_col: str = "type",
    control_label: str = "negative_control",
    center_time: bool = True,
    add_timepoint_effects: bool = True,
) -> pd.DataFrame:
    """
    Per-gene OLS model:
        y ~ 1 + time + treat + time:treat

    expr_gene_by_sample: rows=genes, cols=samples (log2 expression)
    pheno: index=samples, columns include time_col (numeric hours) and type_col (control vs treated)

    Returns a results DataFrame indexed by gene with betas, pvals, and FDRs.
    """
    # --- Align samples ---
    samples = pheno.index
    if not set(samples).issubset(expr_gene_by_sample.columns):
        missing = set(samples) - set(expr_gene_by_sample.columns)
        raise ValueError(f"Samples in pheno missing from expr columns: {sorted(list(missing))[:10]} ...")
    expr = expr_gene_by_sample.loc[:, samples]

    # --- Build design matrix ---
    time = pd.to_numeric(pheno[time_col], errors="raise").astype(float).to_numpy()

    # treatment indicator: 0=control, 1=treated (anything not equal to control_label is treated)
    treat = ~pheno[type_col].to_numpy().astype(int)+2

    if center_time:
        time_c = time - np.mean(time)
    else:
        time_c = time.copy()

    X = np.column_stack([
        np.ones_like(time_c),          # intercept
        time_c,                        # time
        treat,                         # treatment main effect
        time_c * treat,                # interaction
    ])
    X_cols = ["Intercept", "time_c", "treat", "time_c:treat"]

    # Precompute for speed
    XtX_inv = np.linalg.inv(X.T @ X)
    H = XtX_inv @ X.T  # (p x n)

    # --- Fit per gene ---
    Y = expr.to_numpy(dtype=float)          # (G x N)
    G, N = Y.shape
    P = X.shape[1]

    betas = (H @ Y.T).T                     # (G x P)
    resid = Y - (betas @ X.T)               # (G x N)

    dof = N - P
    sigma2 = (resid**2).sum(axis=1) / dof   # (G,)
    se = np.sqrt(sigma2[:, None] * np.diag(XtX_inv)[None, :])  # (G x P)

    # t-stats and 2-sided p-values (use statsmodels' survival function for numerical stability)
    # statsmodels uses scipy under the hood; if unavailable, fall back to normal approx.
    try:
        from scipy import stats
        tvals = betas / se
        pvals = 2.0 * stats.t.sf(np.abs(tvals), df=dof)
    except Exception:
        # Normal approximation fallback
        from math import erf, sqrt
        tvals = betas / se
        # p = 2*(1-Phi(|t|))
        pvals = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(lambda z: erf(z / sqrt(2.0)))(np.abs(tvals))))

    res = pd.DataFrame(
        betas,
        index=expr.index,
        columns=[f"beta_{c}" for c in X_cols],
    )
    for j, c in enumerate(X_cols):
        res[f"se_{c}"] = se[:, j]
        res[f"t_{c}"] = tvals[:, j]
        res[f"p_{c}"] = pvals[:, j]

    # Multiple-testing correction on the key hypothesis tests
    # - treat main effect: overall shift between treated vs control at mean time (if centered)
    # - interaction: difference in slope over time
    res["fdr_treat"] = multipletests(res["p_treat"].to_numpy(), method="fdr_bh")[1]
    res["fdr_interaction"] = multipletests(res["p_time_c:treat"].to_numpy(), method="fdr_bh")[1]

    # Optional: compute treated-vs-control effect at each observed timepoint
    # If centered: effect(t) = beta_treat + beta_int * (t - mean_t)
    if add_timepoint_effects:
        unique_times = np.sort(np.unique(time))
        mean_t = np.mean(time) if center_time else 0.0
        b_treat = res["beta_treat"].to_numpy()
        b_int = res["beta_time_c:treat"].to_numpy()

        for t in unique_times:
            tc = (t - mean_t) if center_time else t
            res[f"effect_treat_at_{t:g}h"] = b_treat + b_int * tc

    return res


# -----------------------
# Example usage
# -----------------------
# expr_filt: genes x samples (already minimally filtered)
# pheno_df: samples x metadata, with columns e.g.:
#   pheno_df["time_hours"] (numeric)
#   pheno_df["type"] (e.g., "negative_control" vs "miR-124")
#
results = fit_timecourse_ols(
     expr_gene_by_sample=exp_filter,
     pheno=phenotype_df,
     time_col="hours",
     type_col="ctrl",
     control_label="negative_control",
     center_time=True,
     add_timepoint_effects=True,
 )
#
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

