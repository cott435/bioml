import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.linear_model import LinearRegression

def fit_timecourse_ols(
    expr_gene_by_sample: pd.DataFrame,
    pheno: pd.DataFrame,
    time_col: str = "time",
    type_col: str = "type",
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
    samples = pheno.index
    if not set(samples).issubset(expr_gene_by_sample.columns):
        missing = set(samples) - set(expr_gene_by_sample.columns)
        raise ValueError(f"Samples in pheno missing from expr columns: {sorted(list(missing))[:10]} ...")
    expr = expr_gene_by_sample.loc[:, samples]

    time = pd.to_numeric(pheno[time_col], errors="raise").astype(float).to_numpy()
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
    tvals = betas / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=dof)

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

def split_significance_groups(
    res: pd.DataFrame,
    fdr_treat_col: str = "fdr_treat",
    fdr_int_col: str = "fdr_interaction",
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    req = {fdr_treat_col, fdr_int_col}
    missing = req - set(res.columns)
    if missing:
        raise ValueError(f"res is missing columns: {missing}")

    hits = res[(res[fdr_treat_col] < alpha) | (res[fdr_int_col] < alpha)].copy()
    main_only = res[(res[fdr_treat_col] < alpha) & (res[fdr_int_col] >= alpha)].copy()
    interaction_only = res[(res[fdr_int_col] < alpha) & (res[fdr_treat_col] >= alpha)].copy()
    both = res[(res[fdr_treat_col] < alpha) & (res[fdr_int_col] < alpha)].copy()
    no_response = res[(res[fdr_treat_col] >= alpha) & (res[fdr_int_col] >= alpha)].copy()

    return {
        "hits": hits,
        "main_only": main_only,
        "interaction_only": interaction_only,
        "both": both,
        "no_response": no_response,
    }


# ------------------------------------
# 2) Utilities for trajectory features
# ------------------------------------
def _validate_inputs(expr: pd.DataFrame, pheno: pd.DataFrame) -> None:
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("expr must be a pandas DataFrame (genes x samples).")
    if not isinstance(pheno, pd.DataFrame):
        raise TypeError("pheno must be a pandas DataFrame indexed by sample IDs.")
    if expr.columns.duplicated().any():
        raise ValueError("expr has duplicated sample columns.")
    if pheno.index.duplicated().any():
        raise ValueError("pheno has duplicated sample index.")
    if not set(expr.columns).issubset(set(pheno.index)):
        missing = set(expr.columns) - set(pheno.index)
        raise ValueError(f"pheno is missing {len(missing)} expr samples (examples: {list(sorted(missing))[:5]})")


def _robust_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad  # consistent w/ std for normal
    return (x - med) / (scale + eps)


def compute_gene_trajectory_features(
    expr: pd.DataFrame,
    pheno: pd.DataFrame,
    time_col: str = "time",
    treat_col: str = "treat",
    treat_value: int | float = 1,
    early_q: float = 0.33,
    late_q: float = 0.67,
    center_each_gene: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Computes treated-sample trajectory features per gene.

    Features are computed on treated samples only.
    Windows: early = t <= quantile(early_q), late = t >= quantile(late_q)
    """
    _validate_inputs(expr, pheno)
    if time_col not in pheno.columns or treat_col not in pheno.columns:
        raise ValueError(f"pheno must have columns: {time_col!r}, {treat_col!r}")

    treated_samples = pheno.index[pheno[treat_col] == treat_value]
    treated_samples = [s for s in treated_samples if s in expr.columns]
    if len(treated_samples) < 3:
        raise ValueError("Need at least 3 treated samples to compute slope/shape features.")

    t = pheno.loc[treated_samples, time_col].astype(float)
    if t.isna().any():
        raise ValueError("pheno[time] contains NaNs for treated samples.")

    t_min = float(t.min())
    t_early_thr = float(t.quantile(early_q))
    t_late_thr = float(t.quantile(late_q))

    # Precompute masks on treated samples
    t_vals = t.to_numpy()
    early_mask = t_vals <= t_early_thr
    late_mask = t_vals >= t_late_thr
    base_mask = t_vals == t_min

    # Fit helper
    lr = LinearRegression()

    out = []
    X = t_vals.reshape(-1, 1)

    # Iterate genes
    sub = expr[treated_samples]
    for gene, y_series in sub.iterrows():
        y = y_series.to_numpy(dtype=float)

        # optional per-gene centering to reduce baseline offsets
        # (baseline features still computed relative to earliest timepoint mean)
        if center_each_gene:
            y = y - np.nanmean(y)

        baseline = float(np.nanmean(y[base_mask])) if base_mask.any() else float(np.nanmean(y))
        early_mean = float(np.nanmean(y[early_mask])) if early_mask.any() else float(np.nanmean(y))
        late_mean = float(np.nanmean(y[late_mask])) if late_mask.any() else float(np.nanmean(y))

        # slope y ~ t
        if np.isfinite(y).sum() >= 2 and np.isfinite(t_vals).sum() == len(t_vals):
            lr.fit(X, y)
            slope = float(lr.coef_[0])
            intercept = float(lr.intercept_)
            yhat = lr.predict(X)
            # simple R^2
            ss_res = float(np.nansum((y - yhat) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2)) + eps
            r2 = 1.0 - ss_res / ss_tot
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan

        # peak relative to baseline
        delta = y - baseline
        idx = int(np.nanargmax(np.abs(delta))) if np.isfinite(delta).any() else 0
        peak_time = float(t_vals[idx])
        peak_magnitude = float(delta[idx])

        out.append(
            {
                "gene": gene,
                "baseline": baseline,
                "early_change": early_mean - baseline,
                "late_change": late_mean - baseline,
                "slope": slope,
                "intercept": intercept,
                "r2_time": r2,
                "peak_time": peak_time,
                "peak_magnitude": peak_magnitude,
                "t_min": t_min,
                "t_early_thr": t_early_thr,
                "t_late_thr": t_late_thr,
            }
        )

    feats = pd.DataFrame(out).set_index("gene")
    return feats


# -----------------------------------------
# 3) Response-class rules (deterministic)
# -----------------------------------------
def classify_temporal_response(
    features: pd.DataFrame,
    *,
    change_thr: float | None = None,
    slope_thr: float | None = None,
    transient_ratio: float = 0.5,
    use_robust_thresholds: bool = True,
) -> pd.Series:
    """
    Classifies genes into temporal response shapes based on treated-trajectory features.

    If change_thr/slope_thr are None and use_robust_thresholds=True, thresholds are set
    data-adaptively using robust z-scores of |early_change|, |late_change| and |slope|.
    """
    req = {"early_change", "late_change", "slope"}
    missing = req - set(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {missing}")

    ec = features["early_change"].to_numpy(float)
    lc = features["late_change"].to_numpy(float)
    s = features["slope"].to_numpy(float)

    abs_ec = np.abs(ec)
    abs_lc = np.abs(lc)
    abs_s = np.abs(s)

    # Adaptive thresholds (robust) if not provided
    if use_robust_thresholds and (change_thr is None or slope_thr is None):
        # mark "meaningful change" as robust z >= 1.0 (tune if needed)
        z_change = np.maximum(_robust_z(abs_ec), _robust_z(abs_lc))
        z_slope = _robust_z(abs_s)
        if change_thr is None:
            # Convert robust z back into a magnitude threshold using median+MAD scale
            # Here: choose percentile threshold as a practical proxy
            change_thr = float(np.nanpercentile(np.maximum(abs_ec, abs_lc), 60))
        if slope_thr is None:
            slope_thr = float(np.nanpercentile(abs_s, 60))
    else:
        if change_thr is None:
            change_thr = 0.0
        if slope_thr is None:
            slope_thr = 0.0

    labels = []
    for eci, lci, si in zip(ec, lc, s):
        ae, al, as_ = abs(eci), abs(lci), abs(si)

        # flat / weak temporal
        if ae < change_thr and al < change_thr:
            labels.append("Treatment_flat")
            continue

        # early vs late dominance (magnitude-based)
        if ae > al * (1.0 + transient_ratio):
            labels.append("Early_responder")
            continue
        if al > ae * (1.0 + transient_ratio):
            labels.append("Late_responder")
            continue

        # sustained vs transient (sign consistency + slope)
        if np.sign(eci) == np.sign(lci) and as_ >= slope_thr:
            labels.append("Sustained_responder")
        else:
            labels.append("Transient_responder")

    return pd.Series(labels, index=features.index, name="response_class")


# ---------------------------------------------------
# 4) Glue: combine significance groups + classifications
# ---------------------------------------------------
def build_gene_response_deliverables(
    expr: pd.DataFrame,
    pheno: pd.DataFrame,
    res: pd.DataFrame,
    *,
    time_col: str = "time",
    treat_col: str = "treat",
    treat_value: int | float = 1,
    fdr_treat_col: str = "fdr_treat",
    fdr_int_col: str = "fdr_interaction",
    alpha: float = 0.05,
    # feature params
    early_q: float = 0.33,
    late_q: float = 0.67,
    center_each_gene: bool = True,
    # classification params
    change_thr: float | None = None,
    slope_thr: float | None = None,
    transient_ratio: float = 0.5,
    use_robust_thresholds: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Returns:
      - groups: dict of significance group dfs
      - features_all: trajectory features for all genes in res (treated only)
      - classes_df: per-gene: sig_group + response_class (for temporal genes) + FDRs + features
      - summaries: counts per group/class
    """
    groups = split_significance_groups(res, fdr_treat_col, fdr_int_col, alpha=alpha)

    # compute features for genes in res index only (intersection)
    common_genes = res.index.intersection(expr.index)
    if len(common_genes) == 0:
        raise ValueError("No overlap between res.index and expr.index (genes).")
    feats_all = compute_gene_trajectory_features(
        expr.loc[common_genes],
        pheno,
        time_col=time_col,
        treat_col=treat_col,
        treat_value=treat_value,
        early_q=early_q,
        late_q=late_q,
        center_each_gene=center_each_gene,
    )

    # significance group label per gene
    sig_group = pd.Series("no_response", index=common_genes)
    sig_group.loc[groups["main_only"].index.intersection(common_genes)] = "main_only"
    sig_group.loc[groups["interaction_only"].index.intersection(common_genes)] = "interaction_only"
    sig_group.loc[groups["both"].index.intersection(common_genes)] = "both"
    # hits includes main/inter/both; keep sig_group as above

    # classify temporal shapes only for genes with interaction (interaction_only + both)
    temporal_genes = groups["interaction_only"].index.union(groups["both"].index).intersection(common_genes)
    temporal_feats = feats_all.loc[temporal_genes]
    temporal_labels = classify_temporal_response(
        temporal_feats,
        change_thr=change_thr,
        slope_thr=slope_thr,
        transient_ratio=transient_ratio,
        use_robust_thresholds=use_robust_thresholds,
    )

    # assemble final table
    classes_df = (
        res.loc[common_genes, [fdr_treat_col, fdr_int_col]]
        .join(feats_all, how="left")
        .assign(sig_group=sig_group)
    )
    classes_df["response_class"] = np.where(
        classes_df.index.isin(temporal_genes),
        temporal_labels.reindex(classes_df.index),
        np.nan,
    )

    # For main_only genes, assign a simple label
    # (deliverable: "Treatment_response_flat" vs leave unclassified)
    main_genes = groups["main_only"].index.intersection(common_genes)
    classes_df.loc[main_genes, "response_class"] = "Treatment_response_flat"

    # Summaries
    summary_by_sig = classes_df["sig_group"].value_counts(dropna=False).to_frame("n")
    summary_by_class = classes_df["response_class"].value_counts(dropna=False).to_frame("n")
    summary_sig_x_class = (
        classes_df.pivot_table(index="sig_group", columns="response_class", values=fdr_treat_col, aggfunc="size", fill_value=0)
    )

    return {
        **groups,
        "features_all": feats_all,
        "classes_df": classes_df,
        "summary_by_sig": summary_by_sig,
        "summary_by_class": summary_by_class,
        "summary_sig_x_class": summary_sig_x_class,
    }


# -----------------------------
# 5) Example usage
# -----------------------------
# out = build_gene_response_deliverables(expr, pheno, res)
# classes_df = out["classes_df"]
# out["summary_sig_x_class"]
#
# Deliverables to save:
# classes_df.to_csv("gene_response_classes.csv")
# out["summary_sig_x_class"].to_csv("gene_response_summary_sig_x_class.csv")



