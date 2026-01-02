import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.linear_model import LinearRegression
import GEOparse
import os
from scipy.stats import linregress


def get_geoparse(
    gse_id: str = "GSE6207",
    geo_dir: str = "./geo_data",
    force_download: bool = False,
):
    os.makedirs(geo_dir, exist_ok=True)
    gse_soft_file = os.path.join(geo_dir, f"{gse_id}_family.soft.gz")
    if force_download or not os.path.exists(gse_soft_file):
        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=geo_dir,
            how="full",
        )
    else:
        gse = GEOparse.get_GEO(
            filepath=gse_soft_file,
        )
    expression_df = gse.pivot_samples("VALUE")
    phenotype_df = gse.phenotype_data
    gpl = next(iter(gse.gpls.values()))
    annot = gpl.table
    return expression_df, phenotype_df, annot

def fit_timecourse_ols(
    expr_gene_by_sample: pd.DataFrame,
    pheno: pd.DataFrame,
    time_col: str = "time",
    type_col: str = "treated",
    center_time: bool = True,
) -> pd.DataFrame:
    """
    y = 1 + time + treat + time:treat

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
    treat = pheno[type_col].astype(int).to_numpy()
    time_c = time - np.mean(time) if center_time else time.copy()

    X = np.column_stack([
        np.ones_like(time_c),
        time_c,
        treat,
        time_c * treat
    ])
    X_cols = ["Intercept", "time_c", "treat", "time_c:treat"]

    XtX_inv = np.linalg.inv(X.T @ X)
    H = XtX_inv @ X.T  # (p x n)

    Y = expr.to_numpy(dtype=float)  # (G x N)
    G, N = Y.shape
    P = X.shape[1]

    betas = (H @ Y.T).T  # (G x P)
    resid = Y - (betas @ X.T)  # (G x N)

    dof = N - P
    sigma2 = (resid**2).sum(axis=1) / dof  # (G,)
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

    # Multiple-testing correction
    res["fdr_treat"] = multipletests(res["p_treat"].to_numpy(), method="fdr_bh")[1]
    res["fdr_interaction"] = multipletests(res["p_time_c:treat"].to_numpy(), method="fdr_bh")[1]

    return res


def fit_trajectories(expr_df: pd.DataFrame, time: pd.Series):
    popts = []
    t_norm = time / (time.max() - time.min() + 1e-8)
    X = np.column_stack([np.ones_like(t_norm), t_norm, t_norm ** 2])
    for gene, y in expr_df.iterrows():
        ss_tot = np.sum((y - y.mean()) ** 2)

        # linear fit
        slope, intercept, _, _, _ = linregress(t_norm, y)
        y_pred_linear = intercept + slope * t_norm
        ss_res_linear = np.sum((y - y_pred_linear) ** 2)
        r2_linear = 1 - ss_res_linear / ss_tot if ss_tot > 0 else 0

        # quadratic fit
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs
        y_pred_quad = X @ coeffs
        ss_res_quad = np.sum((y - y_pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
        quad_importance = abs(c) * (t_norm.iloc[-1] ** 2) / (abs(b) * t_norm.iloc[-1] + 1e-8)
        popts.append({
            'gene': gene,
            'q_a': a,
            'q_b': b,
            'q_c': c,
            'lin_slope': slope,
            'lin_intercept': intercept,
            'r2_linear': r2_linear,
            'r2_quad': r2_quad,
            'quad_importance': quad_importance,
            'peak_time': float(time.loc[y.idxmax()])
        })
    return pd.DataFrame(popts).set_index('gene')


