import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.stats import linregress


def fit_ols_df(X_df, Y_df, fdr=None) -> pd.DataFrame:
    """
    fits linear regression to all columns in X_df
    Returns a results DataFrame indexed by gene with betas, pvals, and FDRs.
    """
    X = np.concatenate([
        np.ones((len(X_df), 1)),
        X_df.to_numpy()
    ], axis=1)

    X_cols = ["Intercept"] + X_df.columns.tolist()
    XtX_inv = np.linalg.inv(X.T @ X)
    H = XtX_inv @ X.T  # (p x n)

    Y = Y_df.to_numpy(dtype=float)  # (G x N)
    G, N = Y.shape
    P = X.shape[1]

    betas = (H @ Y.T).T  # (G x P)
    resid = Y - (betas @ X.T)  # (G x N)

    dof = N - P
    sigma2 = (resid**2).sum(axis=1) / dof  # (G,)
    se = np.sqrt(sigma2[:, None] * np.diag(XtX_inv)[None, :])  # (G x P)
    tvals = betas / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=dof)

    bases = ["beta", "se", "t", "p"]
    data = np.hstack([betas, se, tvals, pvals])
    cols = pd.MultiIndex.from_product(
        [bases, X_cols],
        names=["stat", "variable"]
    )
    res = pd.DataFrame(
        data,
        index=Y_df.index,
        columns=cols,
    )

    if fdr is not None:
        for f in fdr:
            res[("fdr", f)] = multipletests(
                res[("p", f)].to_numpy(),
                method="fdr_bh",
            )[1]
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



