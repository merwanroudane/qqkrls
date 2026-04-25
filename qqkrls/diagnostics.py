"""
Comprehensive diagnostics for KRLS & QQKRLS models.

Provides pre-estimation diagnostics (nonlinearity justification)
and post-estimation diagnostics (residual analysis, goodness-of-fit).

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Optional, Dict, Any
from .utils import safe_print


# ═══════════════════════════════════════════════════════════════════════
#  Pre-estimation Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def linearity_test_battery(y, X, col_names=None):
    """
    Run a battery of linearity tests to justify KRLS/QQKRLS.

    Tests include:
    - Ramsey RESET test (functional form misspecification)
    - Breusch-Pagan test (heteroskedasticity)
    - Jarque-Bera test (normality of residuals)
    - BDS test (nonlinear dependence in residuals)

    Parameters
    ----------
    y : array-like, shape (N,)
    X : array-like, shape (N, D) or (N,)
    col_names : list of str or None

    Returns
    -------
    pd.DataFrame
        Test results with columns: Test, Statistic, P-value, Decision.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape

    if col_names is None:
        col_names = [f"x{j+1}" for j in range(d)]

    results = []

    # 1. OLS residuals
    X_ols = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_ols, y, rcond=None)[0]
        resid = y - X_ols @ beta
        yhat = X_ols @ beta
    except Exception:
        resid = y - np.mean(y)
        yhat = np.full(n, np.mean(y))

    ssr = np.sum(resid**2)

    # 2. Ramsey RESET test (powers of fitted values)
    try:
        X_reset = np.column_stack([X_ols, yhat**2, yhat**3])
        beta_r = np.linalg.lstsq(X_reset, y, rcond=None)[0]
        resid_r = y - X_reset @ beta_r
        ssr_r = np.sum(resid_r**2)
        df1 = 2  # two added regressors
        df2 = n - X_reset.shape[1]
        f_stat = ((ssr - ssr_r) / df1) / (ssr_r / max(df2, 1))
        p_reset = 1 - sp_stats.f.cdf(f_stat, df1, max(df2, 1))
        results.append({
            "Test": "Ramsey RESET",
            "Statistic": round(f_stat, 4),
            "P-value": round(p_reset, 4),
            "Decision": "Nonlinear***" if p_reset < 0.01 else
                        "Nonlinear**" if p_reset < 0.05 else
                        "Nonlinear*" if p_reset < 0.10 else "Linear",
        })
    except Exception:
        results.append({"Test": "Ramsey RESET", "Statistic": np.nan,
                        "P-value": np.nan, "Decision": "N/A"})

    # 3. Breusch-Pagan test
    try:
        resid_sq = resid**2
        X_bp = np.column_stack([np.ones(n), X])
        beta_bp = np.linalg.lstsq(X_bp, resid_sq, rcond=None)[0]
        resid_sq_hat = X_bp @ beta_bp
        ssr_bp = np.sum((resid_sq - resid_sq_hat)**2)
        ssr_0 = np.sum((resid_sq - np.mean(resid_sq))**2)
        bp_stat = n * (1 - ssr_bp / max(ssr_0, 1e-20))
        bp_stat = max(bp_stat, 0)
        p_bp = 1 - sp_stats.chi2.cdf(bp_stat, d)
        results.append({
            "Test": "Breusch-Pagan",
            "Statistic": round(bp_stat, 4),
            "P-value": round(p_bp, 4),
            "Decision": "Heteroskedastic***" if p_bp < 0.01 else
                        "Heteroskedastic**" if p_bp < 0.05 else
                        "Heteroskedastic*" if p_bp < 0.10 else "Homoskedastic",
        })
    except Exception:
        results.append({"Test": "Breusch-Pagan", "Statistic": np.nan,
                        "P-value": np.nan, "Decision": "N/A"})

    # 4. Jarque-Bera test
    try:
        jb_stat, jb_p = sp_stats.jarque_bera(resid)
        results.append({
            "Test": "Jarque-Bera",
            "Statistic": round(float(jb_stat), 4),
            "P-value": round(float(jb_p), 4),
            "Decision": "Non-normal***" if jb_p < 0.01 else
                        "Non-normal**" if jb_p < 0.05 else
                        "Non-normal*" if jb_p < 0.10 else "Normal",
        })
    except Exception:
        results.append({"Test": "Jarque-Bera", "Statistic": np.nan,
                        "P-value": np.nan, "Decision": "N/A"})

    # 5. BDS test (on residuals, dimension=2)
    try:
        from .utils import bds_test
        bds = bds_test(resid, max_dim=3)
        results.append({
            "Test": "BDS (dim=2)",
            "Statistic": round(bds["z_stats"][0], 4),
            "P-value": round(bds["p_values"][0], 4),
            "Decision": "Nonlinear dep.***" if bds["p_values"][0] < 0.01 else
                        "Nonlinear dep.**" if bds["p_values"][0] < 0.05 else
                        "Nonlinear dep.*" if bds["p_values"][0] < 0.10 else "i.i.d.",
        })
    except Exception:
        results.append({"Test": "BDS (dim=2)", "Statistic": np.nan,
                        "P-value": np.nan, "Decision": "N/A"})

    df = pd.DataFrame(results)
    return df


def print_diagnostics(df):
    """Pretty-print diagnostics table."""
    safe_print("")
    safe_print("=" * 70)
    safe_print("   Pre-Estimation Diagnostics -- Justification for KRLS/QQKRLS")
    safe_print("=" * 70)
    safe_print("")
    safe_print(f"  {'Test':<20s}  {'Statistic':>12s}  {'P-value':>10s}  {'Decision'}")
    safe_print("  " + "-" * 65)
    for _, row in df.iterrows():
        safe_print(f"  {row['Test']:<20s}  {row['Statistic']:>12.4f}  "
                   f"{row['P-value']:>10.4f}  {row['Decision']}")
    safe_print("")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Post-estimation KRLS Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def krls_residual_diagnostics(fit) -> Dict[str, Any]:
    """
    Comprehensive residual diagnostics for a fitted KRLS model.

    Parameters
    ----------
    fit : KRLSResult

    Returns
    -------
    dict with keys:
        residuals, std_residuals, R2, adj_R2, AIC, BIC,
        durbin_watson, jb_stat, jb_pval, mean_abs_error, rmse
    """
    y = fit.y.ravel()
    yhat = fit.fitted.ravel()
    n = len(y)
    resid = y - yhat

    # Effective degrees of freedom (trace of hat matrix approx)
    K = fit.K
    lam = fit.lambda_
    try:
        H = K @ np.linalg.inv(K + lam * np.eye(n))
        df_model = np.trace(H)
    except Exception:
        df_model = fit.X.shape[1]

    ssr = np.sum(resid**2)
    sst = np.sum((y - np.mean(y))**2)
    R2 = 1 - ssr / max(sst, 1e-20)
    adj_R2 = 1 - (1 - R2) * (n - 1) / max(n - df_model - 1, 1)

    sigma2 = ssr / max(n - df_model, 1)
    std_resid = resid / max(np.sqrt(sigma2), 1e-20)

    # AIC, BIC
    log_lik = -0.5 * n * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
    aic = -2 * log_lik + 2 * df_model
    bic = -2 * log_lik + np.log(n) * df_model

    # Durbin-Watson
    dw = np.sum(np.diff(resid)**2) / max(ssr, 1e-20)

    # JB test
    jb_stat, jb_pval = sp_stats.jarque_bera(resid)

    # Error metrics
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid**2))

    result = {
        "residuals": resid,
        "std_residuals": std_resid,
        "R2": R2,
        "adj_R2": adj_R2,
        "eff_df": df_model,
        "AIC": float(aic),
        "BIC": float(bic),
        "durbin_watson": float(dw),
        "jb_stat": float(jb_stat),
        "jb_pval": float(jb_pval),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }

    return result


def print_krls_diagnostics(diag: dict):
    """Pretty-print KRLS residual diagnostics."""
    safe_print("")
    safe_print("=" * 55)
    safe_print("   KRLS Post-Estimation Diagnostics")
    safe_print("=" * 55)
    safe_print("")
    safe_print(f"  R-squared        : {diag['R2']:.4f}")
    safe_print(f"  Adjusted R2      : {diag['adj_R2']:.4f}")
    safe_print(f"  Eff. df (model)  : {diag['eff_df']:.2f}")
    safe_print(f"  AIC              : {diag['AIC']:.2f}")
    safe_print(f"  BIC              : {diag['BIC']:.2f}")
    safe_print(f"  Durbin-Watson    : {diag['durbin_watson']:.4f}")
    safe_print(f"  MAE              : {diag['MAE']:.4f}")
    safe_print(f"  RMSE             : {diag['RMSE']:.4f}")
    safe_print(f"  Jarque-Bera      : {diag['jb_stat']:.4f}  (p={diag['jb_pval']:.4f})")
    safe_print("")


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Variable QQKRLS  (paper-style panel estimation)
# ═══════════════════════════════════════════════════════════════════════

def multi_qqkrls(y, X, col_names=None,
                 y_quantiles=None, x_quantiles=None,
                 sigma=None, lambda_=None,
                 min_obs=15, n_boot=200, verbose=True):
    """
    Run QQKRLS for each column of X against y.

    This mirrors the multi-panel results in Adebayo et al. (2024)
    where QQKRLS is run for each independent variable separately.

    Parameters
    ----------
    y : array-like, shape (N,)
        Dependent variable.
    X : array-like, shape (N, D)
        Matrix of independent variables.
    col_names : list of str or None
    y_quantiles, x_quantiles : array-like or None
    sigma, lambda_ : float or None
    min_obs : int
    n_boot : int
    verbose : bool

    Returns
    -------
    dict
        Mapping {var_name: QQKRLSResult}.
    """
    from .qqkrls import qqkrls as _qqkrls

    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape

    if col_names is None:
        col_names = [f"x{j+1}" for j in range(d)]

    results = {}
    for j in range(d):
        name = col_names[j]
        if verbose:
            safe_print(f"\n{'='*62}")
            safe_print(f"  QQKRLS: {name} -> Y  (variable {j+1}/{d})")
            safe_print(f"{'='*62}")

        res = _qqkrls(
            y, X[:, j],
            y_quantiles=y_quantiles,
            x_quantiles=x_quantiles,
            sigma=sigma,
            lambda_=lambda_,
            min_obs=min_obs,
            n_boot=n_boot,
            verbose=verbose,
        )
        results[name] = res

    return results
