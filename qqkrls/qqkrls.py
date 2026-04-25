"""
Quantile-on-Quantile Kernel-Based Regularized Least Squares (QQKRLS).

Combines KRLS (Hainmueller & Hazlett, 2014) with Quantile-on-Quantile
regression (Sim & Zhou, 2015) to provide nonparametric marginal effects
across all quantile pairs of the predictor and predicted variables.

References
----------
Adebayo, T.S., Ozkan, O. & Eweade, B.S. (2024).
    Do energy efficiency R&D investments and ICT promote
    environmental sustainability in Sweden? A QQKRLS investigation.
    *Journal of Cleaner Production*, 440, 140832.

Adebayo, T.S., Meo, M.S., Eweade, B.S. & Ozkan, O. (2024).
    Analyzing the effects of solar energy innovations, digitalization,
    and economic globalization on environmental quality in the United
    States. *Clean Technologies and Environmental Policy*, 26, 4157-4176.

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

from .krls import krls as _krls_fit
from .utils import safe_print


# ═══════════════════════════════════════════════════════════════════════
#  Result Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QQKRLSResult:
    """Container for QQKRLS estimation results.

    Attributes
    ----------
    results : pd.DataFrame
        Long-format table with columns: y_quantile, x_quantile,
        coefficient, std_error, t_value, p_value, significance.
    y_quantiles : np.ndarray
    x_quantiles : np.ndarray
    n_obs : int
    method : str
    """
    results: pd.DataFrame
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    method: str = "Quantile-on-Quantile KRLS"

    def to_matrix(self, value: str = "coefficient") -> np.ndarray:
        """Pivot results into (len(y_quantiles) × len(x_quantiles)) matrix."""
        df = self.results.dropna(subset=[value])
        mat = df.pivot(index="y_quantile", columns="x_quantile", values=value)
        return mat.values

    def significance_matrix(self, alpha: float = 0.05) -> np.ndarray:
        """Binary matrix: 1 where p_value < alpha."""
        df = self.results.dropna(subset=["p_value"])
        mat = df.pivot(index="y_quantile", columns="x_quantile", values="p_value")
        return (mat.values < alpha).astype(int)

    def stars_matrix(self) -> np.ndarray:
        """Matrix of significance stars (str): '***', '**', '*', ''."""
        df = self.results.copy()
        def _stars(p):
            if pd.isna(p):
                return ""
            if p < 0.01:
                return "***"
            elif p < 0.05:
                return "**"
            elif p < 0.10:
                return "*"
            return ""
        df["stars"] = df["p_value"].apply(_stars)
        mat = df.pivot(index="y_quantile", columns="x_quantile", values="stars")
        return mat.values

    def to_dataframe(self) -> pd.DataFrame:
        """Return full results DataFrame."""
        return self.results.copy()

    def summary(self):
        """Print publication-quality summary."""
        r = self.results.dropna(subset=["coefficient"])
        n_sig01 = (r["p_value"] < 0.01).sum()
        n_sig05 = (r["p_value"] < 0.05).sum()
        n_sig10 = (r["p_value"] < 0.10).sum()
        n_total = len(r)

        # Count positive/negative
        n_pos = (r["coefficient"] > 0).sum()
        n_neg = (r["coefficient"] < 0).sum()

        lines = [
            "",
            "=" * 62,
            "   Quantile-on-Quantile KRLS (QQKRLS) -- Summary",
            "=" * 62,
            "",
            f"  Observations     : {self.n_obs}",
            f"  Y quantiles      : {len(self.y_quantiles)} levels  [{self.y_quantiles[0]:.2f} ... {self.y_quantiles[-1]:.2f}]",
            f"  X quantiles      : {len(self.x_quantiles)} levels  [{self.x_quantiles[0]:.2f} ... {self.x_quantiles[-1]:.2f}]",
            f"  Total cells      : {n_total}",
            "",
            "  -- Coefficient Statistics --",
            f"    Mean   : {r['coefficient'].mean():.4f}",
            f"    Median : {r['coefficient'].median():.4f}",
            f"    Min    : {r['coefficient'].min():.4f}",
            f"    Max    : {r['coefficient'].max():.4f}",
            f"    Std    : {r['coefficient'].std():.4f}",
            "",
            f"    Positive : {n_pos} / {n_total}  ({100*n_pos/max(n_total,1):.1f}%)",
            f"    Negative : {n_neg} / {n_total}  ({100*n_neg/max(n_total,1):.1f}%)",
            "",
            "  -- Significance --",
            f"    p < 0.01  (***) : {n_sig01} / {n_total}  ({100*n_sig01/max(n_total,1):.1f}%)",
            f"    p < 0.05  (**)  : {n_sig05} / {n_total}  ({100*n_sig05/max(n_total,1):.1f}%)",
            f"    p < 0.10  (*)   : {n_sig10} / {n_total}  ({100*n_sig10/max(n_total,1):.1f}%)",
            "",
        ]
        safe_print("\n".join(lines))
        return self

    def export_csv(self, path: str, digits: int = 4):
        """Export results to CSV."""
        df = self.results.copy()
        for c in ["coefficient", "std_error", "t_value", "p_value"]:
            if c in df.columns:
                df[c] = df[c].round(digits)
        df.to_csv(path, index=False)

    def export_latex(self, value: str = "coefficient",
                     caption: str = "QQKRLS Coefficients",
                     show_stars: bool = True) -> str:
        """Return LaTeX table string of the coefficient matrix."""
        mat = self.to_matrix(value)
        stars = self.stars_matrix() if show_stars else None
        y_q = sorted(self.results["y_quantile"].dropna().unique())
        x_q = sorted(self.results["x_quantile"].dropna().unique())

        header = " & ".join([f"{q:.2f}" for q in x_q])
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\footnotesize",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{l" + "r" * len(x_q) + "}",
            r"\hline",
            r"$\theta \backslash \tau$ & " + header + r" \\",
            r"\hline",
        ]
        for i, yq in enumerate(y_q):
            if i < mat.shape[0]:
                cells = []
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    s = stars[i, j] if stars is not None else ""
                    if np.isnan(v):
                        cells.append("---")
                    else:
                        cells.append(f"{v:.3f}{s}")
                vals = " & ".join(cells)
                lines.append(f"{yq:.2f} & {vals}" + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Main QQKRLS Estimator
# ═══════════════════════════════════════════════════════════════════════

def qqkrls(y, x,
           y_quantiles=None,
           x_quantiles=None,
           sigma=None,
           lambda_=None,
           min_obs: int = 15,
           n_boot: int = 200,
           verbose: bool = True) -> QQKRLSResult:
    """
    Quantile-on-Quantile Kernel-Based Regularized Least Squares.

    For each x-quantile theta, subsets data where x <= Q_x(theta) and
    fits KRLS of y on x.  The tau-th quantile of the resulting pointwise
    marginal effects gives the coefficient beta(theta, tau).

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Independent variable.
    y_quantiles : array-like or None
        Quantiles of y (tau).  Default ``np.arange(0.05, 1.0, 0.05)``.
    x_quantiles : array-like or None
        Quantiles of x (theta).  Default same as y_quantiles.
    sigma : float or None
        Gaussian kernel bandwidth.  None = auto (dim of X = 1).
    lambda_ : float or None
        Regularisation parameter.  None = LOO cross-validation per subset.
    min_obs : int
        Minimum observations in a subset to fit KRLS.
    n_boot : int
        Bootstrap replications for standard errors.
    verbose : bool
        Print progress.

    Returns
    -------
    QQKRLSResult

    References
    ----------
    Adebayo, T.S., Ozkan, O. & Eweade, B.S. (2024).
    *Journal of Cleaner Production*, 440, 140832.

    Examples
    --------
    >>> import numpy as np
    >>> from qqkrls import qqkrls
    >>> x = np.random.randn(200)
    >>> y = 0.5 * np.sin(x) + np.random.randn(200) * 0.3
    >>> result = qqkrls(y, x, verbose=True)
    >>> result.summary()
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()

    if len(y) != len(x):
        raise ValueError("y and x must have equal length.")

    # Drop missing
    mask = np.isfinite(y) & np.isfinite(x)
    y_data, x_data = y[mask], x[mask]
    n_obs = len(y_data)

    if n_obs < 20:
        raise ValueError(f"Need >= 20 observations, got {n_obs}.")

    if y_quantiles is None:
        y_quantiles = np.arange(0.05, 1.0, 0.05)
    if x_quantiles is None:
        x_quantiles = np.arange(0.05, 1.0, 0.05)
    y_quantiles = np.asarray(y_quantiles)
    x_quantiles = np.asarray(x_quantiles)

    if verbose:
        safe_print("=" * 62)
        safe_print("   Running Quantile-on-Quantile KRLS (QQKRLS)")
        safe_print("=" * 62)
        safe_print(f"  n = {n_obs},  Y quantiles = {len(y_quantiles)},  X quantiles = {len(x_quantiles)}")

    records = []
    total = len(x_quantiles)
    done = 0

    for theta in x_quantiles:
        done += 1
        x_thresh = np.quantile(x_data, theta)
        sel = x_data <= x_thresh
        x_sub = x_data[sel]
        y_sub = y_data[sel]

        derivs = None
        krls_fit = None

        if len(x_sub) >= min_obs:
            try:
                krls_fit = _krls_fit(
                    x_sub.reshape(-1, 1), y_sub,
                    sigma=sigma if sigma is not None else 1.0,
                    lambda_=lambda_,
                    derivative=True, vcov=True, binary=False,
                    verbose=0,
                )
                derivs = krls_fit.derivatives[:, 0]  # pointwise marginal effects
            except Exception as e:
                if verbose:
                    print(f"  Warning: KRLS failed at theta={theta:.2f}: {e}")
                derivs = None

        for tau in y_quantiles:
            rec = {
                "y_quantile": round(float(tau), 4),
                "x_quantile": round(float(theta), 4),
                "coefficient": np.nan,
                "std_error": np.nan,
                "t_value": np.nan,
                "p_value": np.nan,
                "significance": "",
            }

            if derivs is not None and len(derivs) >= 5:
                try:
                    # Coefficient = tau-th quantile of the pointwise marginal effects
                    beta = float(np.quantile(derivs, tau))

                    # Bootstrap SE
                    boot_betas = np.empty(n_boot)
                    rng = np.random.default_rng(seed=int(theta * 1000 + tau * 100))
                    n_sub = len(x_sub)
                    for b in range(n_boot):
                        idx = rng.choice(n_sub, size=n_sub, replace=True)
                        try:
                            bfit = _krls_fit(
                                x_sub[idx].reshape(-1, 1), y_sub[idx],
                                sigma=sigma if sigma is not None else 1.0,
                                lambda_=krls_fit.lambda_,
                                derivative=True, vcov=False, binary=False,
                                verbose=0,
                            )
                            boot_betas[b] = np.quantile(bfit.derivatives[:, 0], tau)
                        except Exception:
                            boot_betas[b] = np.nan

                    valid = boot_betas[np.isfinite(boot_betas)]
                    if len(valid) > 10:
                        se = float(np.std(valid, ddof=1))
                    else:
                        # Fallback: asymptotic SE from KRLS variance
                        if krls_fit.var_avg_derivatives is not None:
                            se = float(np.sqrt(krls_fit.var_avg_derivatives[0, 0]))
                        else:
                            se = np.nan

                    if se > 0 and np.isfinite(se):
                        tval = beta / se
                        from scipy import stats as _st
                        pval = 2 * _st.norm.sf(abs(tval))
                    else:
                        tval = np.nan
                        pval = np.nan

                    stars = ""
                    if np.isfinite(pval):
                        if pval < 0.01:
                            stars = "***"
                        elif pval < 0.05:
                            stars = "**"
                        elif pval < 0.10:
                            stars = "*"

                    rec.update(
                        coefficient=beta,
                        std_error=se,
                        t_value=tval,
                        p_value=pval,
                        significance=stars,
                    )
                except Exception:
                    pass

            records.append(rec)

        if verbose:
            pct = int(100 * done / total)
            print(f"  theta = {theta:.2f}  ({pct}%)")

    if verbose:
        print("  Done!")

    df = pd.DataFrame(records)
    result = QQKRLSResult(
        results=df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_obs,
    )

    if verbose:
        result.summary()

    return result
