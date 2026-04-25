"""
Shared utilities for the qqkrls package.

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import sys
import numpy as np
from scipy import stats as sp_stats


def safe_print(text: str):
    """Print text safely, handling Windows encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


def bds_test(series, max_dim: int = 6, epsilon: float = None):
    """
    BDS (Broock-Dechert-Scheinkman) test for nonlinearity.

    Tests the null hypothesis that the series is i.i.d. against
    the alternative of nonlinear dependence.

    Parameters
    ----------
    series : array-like
        Time series data.
    max_dim : int
        Maximum embedding dimension (2 to max_dim).
    epsilon : float or None
        Distance threshold.  Default = 0.7 * std(series).

    Returns
    -------
    dict
        Keys: 'dimensions', 'z_stats', 'p_values'.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if epsilon is None:
        epsilon = 0.7 * np.std(x, ddof=1)

    dims = list(range(2, max_dim + 1))
    z_stats = []
    p_values = []

    for m in dims:
        # Correlation integral C(m, epsilon)
        embedded = np.column_stack([x[i:n-m+1+i] for i in range(m)])
        n_e = len(embedded)

        # Count pairs within epsilon
        count = 0
        for i in range(n_e):
            diffs = np.abs(embedded[i+1:] - embedded[i])
            close = np.all(diffs <= epsilon, axis=1)
            count += np.sum(close)

        C_m = 2 * count / (n_e * (n_e - 1)) if n_e > 1 else 0

        # C(1, epsilon)
        count1 = 0
        for i in range(n - 1):
            count1 += np.sum(np.abs(x[i+1:] - x[i]) <= epsilon)
        C_1 = 2 * count1 / (n * (n - 1)) if n > 1 else 0

        # BDS statistic
        bds = C_m - C_1 ** m

        # Standard deviation (simplified)
        K = C_1 ** 2
        sigma_sq = 4 * (K ** m - C_1 ** (2 * m)) / n
        sigma_sq = max(sigma_sq, 1e-20)
        z = bds / np.sqrt(sigma_sq)

        p = 2 * sp_stats.norm.sf(abs(z))
        z_stats.append(z)
        p_values.append(p)

    return {
        "dimensions": dims,
        "z_stats": z_stats,
        "p_values": p_values,
    }


def parameter_stability_test(series, trim: float = 0.15):
    """
    Andrews (1993) / Andrews & Ploberger (1994) parameter stability test.

    Computes Max-F, Exp-F, and Ave-F test statistics.

    Parameters
    ----------
    series : array-like
    trim : float
        Fraction to trim from each end (default 15%).

    Returns
    -------
    dict with 'max_f', 'exp_f', 'ave_f' statistics.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    start = int(n * trim)
    end = int(n * (1 - trim))

    f_stats = []
    for t in range(start, end):
        x1 = x[:t]
        x2 = x[t:]
        if len(x1) < 5 or len(x2) < 5:
            continue

        ssr_full = np.sum((x - np.mean(x)) ** 2)
        ssr_1 = np.sum((x1 - np.mean(x1)) ** 2)
        ssr_2 = np.sum((x2 - np.mean(x2)) ** 2)
        ssr_parts = ssr_1 + ssr_2

        if ssr_parts > 0:
            f = (ssr_full - ssr_parts) / (ssr_parts / (n - 2))
        else:
            f = 0
        f_stats.append(f)

    f_arr = np.array(f_stats)

    return {
        "max_f": float(np.max(f_arr)) if len(f_arr) > 0 else np.nan,
        "exp_f": float(np.log(np.mean(np.exp(0.5 * f_arr)))) if len(f_arr) > 0 else np.nan,
        "ave_f": float(np.mean(f_arr)) if len(f_arr) > 0 else np.nan,
    }


def jarque_bera(series):
    """
    Jarque-Bera normality test.

    Returns
    -------
    dict with 'statistic' and 'p_value'.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    stat, pval = sp_stats.jarque_bera(x)
    return {"statistic": float(stat), "p_value": float(pval)}
