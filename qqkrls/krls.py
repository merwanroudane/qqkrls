"""
Kernel-Based Regularized Least Squares (KRLS).

Faithful Python port of the R CRAN package ``KRLS`` (v1.0-0)
by Jens Hainmueller (Stanford) and Chad Hazlett (UCLA).

Reference
---------
Hainmueller, J. & Hazlett, C. (2014).
    Kernel Regularized Least Squares.
    *Political Analysis*, 22(2), 143-168.
    doi:10.1093/pan/mpt024

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union
from scipy.spatial.distance import cdist

from .utils import safe_print


# ═══════════════════════════════════════════════════════════════════════
#  Data Containers
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KRLSResult:
    """Container for KRLS estimation results.

    Attributes
    ----------
    coeffs : np.ndarray           (N, 1) – solution vector c*
    fitted : np.ndarray           (N, 1) – fitted values
    X : np.ndarray                (N, D) – original (un-scaled) X
    y : np.ndarray                (N, 1) – original y
    K : np.ndarray                (N, N) – kernel matrix
    sigma : float                 bandwidth parameter
    lambda_ : float               regularisation parameter
    R2 : float                    R-squared
    Looe : float                  leave-one-out error (scaled)
    derivatives : np.ndarray      (N, D) – pointwise marginal effects
    avg_derivatives : np.ndarray  (1, D) – average marginal effects
    var_avg_derivatives : np.ndarray  (1, D) – variance of avg derivatives
    vcov_c : np.ndarray           (N, N) – Var-Cov of c
    vcov_fitted : np.ndarray      (N, N) – Var-Cov of fitted values
    binary_indicator : np.ndarray (1, D) – binary predictor flags
    first_diff : dict | None      first-difference results for binary vars
    col_names : list[str]         predictor names
    """
    coeffs: np.ndarray
    fitted: np.ndarray
    X: np.ndarray
    y: np.ndarray
    K: np.ndarray
    sigma: float
    lambda_: float
    R2: float
    Looe: float
    derivatives: Optional[np.ndarray] = None
    avg_derivatives: Optional[np.ndarray] = None
    var_avg_derivatives: Optional[np.ndarray] = None
    vcov_c: Optional[np.ndarray] = None
    vcov_fitted: Optional[np.ndarray] = None
    binary_indicator: Optional[np.ndarray] = None
    first_diff: Optional[dict] = None
    col_names: List[str] = field(default_factory=list)

    # ── Convenience ─────────────────────────────────────────────────
    def summary(self):
        """Print a publication-quality summary of KRLS results."""
        n, d = self.X.shape
        lines = [
            "",
            "=" * 62,
            "   Kernel Regularized Least Squares (KRLS) -- Results",
            "=" * 62,
            "",
            f"  Observations (N) : {n}",
            f"  Predictors   (D) : {d}",
            f"  Kernel           : Gaussian",
            f"  Sigma            : {self.sigma:.4f}",
            f"  Lambda           : {self.lambda_:.6f}",
            f"  R-squared        : {self.R2:.4f}",
            f"  LOO Error        : {self.Looe:.4f}",
            "",
        ]

        if self.avg_derivatives is not None:
            lines.append("  -- Average Marginal Effects --")
            lines.append(f"  {'Variable':>15s}  {'Avg ME':>10s}  {'Std Err':>10s}  {'t-value':>10s}  {'p-value':>10s}")
            lines.append("  " + "-" * 60)
            for j in range(d):
                name = self.col_names[j] if j < len(self.col_names) else f"x{j+1}"
                avg_d = self.avg_derivatives[0, j]
                se = np.sqrt(self.var_avg_derivatives[0, j]) if self.var_avg_derivatives is not None else np.nan
                tval = avg_d / se if se > 0 else np.nan
                # two-sided p-value from normal
                from scipy import stats as _st
                pval = 2 * _st.norm.sf(abs(tval))
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                lines.append(f"  {name:>15s}  {avg_d:>10.4f}  {se:>10.4f}  {tval:>10.4f}  {pval:>10.4f} {stars}")

            lines.append("")
            lines.append("  -- Quartiles of Marginal Effects --")
            lines.append(f"  {'Variable':>15s}  {'Q25':>10s}  {'Q50':>10s}  {'Q75':>10s}")
            lines.append("  " + "-" * 50)
            for j in range(d):
                name = self.col_names[j] if j < len(self.col_names) else f"x{j+1}"
                q25 = np.quantile(self.derivatives[:, j], 0.25)
                q50 = np.quantile(self.derivatives[:, j], 0.50)
                q75 = np.quantile(self.derivatives[:, j], 0.75)
                lines.append(f"  {name:>15s}  {q25:>10.4f}  {q50:>10.4f}  {q75:>10.4f}")
        lines.append("")
        safe_print("\n".join(lines))
        return self

    def predict(self, newdata: np.ndarray) -> np.ndarray:
        """Predict for new observations using the fitted KRLS model."""
        return predict_krls(self, newdata)


# ═══════════════════════════════════════════════════════════════════════
#  Low-Level Functions  (ported from R)
# ═══════════════════════════════════════════════════════════════════════

def gaussian_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """Compute Gaussian kernel matrix.

    K[i,j] = exp( -||x_i - x_j||^2 / sigma )

    Matches R: ``gausskernel <- function(X, sigma) exp(-1*as.matrix(dist(X)^2)/sigma)``
    """
    sq_dists = cdist(X, X, metric="sqeuclidean")
    return np.exp(-sq_dists / sigma)


def _multdiag(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Multiply matrix X by diagonal matrix diag(d):  X @ diag(d).

    Equivalent to R's ``multdiag(X, d)`` but vectorised.
    """
    return X * d[np.newaxis, :]


def _solve_for_c(y: np.ndarray, eigenvalues: np.ndarray,
                 eigenvectors: np.ndarray, lambda_: float,
                 eigtrunc: Optional[float] = None):
    """Solve for KRLS coefficients via eigendecomposition.

    c* = V (Λ + λI)^{-1} V' y
    LOO error = Σ (c_i / G^{-1}_{ii})^2

    Returns (coeffs, Looe).
    """
    n = len(y)

    if eigtrunc is not None:
        last = np.max(np.where(eigenvalues >= eigtrunc * eigenvalues[0])[0]) + 1
        last = max(1, last)
        evals = eigenvalues[:last]
        evecs = eigenvectors[:, :last]
    else:
        evals = eigenvalues
        evecs = eigenvectors

    inv_diag = 1.0 / (evals + lambda_)
    Ginv = _multdiag(evecs, inv_diag) @ evecs.T

    coeffs = Ginv @ y
    Le = float(np.sum((coeffs / np.diag(Ginv)) ** 2))
    return coeffs, Le, Ginv


def _loo_loss(y, eigenvalues, eigenvectors, lambda_, eigtrunc=None):
    """Leave-one-out loss for a given lambda."""
    _, Le, _ = _solve_for_c(y, eigenvalues, eigenvectors, lambda_, eigtrunc)
    return Le


def _lambda_search(y, eigenvalues, eigenvectors, L=None, U=None,
                   tol=None, eigtrunc=None, verbose=False):
    """Golden-section search for optimal lambda minimising LOO loss.

    Faithfully mirrors R's ``lambdasearch()``.
    """
    n = len(y)
    if tol is None:
        tol = 1e-3 * n

    # Upper bound
    if U is None:
        U = float(n)
        while np.sum(eigenvalues / (eigenvalues + U)) < 1:
            U -= 1
            if U <= 0:
                U = float(n)
                break

    # Lower bound
    if L is None:
        q_idx = np.argmin(np.abs(eigenvalues - eigenvalues[0] / 1000))
        q = q_idx + 1
        L = np.finfo(float).eps
        while np.sum(eigenvalues / (eigenvalues + L)) > q:
            L += 0.05
            if L > U:
                L = np.finfo(float).eps
                break

    # Golden section
    phi = 0.381966
    X1 = L + phi * (U - L)
    X2 = U - phi * (U - L)
    S1 = _loo_loss(y, eigenvalues, eigenvectors, X1, eigtrunc)
    S2 = _loo_loss(y, eigenvalues, eigenvectors, X2, eigtrunc)

    if verbose:
        print(f"  λ search: L={L:.4f}  X1={X1:.4f}  X2={X2:.4f}  U={U:.4f}  S1={S1:.2f}  S2={S2:.2f}")

    max_iter = 500
    it = 0
    while abs(S1 - S2) > tol and it < max_iter:
        if S1 < S2:
            U = X2
            X2 = X1
            X1 = L + phi * (U - L)
            S2 = S1
            S1 = _loo_loss(y, eigenvalues, eigenvectors, X1, eigtrunc)
        else:
            L = X1
            X1 = X2
            X2 = U - phi * (U - L)
            S1 = S2
            S2 = _loo_loss(y, eigenvalues, eigenvectors, X2, eigtrunc)
        it += 1

    return X1 if S1 < S2 else X2


# ═══════════════════════════════════════════════════════════════════════
#  Main KRLS Estimator
# ═══════════════════════════════════════════════════════════════════════

def krls(X, y,
         kernel: str = "gaussian",
         lambda_: Optional[float] = None,
         sigma: Optional[float] = None,
         derivative: bool = True,
         binary: bool = True,
         vcov: bool = True,
         eigtrunc: Optional[float] = None,
         col_names: Optional[List[str]] = None,
         verbose: int = 1) -> KRLSResult:
    """
    Kernel Regularized Least Squares.

    Fits a flexible regression function y = f(X) using Gaussian kernels
    and Tikhonov regularisation, without linearity or additivity
    assumptions.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Predictor matrix.
    y : array-like, shape (N,) or (N, 1)
        Outcome vector.
    kernel : str
        Kernel type: 'gaussian', 'linear', 'poly2', 'poly3', 'poly4'.
    lambda_ : float or None
        Regularisation parameter.  If None, chosen by LOO cross-validation.
    sigma : float or None
        Bandwidth for Gaussian kernel.  Default = D (number of predictors).
    derivative : bool
        Compute pointwise marginal effects (requires kernel='gaussian').
    binary : bool
        Compute first differences for binary predictors.
    vcov : bool
        Compute variance-covariance matrices.
    eigtrunc : float or None
        Eigenvalue truncation threshold (0, 1].  None = no truncation.
    col_names : list of str or None
        Predictor names.
    verbose : int
        0 = silent, 1 = summary, 2+ = detailed.

    Returns
    -------
    KRLSResult
        Fitted KRLS model with coefficients, marginal effects, etc.

    References
    ----------
    Hainmueller, J. & Hazlett, C. (2014). Kernel Regularized Least Squares.
    *Political Analysis*, 22(2), 143-168.

    Examples
    --------
    >>> import numpy as np
    >>> from qqkrls import krls
    >>> X = np.random.randn(200, 3)
    >>> y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.randn(200) * 0.3
    >>> fit = krls(X, y, verbose=0)
    >>> fit.summary()
    """
    # ── Input validation ────────────────────────────────────────────
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, d = X.shape
    if n != y.shape[0]:
        raise ValueError(f"nrow(X)={n} != length(y)={y.shape[0]}")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")
    if np.std(y) == 0:
        raise ValueError("y is constant")
    if np.any(np.std(X, axis=0) == 0):
        raise ValueError("At least one column in X is constant — please remove it")
    if derivative and not vcov:
        raise ValueError("derivative=True requires vcov=True")
    if derivative and kernel != "gaussian":
        raise ValueError("Derivatives are only available with kernel='gaussian'")

    if sigma is None:
        sigma = float(d)
    if col_names is None:
        col_names = [f"x{j+1}" for j in range(d)]

    # ── Standardise ─────────────────────────────────────────────────
    X_init = X.copy()
    X_sd = np.std(X_init, axis=0, ddof=0)
    y_init = y.copy()
    y_mean = float(np.mean(y_init))
    y_sd = float(np.std(y_init, ddof=0))

    X_sc = (X - np.mean(X, axis=0)) / X_sd
    y_sc = (y - y_mean) / y_sd

    # ── Kernel matrix ───────────────────────────────────────────────
    if kernel == "gaussian":
        K = gaussian_kernel(X_sc, sigma)
    elif kernel == "linear":
        K = X_sc @ X_sc.T
    elif kernel == "poly2":
        K = (X_sc @ X_sc.T + 1) ** 2
    elif kernel == "poly3":
        K = (X_sc @ X_sc.T + 1) ** 3
    elif kernel == "poly4":
        K = (X_sc @ X_sc.T + 1) ** 4
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # ── Eigendecomposition ──────────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Clamp tiny negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0)

    # ── Lambda selection ────────────────────────────────────────────
    if lambda_ is None:
        lambda_ = _lambda_search(
            y_sc, eigenvalues, eigenvectors, eigtrunc=eigtrunc,
            verbose=(verbose >= 2),
        )
        if verbose >= 2:
            print(f"  Optimal λ = {lambda_:.6f}")

    # ── Solve for c* ────────────────────────────────────────────────
    coeffs, Le, Ginv = _solve_for_c(y_sc, eigenvalues, eigenvectors, lambda_, eigtrunc)
    yfitted_sc = K @ coeffs

    # ── Variance-Covariance ─────────────────────────────────────────
    vcov_c_mat = None
    vcov_fitted_mat = None
    if vcov:
        sigma_sq = float((1.0 / n) * (y_sc - yfitted_sc).T @ (y_sc - yfitted_sc))

        if eigtrunc is not None:
            last = max(1, np.max(np.where(eigenvalues >= eigtrunc * eigenvalues[0])[0]) + 1)
            ev = eigenvalues[:last]
            evec = eigenvectors[:, :last]
        else:
            ev = eigenvalues
            evec = eigenvectors

        vcov_c_sc = _multdiag(evec, sigma_sq * (ev + lambda_) ** (-2)) @ evec.T
        vcov_fitted_sc = K.T @ vcov_c_sc @ K

    # ── Derivatives (Gaussian kernel only) ──────────────────────────
    derivmat = None
    avgderiv = None
    var_avgderiv = None
    if derivative:
        derivmat_sc = np.zeros((n, d))
        var_avgderiv_sc = np.zeros((1, d))

        for k in range(d):
            # Pairwise differences in dimension k
            dist_k = X_sc[:, k:k+1] - X_sc[:, k:k+1].T   # (n, n)
            L = dist_k * K
            derivmat_sc[:, k] = (-2.0 / sigma) * (L @ coeffs).ravel()

            if vcov:
                var_avgderiv_sc[0, k] = (1.0 / n**2) * (2.0 / sigma)**2 * np.sum(L.T @ vcov_c_sc @ L)

        avgderiv_sc = np.mean(derivmat_sc, axis=0, keepdims=True)

        # Rescale back to original units
        derivmat = (y_sd * derivmat_sc) / X_sd[np.newaxis, :]
        avgderiv = (y_sd * avgderiv_sc) / X_sd[np.newaxis, :]
        var_avgderiv = (y_sd / X_sd[np.newaxis, :]) ** 2 * var_avgderiv_sc

    # ── Rescale to original units ───────────────────────────────────
    yfitted = yfitted_sc * y_sd + y_mean
    Looe = float(Le) * y_sd

    if vcov:
        vcov_c_mat = y_sd**2 * vcov_c_sc
        vcov_fitted_mat = y_sd**2 * vcov_fitted_sc

    R2 = 1.0 - float(np.var(y_init - yfitted)) / float(np.var(y_init))

    # ── Binary first differences ────────────────────────────────────
    binary_ind = np.array([[False] * d])
    fd = None
    if derivative and binary:
        fd = _first_diff_krls(X_init, X_sd, y_sd, y_mean, X_sc, sigma,
                              K, coeffs, eigenvalues, eigenvectors,
                              lambda_, eigtrunc, vcov_c_sc if vcov else None)
        if fd is not None:
            binary_ind = fd.get("binary_indicator", binary_ind)

    # ── Build result ────────────────────────────────────────────────
    result = KRLSResult(
        coeffs=coeffs,
        fitted=yfitted,
        X=X_init,
        y=y_init,
        K=K,
        sigma=sigma,
        lambda_=lambda_,
        R2=R2,
        Looe=Looe,
        derivatives=derivmat,
        avg_derivatives=avgderiv,
        var_avg_derivatives=var_avgderiv,
        vcov_c=vcov_c_mat,
        vcov_fitted=vcov_fitted_mat,
        binary_indicator=binary_ind,
        first_diff=fd,
        col_names=col_names,
    )

    if verbose >= 1 and derivative:
        result.summary()

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Prediction
# ═══════════════════════════════════════════════════════════════════════

def predict_krls(fit: KRLSResult, newdata: np.ndarray) -> np.ndarray:
    """
    Predict outcomes for new data using a fitted KRLS model.

    Parameters
    ----------
    fit : KRLSResult
        Fitted model from ``krls()``.
    newdata : np.ndarray, shape (M, D)
        New predictor matrix.

    Returns
    -------
    np.ndarray, shape (M, 1)
        Predicted values.
    """
    newdata = np.asarray(newdata, dtype=np.float64)
    if newdata.ndim == 1:
        newdata = newdata.reshape(-1, 1)

    X_train = fit.X
    n, d = X_train.shape
    X_sd = np.std(X_train, axis=0, ddof=0)
    X_mean = np.mean(X_train, axis=0)
    y_mean = float(np.mean(fit.y))
    y_sd = float(np.std(fit.y, ddof=0))

    # Scale new data with training parameters
    newdata_sc = (newdata - X_mean) / X_sd

    # Training data scaled
    X_sc = (X_train - X_mean) / X_sd

    # Kernel between new and training
    sq_dists = cdist(newdata_sc, X_sc, metric="sqeuclidean")
    K_new = np.exp(-sq_dists / fit.sigma)

    # Predict (scaled)
    y_sc = fit.y
    y_sc_mean = float(np.mean((y_sc - y_mean) / y_sd))

    ypred_sc = K_new @ fit.coeffs
    ypred = ypred_sc * y_sd + y_mean

    return ypred


# ═══════════════════════════════════════════════════════════════════════
#  First Differences for Binary Predictors
# ═══════════════════════════════════════════════════════════════════════

def _first_diff_krls(X_init, X_sd, y_sd, y_mean, X_sc, sigma,
                     K, coeffs, eigenvalues, eigenvectors, lambda_,
                     eigtrunc, vcov_c_sc):
    """Compute first differences for binary predictors (0/1)."""
    n, d = X_init.shape
    binary_indicator = np.array([[False] * d])

    for j in range(d):
        vals = np.unique(X_init[:, j])
        if len(vals) == 2 and set(vals).issubset({0, 1}):
            binary_indicator[0, j] = True

    if not np.any(binary_indicator):
        return {"binary_indicator": binary_indicator}

    fd_results = {"binary_indicator": binary_indicator}

    for j in range(d):
        if not binary_indicator[0, j]:
            continue

        # Counterfactual: flip binary predictor
        X0 = X_sc.copy()
        X1 = X_sc.copy()
        x0_val = (0 - np.mean(X_init[:, j])) / X_sd[j]
        x1_val = (1 - np.mean(X_init[:, j])) / X_sd[j]
        X0[:, j] = x0_val
        X1[:, j] = x1_val

        # Kernel matrices for counterfactuals
        K0 = np.exp(-cdist(X0, X_sc, metric="sqeuclidean") / sigma)
        K1 = np.exp(-cdist(X1, X_sc, metric="sqeuclidean") / sigma)

        yhat0 = (K0 @ coeffs) * y_sd + y_mean
        yhat1 = (K1 @ coeffs) * y_sd + y_mean
        fd = yhat1 - yhat0
        avg_fd = float(np.mean(fd))

        if vcov_c_sc is not None:
            Kdiff = K1 - K0
            var_fd = float(y_sd**2 / n**2 * np.sum(Kdiff.T @ vcov_c_sc @ Kdiff))
        else:
            var_fd = np.nan

        fd_results[f"fd_{j}"] = {
            "first_diff": fd.ravel(),
            "avg_first_diff": avg_fd,
            "var_avg_first_diff": var_fd,
        }

    return fd_results
