"""
Publication-quality MATLAB-style visualizations for KRLS & QQKRLS.

All plots use the MATLAB Jet colormap for consistency with top-journal
econometric publications.  Supports heatmaps, 3D surfaces, contour
plots, marginal-effect distributions, and derivative scatter plots.

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
#  Global Style Configuration
# ═══════════════════════════════════════════════════════════════════════

# MATLAB Jet-style colormaps
def _get_cmap(name):
    """Get colormap in a way compatible with all matplotlib versions."""
    try:
        return mpl.colormaps[name]
    except (AttributeError, KeyError):
        return cm.get_cmap(name)

_JET = _get_cmap("jet")
_VIRIDIS = _get_cmap("viridis")
_PLASMA = _get_cmap("plasma")
_COOLWARM = _get_cmap("coolwarm")
_RDYLGN = _get_cmap("RdYlGn")

CMAP_REGISTRY = {
    "jet": _JET,
    "viridis": _VIRIDIS,
    "plasma": _PLASMA,
    "coolwarm": _COOLWARM,
    "rdylgn": _RDYLGN,
    "bluered": _COOLWARM,
}


def _journal_style():
    """Apply journal-quality matplotlib rc settings."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.linewidth": 0.8,
    })


# ═══════════════════════════════════════════════════════════════════════
#  Custom Paper-Exact Colormaps
# ═══════════════════════════════════════════════════════════════════════

def _paper_cmap():
    """Red-White-Green diverging colormap matching Adebayo et al. (2024)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "paper_rwg",
        [
            (0.00, "#c62828"),   # deep red  (most negative)
            (0.15, "#e57373"),   # medium red
            (0.30, "#ffcdd2"),   # light red / pink
            (0.45, "#fff8e1"),   # warm cream
            (0.50, "#ffffff"),   # pure white (zero)
            (0.55, "#f1f8e9"),   # faint green
            (0.70, "#c5e1a5"),   # light green
            (0.85, "#66bb6a"),   # medium green
            (1.00, "#1b5e20"),   # deep green (most positive)
        ],
        N=512,
    )


def _paper_cmap_warm():
    """Red-White warm colormap (for mostly-negative data like the paper)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "paper_rw",
        [
            (0.00, "#b71c1c"),   # deep red
            (0.20, "#e53935"),   # red
            (0.40, "#ef9a9a"),   # light red
            (0.60, "#ffcdd2"),   # pink
            (0.80, "#fce4ec"),   # faint pink
            (0.90, "#fff3e0"),   # warm cream
            (1.00, "#ffffff"),   # white
        ],
        N=512,
    )


# ═══════════════════════════════════════════════════════════════════════
#  1. QQKRLS Coefficient Heatmap  (Paper-style with significance stars)
# ═══════════════════════════════════════════════════════════════════════

def plot_qqkrls_heatmap(result, title: str = "QQKRLS Coefficients",
                        colorscale: str = "paper",
                        show_stars: bool = True,
                        show_values: bool = False,
                        x_label: Optional[str] = None,
                        y_label: Optional[str] = None,
                        figsize: tuple = (12, 9),
                        save_path: Optional[str] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        star_fontsize: float = 8,
                        dpi: int = 300):
    """
    Publication-quality heatmap of QQKRLS coefficients.

    Matches the exact style of Adebayo et al. (2024, JCP, Fig. 3):
    red-white-green diverging colors with significance stars.

    Parameters
    ----------
    result : QQKRLSResult
        Output from ``qqkrls()``.
    title : str
        Plot title (appears as figure caption style below).
    colorscale : str
        'paper' (red-white-green, DEFAULT),
        'paper_warm' (red-white only),
        'jet', 'rdylgn', 'coolwarm', 'viridis', 'plasma'.
    show_stars : bool
        Annotate significance stars (***, **, *).
    show_values : bool
        Also show coefficient values in each cell.
    x_label : str or None
        Custom X-axis label (default: "Quantiles of X").
    y_label : str or None
        Custom Y-axis label (default: "Quantiles of Y").
    figsize : tuple
    save_path : str or None
    vmin, vmax : float or None
        Colorbar range.  If None, auto-symmetric around zero.
    star_fontsize : float
        Font size for significance stars.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    (fig, ax)
    """
    _journal_style()

    # Select colormap
    if colorscale.lower() == "paper":
        cmap = _paper_cmap()
    elif colorscale.lower() == "paper_warm":
        cmap = _paper_cmap_warm()
    else:
        cmap = CMAP_REGISTRY.get(colorscale.lower(), _JET)

    mat_df = result.results.dropna(subset=["coefficient"]).pivot(
        index="y_quantile", columns="x_quantile", values="coefficient"
    )
    mat = mat_df.values
    y_q = list(mat_df.index)
    x_q = list(mat_df.columns)
    ny, nx = mat.shape

    # Auto-symmetric range around zero
    if vmin is None or vmax is None:
        abs_max = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        if abs_max == 0:
            abs_max = 1.0
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max

    # ── Figure ──
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        mat, cmap=cmap, aspect="auto", origin="lower",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )

    # Grid lines between cells
    for i in range(ny + 1):
        ax.axhline(i - 0.5, color="#cccccc", linewidth=0.5)
    for j in range(nx + 1):
        ax.axvline(j - 0.5, color="#cccccc", linewidth=0.5)

    # ── Axis labels ──
    ax.set_xticks(range(nx))
    ax.set_xticklabels([f"{q:.2f}" for q in x_q], fontsize=9)
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{q:.2f}" for q in y_q], fontsize=9)

    if x_label is None:
        x_label = r"Quantiles of $X$"
    if y_label is None:
        y_label = r"Quantiles of $Y$"

    ax.set_xlabel(x_label, fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold", labelpad=10)

    # ── Significance stars ──
    if show_stars:
        # Build stars from the same pivot to guarantee dimension match
        def _s(p):
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.10: return "*"
            return ""
        pv_df = result.results.dropna(subset=["coefficient"]).pivot(
            index="y_quantile", columns="x_quantile", values="p_value"
        )
        stars_mat = pv_df.map(_s).values if hasattr(pv_df, 'map') else pv_df.applymap(_s).values
        for i in range(ny):
            for j in range(nx):
                val = mat[i, j]
                star = stars_mat[i, j] if i < stars_mat.shape[0] and j < stars_mat.shape[1] else ""
                if not np.isfinite(val):
                    continue

                # Build cell text
                parts = []
                if show_values:
                    parts.append(f"{val:.3f}")
                if star:
                    parts.append(str(star))
                txt = "\n".join(parts) if parts else ""

                if txt:
                    # Adaptive text colour: dark on light, light on dark
                    norm_val = (val - vmin) / (vmax - vmin + 1e-12)
                    if norm_val < 0.25 or norm_val > 0.75:
                        txt_color = "#222222"
                    else:
                        txt_color = "#333333"

                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=star_fontsize, color=txt_color,
                            fontweight="bold", fontfamily="serif")

    # ── Colorbar ──
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.5)

    # ── Title as figure caption ──
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14,
                 style="italic")

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  2. QQKRLS 3D Surface Plot  (MATLAB-style)
# ═══════════════════════════════════════════════════════════════════════

def plot_qqkrls_3d(result, title: str = "QQKRLS Surface",
                   colorscale: str = "jet",
                   figsize: tuple = (12, 9),
                   elev: float = 25, azim: float = -55,
                   save_path: Optional[str] = None):
    """
    3D surface plot of QQKRLS coefficients.

    Parameters
    ----------
    result : QQKRLSResult
    title : str
    colorscale : str
    figsize : tuple
    elev, azim : float
        Camera elevation and azimuth angles.
    save_path : str or None
    """
    _journal_style()
    cmap = CMAP_REGISTRY.get(colorscale.lower(), _JET)

    mat = result.to_matrix("coefficient")
    y_q = sorted(result.results["y_quantile"].dropna().unique())
    x_q = sorted(result.results["x_quantile"].dropna().unique())

    X_mesh, Y_mesh = np.meshgrid(x_q, y_q)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X_mesh, Y_mesh, mat, cmap=cmap,
                           edgecolor="k", linewidth=0.3, alpha=0.92,
                           antialiased=True)

    ax.set_xlabel(r"$X$ quantile ($\theta$)", fontsize=11, labelpad=12)
    ax.set_ylabel(r"$Y$ quantile ($\tau$)", fontsize=11, labelpad=12)
    ax.set_zlabel("Coefficient", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.view_init(elev=elev, azim=azim)

    # Colorbar
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(mat)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Avg Pointwise ME", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  3. QQKRLS Contour Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_qqkrls_contour(result, title: str = "QQKRLS Contour",
                        colorscale: str = "jet",
                        levels: int = 20,
                        figsize: tuple = (10, 8),
                        save_path: Optional[str] = None):
    """Filled contour plot of QQKRLS coefficients."""
    _journal_style()
    cmap = CMAP_REGISTRY.get(colorscale.lower(), _JET)

    mat = result.to_matrix("coefficient")
    y_q = sorted(result.results["y_quantile"].dropna().unique())
    x_q = sorted(result.results["x_quantile"].dropna().unique())
    X_mesh, Y_mesh = np.meshgrid(x_q, y_q)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X_mesh, Y_mesh, mat, levels=levels, cmap=cmap)
    ax.contour(X_mesh, Y_mesh, mat, levels=levels, colors="k",
               linewidths=0.3, alpha=0.5)

    ax.set_xlabel(r"$X$ quantile ($\theta$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$Y$ quantile ($\tau$)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label("Coefficient", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  4. QQKRLS Significance Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_qqkrls_pvalue(result, title: str = "QQKRLS P-Values",
                       alpha: float = 0.05,
                       figsize: tuple = (10, 8),
                       save_path: Optional[str] = None):
    """Heatmap showing p-values and significance regions."""
    _journal_style()

    mat_df = result.results.dropna(subset=["p_value"]).pivot(
        index="y_quantile", columns="x_quantile", values="p_value"
    )
    pmat = mat_df.values
    y_q = list(mat_df.index)
    x_q = list(mat_df.columns)
    ny, nx = pmat.shape

    # Custom colormap: significant=green, borderline=yellow, not-sig=red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sig", [(0, "#1a9850"), (0.01, "#1a9850"),
                (0.05, "#a6d96a"), (0.10, "#fee08b"),
                (0.5, "#fdae61"), (1.0, "#d73027")]
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pmat, cmap=cmap, aspect="auto", origin="lower",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(nx))
    ax.set_xticklabels([f"{q:.2f}" for q in x_q], rotation=45, ha="right")
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{q:.2f}" for q in y_q])
    ax.set_xlabel(r"$X$ quantile ($\theta$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$Y$ quantile ($\tau$)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Annotate
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            p = pmat[i, j]
            if np.isfinite(p):
                star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                ax.text(j, i, f"{p:.3f}\n{star}", ha="center", va="center",
                        fontsize=6.5, color="white" if p < 0.05 else "black",
                        fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("p-value", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  5. KRLS Marginal Effects Distribution
# ═══════════════════════════════════════════════════════════════════════

def plot_krls_derivatives(fit, var_idx: int = 0,
                          title: Optional[str] = None,
                          figsize: tuple = (10, 6),
                          save_path: Optional[str] = None):
    """
    Plot pointwise marginal effects from KRLS.

    Top panel: scatter of effects against predictor values with CI.
    Bottom panel: density/histogram of marginal effects.
    """
    _journal_style()

    if fit.derivatives is None:
        raise ValueError("No derivatives in KRLS fit (set derivative=True)")

    derivs = fit.derivatives[:, var_idx]
    x_vals = fit.X[:, var_idx]
    name = fit.col_names[var_idx] if var_idx < len(fit.col_names) else f"x{var_idx+1}"

    avg_d = fit.avg_derivatives[0, var_idx]
    se_d = np.sqrt(fit.var_avg_derivatives[0, var_idx]) if fit.var_avg_derivatives is not None else 0

    if title is None:
        title = f"KRLS Marginal Effects: {name}"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # ── Top: Scatter ──
    ax1.scatter(x_vals, derivs, c=derivs, cmap="jet", s=25, alpha=0.7,
                edgecolors="k", linewidths=0.3, zorder=3)
    ax1.axhline(avg_d, color="#d62728", lw=2, ls="--",
                label=f"Avg ME = {avg_d:.4f}", zorder=4)
    if se_d > 0:
        ax1.axhline(avg_d + 1.96 * se_d, color="#d62728", lw=1, ls=":",
                     alpha=0.6, zorder=4)
        ax1.axhline(avg_d - 1.96 * se_d, color="#d62728", lw=1, ls=":",
                     alpha=0.6, zorder=4)
        ax1.fill_between([x_vals.min(), x_vals.max()],
                         avg_d - 1.96 * se_d, avg_d + 1.96 * se_d,
                         color="#d62728", alpha=0.08, zorder=1)
    ax1.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.5, zorder=2)
    ax1.set_ylabel("Pointwise Marginal Effect", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9)

    # ── Bottom: Histogram ──
    ax2.hist(derivs, bins=30, color="#3274A1", edgecolor="white",
             alpha=0.85, density=True, zorder=3)
    ax2.axvline(avg_d, color="#d62728", lw=2, ls="--", zorder=4)
    ax2.axvline(0, color="gray", lw=0.8, ls="-", alpha=0.5, zorder=2)
    ax2.set_xlabel(f"Marginal Effect of {name}", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


# ═══════════════════════════════════════════════════════════════════════
#  6. KRLS Fitted vs Actual
# ═══════════════════════════════════════════════════════════════════════

def plot_krls_fit(fit, title: str = "KRLS: Fitted vs Actual",
                  figsize: tuple = (8, 8),
                  save_path: Optional[str] = None):
    """Scatter plot of actual vs fitted values with 45-degree line."""
    _journal_style()

    actual = fit.y.ravel()
    fitted = fit.fitted.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, fitted, c="#3274A1", s=30, alpha=0.7,
               edgecolors="k", linewidths=0.3, zorder=3)

    lims = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
    margin = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "r--", lw=1.5, alpha=0.7, zorder=4)

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Fitted", fontsize=12)
    ax.set_title(f"{title}  (R² = {fit.R2:.4f})", fontsize=13, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  7. QQKRLS Paper-Style Panel  (multi-variable)
# ═══════════════════════════════════════════════════════════════════════

def plot_qqkrls_panel(results_dict: dict,
                      dep_var: str = "Y",
                      colorscale: str = "rdylgn",
                      figsize: Optional[tuple] = None,
                      save_path: Optional[str] = None):
    """
    Create a panel of QQKRLS heatmaps — one per independent variable.

    Parameters
    ----------
    results_dict : dict
        Mapping ``{var_name: QQKRLSResult}``.
    dep_var : str
        Name of the dependent variable (for titles).
    colorscale : str
    figsize : tuple or None
    save_path : str or None
    """
    _journal_style()
    cmap = CMAP_REGISTRY.get(colorscale.lower(), _RDYLGN)

    n_vars = len(results_dict)
    if figsize is None:
        figsize = (6 * min(n_vars, 3), 5 * ((n_vars - 1) // 3 + 1))

    ncols = min(n_vars, 3)
    nrows = (n_vars - 1) // ncols + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (var_name, res) in enumerate(results_dict.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        mat_df = res.results.dropna(subset=["coefficient"]).pivot(
            index="y_quantile", columns="x_quantile", values="coefficient"
        )
        mat = mat_df.values
        y_q = list(mat_df.index)
        x_q = list(mat_df.columns)

        abs_max = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        if abs_max == 0:
            abs_max = 1

        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower",
                       vmin=-abs_max, vmax=abs_max, interpolation="nearest")

        ax.set_xticks(range(0, len(x_q), max(1, len(x_q)//5)))
        ax.set_xticklabels([f"{x_q[i]:.2f}" for i in range(0, len(x_q), max(1, len(x_q)//5))],
                           rotation=45, fontsize=8)
        ax.set_yticks(range(0, len(y_q), max(1, len(y_q)//5)))
        ax.set_yticklabels([f"{y_q[i]:.2f}" for i in range(0, len(y_q), max(1, len(y_q)//5))],
                           fontsize=8)

        ax.set_xlabel(f"Quantiles of {var_name}", fontsize=9)
        ax.set_ylabel(f"Quantiles of {dep_var}", fontsize=9)
        ax.set_title(f"QQKRLS: {var_name} → {dep_var}", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_vars, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"QQKRLS Analysis — Dependent: {dep_var}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════
#  8. KRLS Summary Panel  (all predictors)
# ═══════════════════════════════════════════════════════════════════════

def plot_krls_panel(fit, figsize: Optional[tuple] = None,
                    save_path: Optional[str] = None):
    """
    Panel of KRLS marginal-effect scatter plots — one per predictor.
    """
    _journal_style()

    if fit.derivatives is None:
        raise ValueError("No derivatives available")

    d = fit.X.shape[1]
    if figsize is None:
        figsize = (6 * min(d, 3), 5 * ((d - 1) // 3 + 1))

    ncols = min(d, 3)
    nrows = (d - 1) // ncols + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for j in range(d):
        r, c = divmod(j, ncols)
        ax = axes[r, c]
        derivs = fit.derivatives[:, j]
        x_vals = fit.X[:, j]
        name = fit.col_names[j] if j < len(fit.col_names) else f"x{j+1}"
        avg_d = fit.avg_derivatives[0, j]

        sc = ax.scatter(x_vals, derivs, c=derivs, cmap="jet", s=20,
                        alpha=0.7, edgecolors="k", linewidths=0.2)
        ax.axhline(avg_d, color="#d62728", lw=1.5, ls="--")
        ax.axhline(0, color="gray", lw=0.7, alpha=0.5)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("ME", fontsize=10)
        ax.set_title(f"ME of {name} (avg={avg_d:.3f})", fontsize=10, fontweight="bold")

    for j in range(d, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("KRLS Pointwise Marginal Effects", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
