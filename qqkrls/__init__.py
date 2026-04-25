"""
QQKRLS — Quantile-on-Quantile Kernel-Based Regularized Least Squares
=====================================================================

A comprehensive Python library implementing:

1. **KRLS** — Kernel Regularized Least Squares
   (Hainmueller & Hazlett, 2014, *Political Analysis*)

2. **QQKRLS** — Quantile-on-Quantile KRLS
   (Adebayo, Ozkan & Eweade, 2024, *Journal of Cleaner Production*)

Ported from the R CRAN packages ``KRLS`` (v1.0-0) and
``QuantileOnQuantile`` (v1.0.3), with publication-quality
MATLAB-style visualizations for top-journal output.

Author
------
Dr. Merwan Roudane
Email:  merwanroudane920@gmail.com
GitHub: https://github.com/merwanroudane/qqkrls

Modules
-------
- ``krls``      : Kernel Regularized Least Squares
- ``qqkrls``    : Quantile-on-Quantile KRLS
- ``plotting``  : MATLAB-style publication-quality plots
- ``tables``    : LaTeX tables and formatted output
- ``utils``     : Diagnostic tests (BDS, stability, normality)
"""

from ._version import __version__

# ── Core estimation ─────────────────────────────────────────────────────
from .krls import krls, KRLSResult, predict_krls, gaussian_kernel
from .qqkrls import qqkrls, QQKRLSResult

# ── Visualization ───────────────────────────────────────────────────────
from .plotting import (
    plot_qqkrls_heatmap,
    plot_qqkrls_3d,
    plot_qqkrls_contour,
    plot_qqkrls_pvalue,
    plot_qqkrls_panel,
    plot_krls_derivatives,
    plot_krls_fit,
    plot_krls_panel,
)

# ── Tables ──────────────────────────────────────────────────────────────
from .tables import (
    qqkrls_coefficient_table,
    krls_summary_table,
    descriptive_statistics,
    export_results_csv,
)

# ── Diagnostics ─────────────────────────────────────────────────────
from .utils import (
    bds_test,
    parameter_stability_test,
    jarque_bera,
)
from .diagnostics import (
    linearity_test_battery,
    print_diagnostics,
    krls_residual_diagnostics,
    print_krls_diagnostics,
    multi_qqkrls,
)

__all__ = [
    # Core
    "krls", "KRLSResult", "predict_krls", "gaussian_kernel",
    "qqkrls", "QQKRLSResult",
    # Plots
    "plot_qqkrls_heatmap", "plot_qqkrls_3d", "plot_qqkrls_contour",
    "plot_qqkrls_pvalue", "plot_qqkrls_panel",
    "plot_krls_derivatives", "plot_krls_fit", "plot_krls_panel",
    # Tables
    "qqkrls_coefficient_table", "krls_summary_table",
    "descriptive_statistics", "export_results_csv",
    # Diagnostics
    "bds_test", "parameter_stability_test", "jarque_bera",
    "linearity_test_battery", "print_diagnostics",
    "krls_residual_diagnostics", "print_krls_diagnostics",
    "multi_qqkrls",
]
