"""
Publication-quality LaTeX tables and formatted output for KRLS & QQKRLS.

Author: Dr. Merwan Roudane  <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd
from typing import Optional


def qqkrls_coefficient_table(result, digits: int = 3,
                             show_stars: bool = True) -> str:
    """
    Format QQKRLS coefficients as a publication-ready LaTeX table.

    Parameters
    ----------
    result : QQKRLSResult
    digits : int
    show_stars : bool

    Returns
    -------
    str
        LaTeX table code.
    """
    return result.export_latex(
        value="coefficient",
        caption="QQKRLS Average Pointwise Marginal Effects",
        show_stars=show_stars,
    )


def krls_summary_table(fit, digits: int = 4) -> str:
    """
    Format KRLS average marginal effects as a LaTeX table.

    Parameters
    ----------
    fit : KRLSResult
    digits : int

    Returns
    -------
    str
        LaTeX table code.
    """
    from scipy import stats as _st

    d = fit.X.shape[1]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{KRLS Average Marginal Effects}",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"Variable & Avg ME & Std Err & $t$-value & $p$-value & Sig. \\",
        r"\hline",
    ]

    for j in range(d):
        name = fit.col_names[j] if j < len(fit.col_names) else f"x{j+1}"
        avg = fit.avg_derivatives[0, j]
        se = np.sqrt(fit.var_avg_derivatives[0, j]) if fit.var_avg_derivatives is not None else np.nan
        tval = avg / se if se > 0 else np.nan
        pval = 2 * _st.norm.sf(abs(tval)) if np.isfinite(tval) else np.nan
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        lines.append(
            f"{name} & {avg:.{digits}f} & {se:.{digits}f} & "
            f"{tval:.{digits}f} & {pval:.{digits}f} & {stars} \\\\"
        )

    lines += [
        r"\hline",
        f"\\multicolumn{{6}}{{l}}{{$N = {fit.X.shape[0]}$, "
        f"$R^2 = {fit.R2:.4f}$, "
        f"$\\sigma^2 = {fit.sigma:.2f}$, "
        f"$\\lambda = {fit.lambda_:.6f}$}} \\\\",
        r"\hline",
        r"\multicolumn{6}{l}{\footnotesize *** $p<0.01$, ** $p<0.05$, * $p<0.10$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def descriptive_statistics(data: pd.DataFrame,
                           caption: str = "Descriptive Statistics",
                           digits: int = 3) -> str:
    """
    Generate a publication-quality descriptive statistics table.

    Parameters
    ----------
    data : pd.DataFrame
        Columns to describe.
    caption : str
    digits : int

    Returns
    -------
    str
        LaTeX table code.
    """
    desc = data.describe().T
    # Add Skewness, Kurtosis, JB test
    from scipy import stats as _st

    rows = []
    for col in data.columns:
        s = data[col].dropna()
        n = len(s)
        skew = float(_st.skew(s))
        kurt = float(_st.kurtosis(s, fisher=True) + 3)  # excess → raw
        jb_stat = (n / 6) * (skew ** 2 + (kurt - 3) ** 2 / 4)
        jb_pval = 1 - _st.chi2.cdf(jb_stat, 2)

        rows.append({
            "Variable": col,
            "Mean": s.mean(),
            "Median": s.median(),
            "Max": s.max(),
            "Min": s.min(),
            "Std.Dev": s.std(),
            "Skewness": skew,
            "Kurtosis": kurt,
            "JB": jb_stat,
            "JB p": jb_pval,
            "N": n,
        })

    df = pd.DataFrame(rows)

    header = " & ".join(df.columns[1:])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}",
        r"\hline",
        "Variable & " + header + r" \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        vals = [row["Variable"]]
        for c in df.columns[1:]:
            v = row[c]
            if c == "N":
                vals.append(f"{int(v)}")
            elif c == "JB p":
                stars = "***" if v < 0.01 else "**" if v < 0.05 else "*" if v < 0.10 else ""
                vals.append(f"{v:.{digits}f}{stars}")
            else:
                vals.append(f"{v:.{digits}f}")
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\hline",
        r"\multicolumn{" + str(len(df.columns)) + r"}{l}{\footnotesize *** $p<0.01$, ** $p<0.05$, * $p<0.10$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def export_results_csv(result, path: str, digits: int = 4):
    """Export QQKRLS or KRLS results to CSV."""
    result.export_csv(path, digits=digits)
