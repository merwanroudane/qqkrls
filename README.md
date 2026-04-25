# QQKRLS — Quantile-on-Quantile Kernel-Based Regularized Least Squares

[![PyPI version](https://badge.fury.io/py/qqkrls.svg)](https://pypi.org/project/qqkrls/)
[![Python](https://img.shields.io/pypi/pyversions/qqkrls.svg)](https://pypi.org/project/qqkrls/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library implementing **KRLS** (Kernel Regularized Least Squares) and **QQKRLS** (Quantile-on-Quantile KRLS) with publication-quality MATLAB-style visualizations.

## Overview

**QQKRLS** combines two powerful econometric methodologies:

1. **KRLS** — A machine-learning regression method using Gaussian kernels and Tikhonov regularization to fit flexible, nonparametric functions without linearity or additivity assumptions (Hainmueller & Hazlett, 2014).

2. **Quantile-on-Quantile Regression** — A distributional analysis framework that examines how quantiles of an independent variable affect quantiles of a dependent variable (Sim & Zhou, 2015).

The combined **QQKRLS** method (Adebayo et al., 2024) provides nonparametric marginal effects across all quantile pairs, enabling researchers to capture nonlinear, heterogeneous relationships between variables at different distributional locations.

## Installation

```bash
pip install qqkrls
```

For full functionality including interactive plots:
```bash
pip install qqkrls[full]
```

## Quick Start

### KRLS — Kernel Regularized Least Squares

```python
import numpy as np
from qqkrls import krls, plot_krls_derivatives, plot_krls_fit

# Generate data with nonlinear relationship
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.randn(n) * 0.3

# Fit KRLS
fit = krls(X, y, col_names=["x1", "x2"])

# Visualize marginal effects
plot_krls_derivatives(fit, var_idx=0, title="Marginal Effect of x1")
plot_krls_fit(fit)
```

### QQKRLS — Quantile-on-Quantile KRLS

```python
import numpy as np
from qqkrls import qqkrls, plot_qqkrls_heatmap, plot_qqkrls_3d

# Generate data
np.random.seed(42)
x = np.random.randn(200)
y = 0.5 * np.sin(x) + np.random.randn(200) * 0.3

# Run QQKRLS
result = qqkrls(y, x, n_boot=100)

# Publication-quality heatmap (like Adebayo et al., 2024)
plot_qqkrls_heatmap(result, title="QQKRLS: X → Y", colorscale="rdylgn")

# 3D surface plot (MATLAB-style)
plot_qqkrls_3d(result, title="QQKRLS Surface", colorscale="jet")

# Export LaTeX table
latex = result.export_latex(caption="QQKRLS Coefficients")
print(latex)
```

### Diagnostic Tests

```python
from qqkrls import bds_test, parameter_stability_test, jarque_bera

# BDS nonlinearity test
bds = bds_test(series, max_dim=6)

# Andrews parameter stability test
stability = parameter_stability_test(series)

# Jarque-Bera normality test
jb = jarque_bera(series)
```

## Features

| Feature | Description |
|---------|-------------|
| **KRLS** | Gaussian kernel regression with LOO cross-validation for λ |
| **QQKRLS** | Nonparametric marginal effects across quantile pairs |
| **Marginal Effects** | Pointwise, average, and quantile-specific derivatives |
| **Variance-Covariance** | Closed-form and bootstrap inference |
| **Diagnostics** | BDS, Andrews stability, Jarque-Bera tests |
| **Visualizations** | Heatmaps, 3D surfaces, contours, panels (MATLAB Jet style) |
| **LaTeX Tables** | Publication-ready tables with significance stars |
| **Predictions** | Out-of-sample prediction with fitted KRLS models |

## Methodology

### KRLS

The KRLS estimator solves:

$$c^* = (K + \lambda I)^{-1} y$$

where $K_{ij} = \exp(-\|x_i - x_j\|^2 / \sigma^2)$ is the Gaussian kernel matrix.

Pointwise marginal effects are computed analytically:

$$\frac{\partial \hat{y}_j}{\partial x_j^{(d)}} = \frac{-2}{\sigma^2} \sum_i c_i \cdot K_{ji} \cdot (x_i^{(d)} - x_j^{(d)})$$

### QQKRLS

For each quantile pair $(\theta, \tau)$:

$$E_N\left[\frac{\partial Q_{Y_\tau}}{\partial Q_{X_\theta,k}}\right] = \frac{-2}{\sigma^2 N} \sum_k \sum_i c_i \cdot e^{-\|X_{\theta,i} - X_{\theta,k}\|^2 / \sigma^2} \cdot (X_{\theta,i} - X_{\theta,k})$$

## Ported From

- **R CRAN `KRLS`** (v1.0-0) by Hainmueller & Hazlett
- **R CRAN `QuantileOnQuantile`** (v1.0.3) by Roudane
- **Python `wqr`** (v1.0.1) by Roudane

## References

1. Hainmueller, J. & Hazlett, C. (2014). Kernel Regularized Least Squares. *Political Analysis*, 22(2), 143-168. [doi:10.1093/pan/mpt024](https://doi.org/10.1093/pan/mpt024)

2. Sim, N. & Zhou, H. (2015). Oil Prices, US Stock Return, and the Dependence Between Their Quantiles. *Journal of Banking & Finance*, 55, 1-12. [doi:10.1016/j.jbankfin.2015.01.013](https://doi.org/10.1016/j.jbankfin.2015.01.013)

3. Adebayo, T.S., Ozkan, O. & Eweade, B.S. (2024). Do energy efficiency R&D investments and ICT promote environmental sustainability in Sweden? A QQKRLS investigation. *Journal of Cleaner Production*, 440, 140832. [doi:10.1016/j.jclepro.2024.140832](https://doi.org/10.1016/j.jclepro.2024.140832)

4. Adebayo, T.S., Meo, M.S., Eweade, B.S. & Ozkan, O. (2024). Analyzing the effects of solar energy innovations, digitalization, and economic globalization on environmental quality in the United States. *Clean Technologies and Environmental Policy*, 26, 4157-4176. [doi:10.1007/s10098-024-02831-0](https://doi.org/10.1007/s10098-024-02831-0)

## Author

**Dr. Merwan Roudane**
- Email: merwanroudane920@gmail.com
- GitHub: [github.com/merwanroudane/qqkrls](https://github.com/merwanroudane/qqkrls)

## License

MIT License. See [LICENSE](LICENSE) for details.
