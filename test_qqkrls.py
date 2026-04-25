"""
Full test script for the qqkrls package.
Tests KRLS, QQKRLS, diagnostics, and generates publication-quality plots.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

print("=" * 65)
print("  QQKRLS Package — Full Test Suite")
print("=" * 65)

# ── 1. Test KRLS ────────────────────────────────────────────────────
print("\n[1] Testing KRLS...")
from qqkrls import krls

np.random.seed(42)
n = 150
X = np.random.randn(n, 2)
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.randn(n) * 0.3

fit = krls(X, y, col_names=["sin_var", "quad_var"], verbose=0)
print(f"  R²     = {fit.R2:.4f}")
print(f"  Lambda = {fit.lambda_:.6f}")
print(f"  Sigma  = {fit.sigma:.2f}")
print(f"  Avg ME (sin_var)  = {fit.avg_derivatives[0,0]:.4f}")
print(f"  Avg ME (quad_var) = {fit.avg_derivatives[0,1]:.4f}")
assert fit.R2 > 0.5, f"R² too low: {fit.R2}"
assert fit.derivatives.shape == (n, 2)
print("  [OK] KRLS passed!")

# ── 2. Test KRLS Prediction ─────────────────────────────────────────
print("\n[2] Testing KRLS Prediction...")
X_new = np.random.randn(20, 2)
y_pred = fit.predict(X_new)
assert y_pred.shape == (20, 1), f"Wrong shape: {y_pred.shape}"
print(f"  Predictions range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
print("  [OK] Prediction passed!")

# ── 3. Test KRLS Summary ────────────────────────────────────────────
print("\n[3] Testing KRLS Summary...")
fit.summary()
print("  [OK] Summary passed!")

# ── 4. Test QQKRLS ──────────────────────────────────────────────────
print("\n[4] Testing QQKRLS...")
from qqkrls import qqkrls

np.random.seed(123)
n2 = 200
x_qq = np.random.randn(n2)
y_qq = 0.8 * np.sin(x_qq) + 0.3 * x_qq**2 + np.random.randn(n2) * 0.4

result = qqkrls(
    y_qq, x_qq,
    y_quantiles=np.arange(0.1, 1.0, 0.2),   # 5 quantiles for speed
    x_quantiles=np.arange(0.1, 1.0, 0.2),
    n_boot=30,  # small for speed
    verbose=True,
)

mat = result.to_matrix("coefficient")
print(f"  Coefficient matrix shape: {mat.shape}")
print(f"  Coefficient range: [{np.nanmin(mat):.4f}, {np.nanmax(mat):.4f}]")
assert mat.shape[0] > 0 and mat.shape[1] > 0
print("  [OK] QQKRLS passed!")

# ── 5. Test QQKRLS Summary & Export ──────────────────────────────────
print("\n[5] Testing QQKRLS Summary & Export...")
result.summary()
latex = result.export_latex(caption="Test QQKRLS Results")
print(f"  LaTeX table: {len(latex)} characters")
assert r"\begin{table}" in latex
print("  [OK] Summary & Export passed!")

# ── 6. Test Diagnostic Tests ────────────────────────────────────────
print("\n[6] Testing Diagnostic Tests...")
from qqkrls import bds_test, parameter_stability_test, jarque_bera

bds = bds_test(y_qq, max_dim=4)
print(f"  BDS z-stats: {[f'{z:.2f}' for z in bds['z_stats']]}")
print(f"  BDS p-vals:  {[f'{p:.4f}' for p in bds['p_values']]}")

stab = parameter_stability_test(y_qq)
print(f"  Max-F: {stab['max_f']:.2f}, Exp-F: {stab['exp_f']:.2f}, Ave-F: {stab['ave_f']:.2f}")

jb = jarque_bera(y_qq)
print(f"  JB stat: {jb['statistic']:.3f}, p-value: {jb['p_value']:.4f}")
print("  [OK] Diagnostics passed!")

# ── 7. Test Plotting ────────────────────────────────────────────────
print("\n[7] Testing Plots...")
from qqkrls import (
    plot_qqkrls_heatmap, plot_qqkrls_3d, plot_qqkrls_contour,
    plot_qqkrls_pvalue, plot_krls_derivatives, plot_krls_fit,
    plot_krls_panel,
)

fig, ax = plot_qqkrls_heatmap(result, title="Test QQKRLS Heatmap", colorscale="rdylgn",
                               save_path="test_heatmap.png")
plt.close(fig)
print("  [OK] QQKRLS Heatmap saved")

fig, ax = plot_qqkrls_3d(result, title="Test QQKRLS 3D", colorscale="jet",
                          save_path="test_3d.png")
plt.close(fig)
print("  [OK] QQKRLS 3D surface saved")

fig, ax = plot_qqkrls_contour(result, title="Test Contour", save_path="test_contour.png")
plt.close(fig)
print("  [OK] QQKRLS Contour saved")

fig, ax = plot_qqkrls_pvalue(result, title="Test P-Values", save_path="test_pvalue.png")
plt.close(fig)
print("  [OK] QQKRLS P-Value heatmap saved")

fig, axes = plot_krls_derivatives(fit, var_idx=0, save_path="test_derivs.png")
plt.close(fig)
print("  [OK] KRLS Derivatives saved")

fig, ax = plot_krls_fit(fit, save_path="test_fit.png")
plt.close(fig)
print("  [OK] KRLS Fit plot saved")

fig, axes = plot_krls_panel(fit, save_path="test_panel.png")
plt.close(fig)
print("  [OK] KRLS Panel saved")

# ── 8. Test Tables ──────────────────────────────────────────────────
print("\n[8] Testing Tables...")
from qqkrls import krls_summary_table, descriptive_statistics
import pandas as pd

latex_krls = krls_summary_table(fit)
print(f"  KRLS LaTeX table: {len(latex_krls)} characters")
assert r"\begin{table}" in latex_krls

df_data = pd.DataFrame({"y": y_qq, "x": x_qq})
desc_latex = descriptive_statistics(df_data, caption="Test Descriptives")
print(f"  Descriptive stats LaTeX: {len(desc_latex)} characters")
assert "Skewness" in desc_latex
print("  [OK] Tables passed!")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 65)
