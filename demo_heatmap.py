"""Generate demo heatmap to test paper-style visualization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
x = np.random.randn(n)
y = 0.8 * np.sin(x) + 0.3 * x**2 + np.random.randn(n) * 0.4

from qqkrls import qqkrls, plot_qqkrls_heatmap, plot_qqkrls_3d, plot_qqkrls_contour

# Run QQKRLS with 19 quantiles (0.05 to 0.95)
res = qqkrls(
    y, x,
    y_quantiles=np.arange(0.05, 1.0, 0.05),
    x_quantiles=np.arange(0.05, 1.0, 0.05),
    n_boot=30,
    verbose=False,
)

# 1. Paper-style heatmap (red-white-green)
fig, ax = plot_qqkrls_heatmap(
    res,
    title="QQKRLS Coefficients",
    colorscale="paper",
    x_label="Quantiles of X",
    y_label="Quantiles of Y",
    save_path="demo_heatmap_paper.png",
)
plt.close(fig)
print("[OK] Paper-style heatmap saved: demo_heatmap_paper.png")

# 2. With values shown
fig, ax = plot_qqkrls_heatmap(
    res,
    title="QQKRLS Coefficients (with values)",
    colorscale="paper",
    show_values=True,
    star_fontsize=6,
    save_path="demo_heatmap_values.png",
)
plt.close(fig)
print("[OK] Heatmap with values saved: demo_heatmap_values.png")

# 3. 3D surface
fig, ax = plot_qqkrls_3d(
    res,
    title="QQKRLS 3D Surface",
    colorscale="jet",
    save_path="demo_3d_surface.png",
)
plt.close(fig)
print("[OK] 3D surface saved: demo_3d_surface.png")

# 4. Contour
fig, ax = plot_qqkrls_contour(
    res,
    title="QQKRLS Contour",
    colorscale="jet",
    save_path="demo_contour.png",
)
plt.close(fig)
print("[OK] Contour saved: demo_contour.png")

print("\nAll demos generated successfully!")
