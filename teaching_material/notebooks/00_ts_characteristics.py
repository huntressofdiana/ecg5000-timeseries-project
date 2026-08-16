import os
import numpy as np
from matplotlib import pyplot as plt

from cpsmehelper import get_colors, export_figure
cpsme_colors = get_colors()
plt.style.use(os.path.join(os.getcwd(), 'cps_presentation.mplstyle'))
fig_path = os.path.join(os.getcwd(), 'teaching_material', 'graphics')

rng = np.random.default_rng(7)
t = np.arange(0, 120)

# 1) Positive trend
y_trend = 0.06 * t + rng.normal(0, 0.5, size=t.size)

# 2) Irregular time series (heteroskedastic + occasional shocks)
base_irregular = np.cumsum(rng.normal(0, 0.12, size=t.size))
noise_scale = np.linspace(0.2, 1.0, t.size)
irregular_noise = rng.normal(0, noise_scale)
shocks = np.zeros_like(t, dtype=float)
shock_idx = rng.choice(t.size, size=7, replace=False)
shocks[shock_idx] = rng.normal(0, 2.8, size=shock_idx.size)
y_irregular = base_irregular + irregular_noise + shocks

# 3) Seasonality (fixed period and amplitude)
y_seasonal = 1.8 * np.sin(2 * np.pi * t / 12) + rng.normal(
    0, 0.25, size=t.size
)

# 4) Cyclic behavior (longer, less regular cycles)
slow_cycle = 2.0 * np.sin(2 * np.pi * t / 42)
modulation = 0.9 * np.sin(2 * np.pi * t / 95 + 0.8)
y_cyclic = slow_cycle + modulation + rng.normal(0, 0.2, size=t.size)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

axes[0, 0].plot(t, y_trend, linewidth=1.8)
axes[0, 0].set_title('Positive Trend')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.25)

axes[0, 1].plot(t, y_irregular, linewidth=1.6)
axes[0, 1].set_title('Irregular Time Series')
axes[0, 1].grid(True, alpha=0.25)

axes[1, 0].plot(t, y_seasonal, linewidth=1.8)
axes[1, 0].set_title('Seasonality')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.25)

axes[1, 1].plot(t, y_cyclic, linewidth=1.8)
axes[1, 1].set_title('Cyclic')
axes[1, 1].set_xlabel('Time')
axes[1, 1].grid(True, alpha=0.25)

fig.tight_layout()
export_figure(fig, os.path.join(fig_path, '00_time_series_characteristics.png'), width=20, height=13, resolution=100)
plt.show()


