import numpy as np
import matplotlib.pyplot as plt
import os

from cpsmehelper import get_colors, export_figure
cpsme_colors = get_colors()
plt.style.use(os.path.join(os.getcwd(), 'cps_presentation.mplstyle'))
fig_path = os.path.join(os.getcwd(), 'teaching_material', 'graphics')

rng = np.random.default_rng(3)

n_hist, n_fut = 130, 90
t = np.arange(n_hist + n_fut)
t_now = n_hist - 1

def f(t):
    return 1.0 * np.sin(2 * np.pi * t / 25) + 0.45 * np.sin(2 * np.pi * t / 9 + 1.0)

y_true = f(t)
y_obs = y_true[:n_hist] + 0.08 * rng.standard_normal(n_hist)

h = np.arange(n_fut + 1)
mu = f(t[t_now:]) * np.exp(-h / 45)
sigma = 0.10 + 0.80 * np.sqrt(1 - np.exp(-h / 25))

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.axvspan(t_now, t[-1], color="0.96", zorder=0)
ax.axvline(t_now, color="0.35", lw=1, ls="--", zorder=3)

for k, a in [(2, 0.16), (1, 0.30)]:
    ax.fill_between(t[t_now:], mu - k * sigma, mu + k * sigma,
                    color="#c1502e", alpha=a, lw=0, zorder=1, label=f"{k} sigma")

ax.plot(t[t_now:], y_true[t_now:], color="0.15", lw=1.0, ls=":", zorder=4,
        label="true future")
ax.plot(t[:n_hist], y_obs, color="0.15", lw=1.4, zorder=4, label="observed")
ax.plot(t[t_now:], mu, color="#c1502e", lw=1.9, zorder=5, label="prediction")

ax.text(t_now - 4, 2.25, "past", ha="right", va="top", color="0.4")
ax.text(t_now + 4, 2.25, "future", ha="left", va="top", color="0.4")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.set_ylim(-2.4, 2.4)
ax.set_xlim(t[0], t[-1])
ax.legend(loc="lower left", frameon=False, ncol=5, fontsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
export_figure(plt.gcf(), os.path.join(fig_path, '00_time_series_Energy.png'), width=20, height=6,resolution=120)
plt.show()