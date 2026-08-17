import os
import numpy as np
from matplotlib import pyplot as plt

from cpsmehelper import get_colors, export_figure
cpsme_colors = get_colors()
plt.style.use(os.path.join(os.getcwd(), 'cps_presentation.mplstyle'))
fig_path = os.path.join(os.getcwd(), 'teaching_material', 'graphics')


"""
DAX
"""

import yfinance as yf

dax = yf.download("^GDAXI", period="10y", interval="1d", auto_adjust=False)["Close"]

dax.plot(figsize=(12, 4.5), lw=1, color="0.2",
         title="DAX, last 10 years", ylabel="index points")
plt.tight_layout()
export_figure(plt.gcf(), os.path.join(fig_path, '00_time_series_DAX.png'), width=14, height=8,resolution=120)
plt.show()


"""
Energy generation in Germany (Energy Charts API from Fraunhofer ISE)
"""
import requests
import pandas as pd

r = requests.get(
    "https://api.energy-charts.info/public_power",
    params={"country": "de", "start": "2026-08-01", "end": "2026-08-15"},
).json()

df = pd.DataFrame(
    {p["name"]: p["data"] for p in r["production_types"]},
    index=pd.to_datetime(r["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin"),
)

sources = ["Solar", "Wind onshore", "Wind offshore", "Fossil gas"]
gen = df[sources].dropna() / 1000  # GW

gen.plot(figsize=(13, 5), lw=1.2,
         title="Public net electricity generation, Germany", ylabel="GW")
plt.legend(frameon=True, facecolor='white', edgecolor='0.85')
plt.tight_layout()
export_figure(plt.gcf(), os.path.join(fig_path, '00_time_series_Energy.png'), width=20, height=6,resolution=120)
plt.show()