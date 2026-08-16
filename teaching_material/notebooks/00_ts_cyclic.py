import os
import csv
import urllib.request
import numpy as np
from matplotlib import pyplot as plt

from cpsmehelper import get_colors, export_figure
cpsme_colors = get_colors()
plt.style.use(os.path.join(os.getcwd(), 'cps_presentation.mplstyle'))
fig_path = os.path.join(os.getcwd(), 'teaching_material', 'graphics')

# NOAA Oceanic Niño Index (ONI)

"""
This is the operational index NOAA uses to define El Niño and La Niña. It is the difference between a 
three month running average of sea surface temperature over the equatorial Pacific from 120 West to 
170 West and the long term average for those same three months, with El Niño conditions declared at 
+0.5 or higher and La Niña at 0.5 or lower. Monthly values run continuously from 1950 to the present, 
it is plain ASCII, four columns, no API key, and as a US Government work it is public domain.

https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""


import io
import urllib.request
 
import matplotlib.pyplot as plt
import pandas as pd
 
URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
 
# The 12 overlapping 3-month seasons; the i-th season of year YR is centered
# on month i+1 of YR (DJF is centered on Jan, NDJ on Dec).
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
 
EL_NINO = 0.5   # degC, CPC threshold
LA_NINA = -0.5
 
 
def load_oni(url=URL, cache="oni.ascii.txt"):
    """Fetch the ASCII table and return a monthly indexed DataFrame."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        with open(cache, "w", encoding="utf-8") as fh:
            fh.write(raw)
    except Exception as err:            # offline fallback
        print(f"download failed ({err}), using cached copy")
        with open(cache, encoding="utf-8") as fh:
            raw = fh.read()
 
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+")
    df.columns = [c.strip().upper() for c in df.columns]
    month = df["SEAS"].map({s: i + 1 for i, s in enumerate(SEASONS)})
    df["date"] = pd.to_datetime(
        dict(year=df["YR"].astype(int), month=month, day=15)
    )
    return df.set_index("date")[["TOTAL", "ANOM"]].sort_index()
 
 
def plot_oni(df, ax=None):
    """Classic red/blue ONI time series plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.5))
 
    t = df.index.to_numpy()
    x = df["ANOM"].to_numpy(dtype=float)
 
    ax.fill_between(t, x, EL_NINO, where=(x >= EL_NINO), interpolate=True,
                    color="#b2182b", alpha=0.85, label="El Nino (ONI above +0.5)")
    ax.fill_between(t, x, LA_NINA, where=(x <= LA_NINA), interpolate=True,
                    color="#2166ac", alpha=0.85, label="La Nina (ONI below 0.5)")
    ax.plot(t, x, color="0.2", lw=0.8)
 
    for y, ls in [(0.0, "-"), (EL_NINO, ":"), (LA_NINA, ":")]:
        ax.axhline(y, color="0.4", lw=0.8, ls=ls)
 
    ax.set_xlabel("year")
    ax.set_ylabel("ONI anomaly [degC]")
    ax.set_title("Oceanic Nino Index (NOAA CPC, ERSST, Nino 3.4 region)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.margins(x=0)
    return ax
 
 
def strongest_events(df, n=5):
    """Peak ONI value per calendar year, sorted descending."""
    peaks = df["ANOM"].groupby(df.index.year).max()
    return peaks.sort_values(ascending=False).head(n)
 
 
if __name__ == "__main__":
    oni = load_oni()
    print(oni.tail())
    print("\nstrongest El Nino years by peak ONI:")
    print(strongest_events(oni).round(2).to_string())
 
    plot_oni(oni)
    plt.tight_layout()
    export_figure(plt.gcf(), os.path.join(fig_path, '00_time_series_cyclic.png'), width=14, height=8,resolution=120)
    plt.show()
 
