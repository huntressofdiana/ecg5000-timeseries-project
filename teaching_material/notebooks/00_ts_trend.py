import os
import csv
import urllib.request
import numpy as np
from matplotlib import pyplot as plt

from cpsmehelper import get_colors, export_figure
cpsme_colors = get_colors()
plt.style.use(os.path.join(os.getcwd(), 'cps_presentation.mplstyle'))
fig_path = os.path.join(os.getcwd(), 'teaching_material', 'graphics')


# Public dataset repository (GitHub): datasets/global-temp
# This contains annual global land-ocean temperature anomalies in deg C.
repo_url = (
    'https://raw.githubusercontent.com/datasets/global-temp/'
    'master/data/annual.csv'
)
data_dir = os.path.join(
    os.getcwd(), 'teaching_material', 'datasets', 'global_temp'
)
os.makedirs(data_dir, exist_ok=True)
local_csv = os.path.join(data_dir, 'annual_global_land_ocean_temp.csv')

if not os.path.exists(local_csv):
    urllib.request.urlretrieve(repo_url, local_csv)

years = []
anomalies = []
with open(local_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Source'] == 'GISTEMP':
            years.append(int(row['Year']))
            anomalies.append(float(row['Mean']))

years = np.array(years)
anomalies = np.array(anomalies)

# Focus on recent decades for clear warming visualization.
mask = years >= 1970
years_recent = years[mask]
anomalies_recent = anomalies[mask]

# Linear trend over the selected decades.
coef = np.polyfit(years_recent, anomalies_recent, 1)
trend = np.polyval(coef, years_recent)

fig = plt.figure()
plt.plot(
    years_recent,
    anomalies_recent,
    marker='o',
    markersize=3,
    linewidth=1.5,
    label='Annual anomaly (NASA GISTEMP land-ocean)',
)
plt.plot(
    years_recent,
    trend,
    linewidth=2.2,
    label='Linear trend since 1970',
)
plt.axhline(0.0, color='black', linewidth=1.0, alpha=0.35)
plt.title('Global Land-Ocean Temperature Anomaly (Recent Decades)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly [deg C]')
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

export_figure(fig, os.path.join(fig_path, '00_time_series_trend.png'), width=14, height=8,resolution=120)
plt.show()


