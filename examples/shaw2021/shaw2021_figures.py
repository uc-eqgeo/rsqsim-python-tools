import pandas as pd

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.visualisation.utilities import plot_coast, format_label_text_wgs
import os

import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable




x1 = 160.
x2 = 185.
y1 = -51.
y2 = -33.001

aspect = 1/np.cos(np.radians(np.mean([y1, y2])))

params = {'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.family': 'serif',
          'font.serif': "CMU Serif",
          'font.size': 12,
          }

mpl.rcParams.update(params)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
# Get directory where script is run
script_dir = os.path.dirname(__file__)

fault_dir = "../../data/shaw2021"
catalogue_dir = fault_dir

seconds_per_year = 31557600.0

# fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
#                                                      os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
#                                                      transform_from_utm=True)
whole_catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(catalogue_dir, "nzcatalogue.csv"),
                                                                list_file_directory=catalogue_dir, list_file_prefix="nzcatalogue",
                                                                reproject=[32759, 4326])

# Fig 1:

window_length = 80. * seconds_per_year
for i in range(100):
    plt.close("all")
    tmin = 1.e12 + i * window_length
    tmax = tmin + window_length

    windowed_df = whole_catalogue.filter_df(min_t0=tmin, max_t0=tmax)

    fig, ax = plt.subplots()
    cmap = ax.scatter(windowed_df.x, windowed_df.y, marker='o', s=windowed_df.area/1.e7, c=windowed_df.mw,
                     cmap="Reds", vmin=4, vmax=9.3)

    plot_coast(ax=ax, edgecolor="k", wgs=True, coarse=True)
    ax.set_xlim(([x1, x2]))
    ax.set_ylim(([y1, y2]))
    format_label_text_wgs(ax, xspacing=4, yspacing=2)
    cbar_ticks = np.arange(4, 9.3, 0.5)
    colorbar = plt.colorbar(cmap, ax=ax, ticks=cbar_ticks)
    colorbar.ax.set_title("$M_W$")

    plt.savefig(f"windows80/window{i}_80year.png")


window_length = 500. * seconds_per_year
for i in range(100):
    plt.close("all")
    tmin = 1.e12 + i * window_length
    tmax = tmin + window_length

    windowed_df = whole_catalogue.filter_df(min_t0=tmin, max_t0=tmax)

    fig, ax = plt.subplots()
    cmap = ax.scatter(windowed_df.x, windowed_df.y, marker='o', s=windowed_df.area/1.e7,
                      c=(windowed_df.t0 - tmin) / seconds_per_year,
                      cmap="jet", vmin=0, vmax=window_length / seconds_per_year - 1.)

    plot_coast(ax=ax, edgecolor="k", wgs=True, coarse=True)
    ax.set_xlim(([x1, x2]))
    ax.set_ylim(([y1, y2]))
    format_label_text_wgs(ax, xspacing=4, yspacing=2)
    colorbar = plt.colorbar(cmap, ax=ax)
    colorbar.ax.set_title("Time (years)")

    plt.savefig(f"windows_time500/window{i}_500year.png")


seismicity = pd.read_csv("../../data/shaw2021/databases/earthquakesNZ_shallowM4.csv")
since_1940 = seismicity[(seismicity.origintime > "1940-01-01T00:00:00") & (seismicity.magnitude >= 4.0)]

fig, ax = plt.subplots()
cmap = ax.scatter(since_1940.longitude, since_1940.latitude, marker='o', s=10**since_1940.magnitude / 2.e5,
                  c=since_1940.magnitude,
                  cmap="Blues", vmin=4, vmax=7.9, linewidth=0, alpha=0.8)

plot_coast(ax=ax, edgecolor="k", wgs=True, coarse=True)
ax.set_xlim(([x1, x2]))
ax.set_ylim(([y1, y2]))
format_label_text_wgs(ax, xspacing=4, yspacing=2)
cbar_ticks = np.arange(4, 7.9, 0.5)
colorbar = plt.colorbar(cmap, ax=ax, ticks=cbar_ticks)
colorbar.ax.set_title("$M_W$")
ax.set_title('Observed Catalog; Shallow<30km; 1940-2020')

plt.savefig(f"recent_shallow_seismicity.png")


window_length = 1500. * seconds_per_year
for i in range(100):
    plt.close("all")
    tmin = 1.e12 + i * window_length
    tmax = tmin + window_length

    windowed_df = whole_catalogue.filter_df(min_t0=tmin, max_t0=tmax)

    fig, ax = plt.subplots()
    cmap = ax.scatter((windowed_df.t0 - tmin) / seconds_per_year, windowed_df.y, marker='o', s=10,
                      c=-1 * windowed_df.z / 1000.,
                      cmap="cividis", vmin=0, vmax=23.)


    ax.set_xlim(([-100, 1600]))
    ax.set_ylim(([y1, y2]))
    format_label_text_wgs(ax, xspacing=4, yspacing=2, y_only=True)

    colorbar = plt.colorbar(cmap, ax=ax)
    colorbar.ax.set_title("Depth (km)")

    plt.savefig(f"latitude_time/window{i}_1500year.png")

