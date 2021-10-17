from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.visualisation.utilities import plot_coast
import os
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import matplotlib.ticker as plticker

loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals


x1 = 160.
x2 = 185.
y1 = -52.
y2 = -34.

aspect = 1/np.cos(np.radians(np.mean([y1, y2])))

params = {'axes.labelsize': 12,
          'axes.titlesize': 12,
          #            'text.fontsize': 18,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.family': 'serif',
          'font.serif': "Baskerville",
          'font.size': 12,
          }

# matplotlib.rcParams.update(params)
mpl.rcParams.update(params)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
# Get directory where script is run
script_dir = os.path.dirname(__file__)

fault_dir = "../../data/shaw/rundir5091"
catalogue_dir = fault_dir

seconds_per_year = 31557600.0

# fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
#                                                      os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
#                                                      transform_from_utm=True)
whole_catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(catalogue_dir, "eqs..out"),
                                                                list_file_directory=catalogue_dir, list_file_prefix="",
                                                                reproject=[32759, 4326])

# Fig 1:

window_length = 80. * seconds_per_year
for i in range(100):
    plt.close("all")
    tmin = 1.e12 + i * window_length
    tmax = tmin + window_length

    windowed_df = whole_catalogue.filter_df(min_t0=tmin, max_t0=tmax)

    fig, ax = plt.subplots()
    ax.scatter(windowed_df.x, windowed_df.y, marker='o', s=windowed_df.area/1.e7, c=windowed_df.mw,
               cmap="Reds")

    plot_coast(ax=ax, edgecolor="k", wgs=True, coarse=True)
    ax.set_xlim(([x1, x2]))
    ax.set_ylim(([y1, y2]))
    ax.yaxis.set_major_locator(loc)

    # start_time =

    plt.savefig(f"windows80/window{i}_80year.png")
