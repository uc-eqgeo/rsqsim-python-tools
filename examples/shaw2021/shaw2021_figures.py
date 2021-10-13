from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.visualisation.utilities import plot_coast
import os
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

x1 = 160.
x2 = 185.
y1 = -51.
y2 = -33.

params = {'axes.labelsize': 18,
          'axes.titlesize': 18,
          #            'text.fontsize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'font.family': 'serif',
          'font.serif': "CMU Serif",
          'font.size': 18,
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

fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
                                                     os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
                                                     transform_from_utm=True)
whole_catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(catalogue_dir, "eqs..out"),
                                                                list_file_directory=catalogue_dir, list_file_prefix="")

# Fig 1:
plt.close("all")
fig, ax = plt.subplots()

plot_coast(ax=ax, edgecolor="k")
ax.set_aspect(0.5)
# ax.set_xlim(([x1, x2]))
# ax.set_ylim(([y1, y2]))

plt.savefig("shaw_f1.png")
