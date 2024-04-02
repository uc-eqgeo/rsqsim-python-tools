from matplotlib import pyplot as plt
import geopandas as gpd
import pathlib
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from typing import Union
from rsqsim_api.io.array_operations import read_tiff
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib import cm, colors
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import rioxarray

coast_shp_fine_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-150k.shp"
coast_shp_coarse_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-1500k.shp"
coast_nat_earth_name = "data/coastline/natural_earth_nztm.shp"
coast_shp_fine = pathlib.Path(__file__).parent / coast_shp_fine_name
coast_shp_coarse = pathlib.Path(__file__).parent / coast_shp_coarse_name
coast_nat_earth = pathlib.Path(__file__).parent / coast_nat_earth_name

roads = pathlib.Path(__file__).parent / "data/other_lines/state_highways.shp"
lakes = pathlib.Path(__file__).parent / "data/other_lines/nz-lake-polygons-topo-1250k.shp"
rivers = pathlib.Path(__file__).parent / "data/other_lines/nz-major-rivers.shp"
regions = pathlib.Path(__file__).parent / "data/other_lines/nz-major-rivers.shp"
hk_boundary = pathlib.Path(__file__).parent / "data/faults/hk_clipping_area.gpkg"


niwa = ""


min_x1 = 800000
min_y1 = 4000000
max_x2 = 3200000
max_y2 = 7500000

min_x1_wgs = 160.
max_x2_wgs = 185.
min_y1_wgs = -51.
max_y2_wgs = -33.


def clip_coast_with_trim(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float],
                         wgs: bool = False, coarse: bool = False, fine: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    assert not all([coarse, fine])
    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    if wgs:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=4326)
    else:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=2193)

    if coarse:
        coast_df = gpd.GeoDataFrame.from_file(coast_nat_earth)
    elif fine:
        coast_df = gpd.GeoDataFrame.from_file(coast_shp_fine)
    else:
        coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)

    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = gpd.clip(coast_df, boundary)
    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item.geoms)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries


def clip_coast(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float],
               wgs: bool = False, coarse: bool = False, fine: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    assert not all([coarse, fine])
    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    if coarse:
        coast_df = gpd.GeoDataFrame.from_file(coast_nat_earth)
    elif fine:
        coast_df = gpd.GeoDataFrame.from_file(coast_shp_fine)
    else:
        coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)

    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = coast_df.cx[x1:x2, y1:y2]

    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item.geoms)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries

def plot_gis_lines(gis_file: Union[str, pathlib.Path], ax: plt.Axes, color: str, linewidth: int = 0.3, clip_bounds: list = None,
                   linestyle: str = "-"):
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(color=color, ax=ax, linewidth=linewidth, linestyle=linestyle)

def plot_gis_polygons(gis_file: Union[str, pathlib.Path], ax: plt.Axes, edgecolor: str, linewidth: int = 0.3, clip_bounds: list = None,
                      linestyle: str = "-", facecolor="none"):
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(edgecolor=edgecolor, ax=ax, linewidth=linewidth, linestyle=linestyle, facecolor=facecolor)


def plot_highway_lines(ax: plt.Axes, color: str = "r", linewidth: int = 1., clip_bounds: list = None,
                  linestyle: str = "-"):
    plot_gis_lines(roads, ax=ax, color=color, linewidth=linewidth, clip_bounds=clip_bounds, linestyle=linestyle)


def plot_river_lines(ax: plt.Axes, color: str = "b", linewidth: int = 0.3, clip_bounds: list = None,
                linestyle: str = "-"):
    plot_gis_lines(rivers, ax=ax, color=color, linewidth=linewidth, clip_bounds=clip_bounds, linestyle=linestyle)


def plot_boundary_polygons(ax: plt.Axes, edgecolor: str = "k", linewidth: int = 0.3, clip_bounds: list = None,
                 linestyle: str = "--", facecolor: str = "none"):
    plot_gis_polygons(regions, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)


def plot_lake_polygons(ax: plt.Axes, edgecolor: str = "b", linewidth: int = 0.3, clip_bounds: list = None,
                 linestyle: str = "-", facecolor: str = "b"):
    plot_gis_polygons(lakes, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)


def plot_hk_boundary(ax: plt.Axes, edgecolor: str = "r", linewidth: int = 0.1, clip_bounds: list = None,
                     linestyle: str = "-", facecolor: str = "0.8"):
    plot_gis_polygons(hk_boundary, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)






def plot_coast(ax: plt.Axes, clip_boundary: list = None, edgecolor: str = "0.5", facecolor: str = 'none', linewidth: int = 0.3,
               trim_polygons=True, wgs: bool = False, coarse: bool = False, fine: bool = False):
    assert not all([coarse, fine])
    if clip_boundary is None:
        if wgs:
            x1, y1, x2, y2 = [min_x1_wgs, min_y1_wgs, max_x2_wgs, max_y2_wgs]
        else:
            x1, y1, x2, y2 = [min_x1, min_y1, max_x2, max_y2]
    else:
        assert isinstance(clip_boundary, list)
        assert len(clip_boundary) == 4
        x1, y1, x2, y2 = clip_boundary
    if trim_polygons:
        clipped_gs = clip_coast_with_trim(x1, y1, x2, y2, wgs=wgs, coarse=coarse)
    else:
        clipped_gs = clip_coast(x1, y1, x2, y2, wgs=wgs, coarse=coarse)

    clipped_gs.plot(ax=ax, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
    if wgs:
        aspect = 1/np.cos(np.radians(np.mean([y1, y2])))
        ax.set_aspect(aspect)
    return x1, y1, x2, y2


def plot_hillshade(ax, alpha: float = 0.3, vertical_exaggeration: float = 0.01, cmap: LinearSegmentedColormap = None,
                   vmin: float = -10000., vmax: float = 10000, clip_bounds: list = None):
    hillshade_name = "data/bathymetry/niwa_combined_10000.tif"
    hillshade = pathlib.Path(__file__).parent / hillshade_name
    xds = rioxarray.open_rasterio(hillshade)
    clipped = xds.rio.clip_box(*clip_bounds)
    xds.close()
    z = np.array(clipped.data)
    z = np.nan_to_num(z)[0]
    x = clipped.x
    y = clipped.y
    if cmap is not None:
        terrain = cmap
    else:
        terrain = plt.cm.gist_earth
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.shade(z, blend_mode="overlay", cmap=terrain, vmin=vmin, vmax=vmax, vert_exag=vertical_exaggeration),
              extent=[min(x), max(x), min(y), max(y)], alpha=alpha)
    clipped.close()


def plot_hillshade_niwa(ax, alpha: float = 0.3, vertical_exaggeration: float = 0.01, clip_bounds: list = None,
                        cmap: LinearSegmentedColormap = None, vmin: float = -10000., vmax: float = 10000):
    hillshade_name = "data/bathymetry/niwa_nztm.tif"
    hillshade = pathlib.Path(__file__).parent / hillshade_name
    if not os.path.exists(hillshade):
        print('Hi-res NIWA dem not found, using low-res instead')
        hillshade_name = "data/bathymetry/niwa_combined_10000.tif"
        hillshade = pathlib.Path(__file__).parent / hillshade_name
    xds = rioxarray.open_rasterio(hillshade)
    clipped = xds.rio.clip_box(*clip_bounds)
    xds.close()
    z = np.array(clipped.data)
    z = np.nan_to_num(z)[0]
    x = clipped.x
    y = clipped.y
    if cmap is not None:
        terrain = cmap
    else:
        terrain = plt.cm.terrain
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.shade(z, blend_mode="overlay", cmap=terrain, vmin=vmin, vmax=vmax, vert_exag=vertical_exaggeration),
              extent=[min(x), max(x), min(y), max(y)], alpha=alpha)
    clipped.close()


def format_label_text_wgs(ax: plt.Axes, xspacing: int = 5, yspacing: int = 5, y_only: bool = False):
    """
    Plots latlon labels for plots in wgs, like in Shaw et al., 2021
    """


    locx = plticker.MultipleLocator(base=xspacing)  # this locator puts ticks at regular intervals
    locy = plticker.MultipleLocator(base=yspacing)  # this locator puts ticks at regular intervals

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.yaxis.set_major_locator(locy)

    if not y_only:
        ax.xaxis.set_major_locator(locx)
        xlocs = ax.xaxis.get_ticklocs()
        ax.xaxis.set_ticks(xlocs)
        xlabels = ax.xaxis.get_ticklabels()
        for label, loc in zip(xlabels, xlocs):
            if loc <= 180.:
                label.set_text("+" + f"{int(loc)}" + "$^\\circ$")
            else:
                new_value = loc - 360
                label.set_text(f"{int(new_value)}$^\\circ$")
        ax.xaxis.set_ticklabels(xlabels)

    ylocs = ax.yaxis.get_ticklocs()
    ax.yaxis.set_ticks(ylocs)
    ylabels = ax.yaxis.get_ticklabels()
    for label, loc in zip(ylabels, ylocs):
        label.set_text(f"{int(loc)}" + "$^\\circ$")
    ax.yaxis.set_ticklabels(ylabels)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_background(figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                    plot_rivers: bool = True, plot_lakes: bool = True, hillshade_fine: bool = False,
                    plot_highways: bool = True, plot_boundaries: bool = False, subplots=None,
                    pickle_name: str = None, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain,plot_edge_label: bool = True,
                    plot_hk: bool = False, plot_fault_outlines: bool = True, wgs: bool =  False, land_color: str ='antiquewhite',
                    plot_sub_cbar: bool = False, sub_slip_max: float = 20., plot_crust_cbar: bool = False, crust_slip_max: float = 10.,
                    subduction_cmap: colors.LinearSegmentedColormap = cm.plasma, crust_cmap: colors.LinearSegmentedColormap = cm.viridis,
                    slider_axis: bool = False, aotearoa: bool = True, displace: bool = False, disp_cmap: colors.LinearSegmentedColormap = cm.bwr,
                    disp_slip_max: float = 1., cum_slip_max: list = [5., 10.], step_size: list = None, tide: dict = None):

        if subplots is not None:
            fig, ax = subplots
            main_ax = ax
        else:
            if not displace:
                if plot_sub_cbar or plot_crust_cbar:
                    if not all([plot_sub_cbar, plot_crust_cbar]):
                        if not slider_axis:
                            mosaic = [["main_figure", "cbar"]]
                        else:
                            mosaic = [["main_figure", "cbar"],
                                    ["slider", "year"]]
                            height_ratio = [1, 0.1]
                        width_ratio = [1, 0.05]

                    else:
                        if not slider_axis:
                            mosaic = [["main_figure", "sub_cbar", "crust_cbar"]]
                        else:
                            mosaic = [["main_figure", "sub_cbar", "crust_cbar"],
                                    ["slider", "year", "year"]]
                            height_ratio = [1, 0.1]
                        width_ratio = [1, 0.05, 0.05]
                else:
                    if not slider_axis:
                        mosaic = [["main_figure"]]
                    else:
                        mosaic =[["main_figure"],["slider","year","year"]]
                        height_ratio = [1,0.1]
                    width_ratio = [1]

                main_ax_id = ["main_figure"]
                main_ax_titles = ["Slip Distribution"]

            else:
                if not any([plot_sub_cbar, plot_crust_cbar]):
                    print("Displacement plot requires subduction or crustal slip to be plotted")
                    plot_crust_cbar = True
                if not slider_axis:
                    print("Displacement plot will plot year slider axis")
                    slider_axis = True
                if not all([plot_sub_cbar, plot_crust_cbar]):
                    mosaic = [["main_figure", "cbar", "ud1", "ucbar1"],
                            ["slider", "slider", "slider", "year"],
                            ["ud2", "ucbar2", "ud3", "ucbar3"]]
                    height_ratio = [1, 0.1, 1]
                    width_ratio = [1, 0.05, 1, 0.05]
                else:
                    mosaic = [["main_figure", "sub_cbar", "crust_cbar", "ud1", "ucbar1"],
                            ["slider", "slider", "slider", "slider", "year"],
                            ["ud2", "ucbar2", ".", "ud3", "ucbar3"]]
                    height_ratio = [1, 0.1, 1]
                    width_ratio = [1, 0.05, 0.05, 1, 0.05]

                main_ax_id = ["main_figure", "ud1", "ud2", "ud3"]
                main_ax_titles = ["Slip Distribution", "Coseismic Uplift", f"{step_size[1]:.0f} yrs", f"{step_size[2]:.0f} yrs"]
            
            if tide['time'] != 0:
                tg = []
                for i in range(len(mosaic[0])):
                    tg.append('tg')
                mosaic.append(tg)
                height_ratio.append(0.25)

            if slider_axis:
                    fig, ax = plt.subplot_mosaic(mosaic, gridspec_kw={"height_ratios": height_ratio,
                                                                    "width_ratios": width_ratio},
                                                layout="tight")
                    year_ax = ax["year"]
            else:
                fig, ax = plt.subplot_mosaic(mosaic,
                                            gridspec_kw={"width_ratios": width_ratio},
                                            layout="tight")
            fig.set_size_inches(figsize)
            main_ax_list = []
            for id in main_ax_id:
                main_ax_list.append(ax[id])

            if displace:
                ud_cbar = [ax["ucbar1"], ax["ucbar2"], ax["ucbar3"]]


        plot_bounds = list(bounds)

        for main_ax in main_ax_list:
            if aotearoa:
                if hillshading_intensity > 0:
                    plot_coast(main_ax, clip_boundary=plot_bounds, facecolor=land_color)
                    x_lim = main_ax.get_xlim()
                    y_lim = main_ax.get_ylim()
                    if hillshade_fine:
                        plot_hillshade_niwa(main_ax, hillshading_intensity, clip_bounds=plot_bounds, cmap=hillshade_cmap)
                    else:
                        plot_hillshade(main_ax, hillshading_intensity, clip_bounds=plot_bounds, cmap=hillshade_cmap)
                    main_ax.set_xlim(x_lim)
                    main_ax.set_ylim(y_lim)
                else:
                    plot_coast(main_ax, clip_boundary=plot_bounds,wgs=wgs, facecolor=land_color)

                if plot_lakes:
                    plot_lake_polygons(ax=main_ax, clip_bounds=plot_bounds)

                if plot_rivers:
                    plot_river_lines(main_ax, clip_bounds=plot_bounds)

                if plot_highways:
                    plot_highway_lines(main_ax, clip_bounds=plot_bounds)

                if plot_boundaries:
                    plot_boundary_polygons(main_ax, clip_bounds=plot_bounds)

                if plot_hk:
                    plot_hk_boundary(main_ax, clip_bounds=plot_bounds)

                if plot_fault_outlines:
                    pass

            main_ax.set_aspect("equal")
            x_lim = (plot_bounds[0], plot_bounds[2])
            y_lim = (plot_bounds[1], plot_bounds[3])
            main_ax.set_xlim(x_lim)
            main_ax.set_ylim(y_lim)
            main_ax.title.set_text(main_ax_titles[main_ax_list.index(main_ax)])

            if not plot_edge_label:
                main_ax.set_xticks([])
                main_ax.set_yticks([])

        if slider_axis:
            year_ax.set_xlim(0,1)
            year_ax.set_ylim(0,1)
            year_ax.set_axis_off()
        sub_mappable = ScalarMappable(cmap=subduction_cmap)
        sub_mappable.set_clim(vmin=0, vmax=sub_slip_max)
        crust_mappable = ScalarMappable(cmap=crust_cmap)
        crust_mappable.set_clim(vmin=0, vmax=crust_slip_max)
        if plot_sub_cbar:
            if plot_crust_cbar:
                sub_ax = ax["sub_cbar"]
                crust_ax = ax["crust_cbar"]
            else:
                sub_ax = ax["cbar"]
            sub_cbar = fig.colorbar(
                sub_mappable, cax=sub_ax, extend='max')
            sub_cbar.set_label("Subduction slip (m)")

        if plot_crust_cbar:
            if plot_sub_cbar:
                crust_ax = ax["crust_cbar"]
            else:
                crust_ax = ax["cbar"]
            crust_cbar = fig.colorbar(
                crust_mappable, cax=crust_ax, extend='max')
            crust_cbar.set_label("Crustal slip (m)")
        
        if displace:
            slip_max = [disp_slip_max] + cum_slip_max
            ulabel = ["Uplift (m)", "Cumulative Uplift (m)", "Cumulative Uplift (m)"]
            uix = []
            for ix, uax in enumerate(ud_cbar):
                disp_mappable = ScalarMappable(cmap=disp_cmap)
                disp_mappable.set_clim(vmin=-slip_max[ix], vmax=slip_max[ix])
                uix.append(fig.colorbar(disp_mappable, cax=uax, extend='both'))
                uix[ix].set_label(ulabel[ix])

        if pickle_name is not None:
            with open(pickle_name, "wb") as pfile:
                pickle.dump((fig, ax), pfile)
            fig.savefig(os.path.join(os.path.dirname(pickle_name), 'background.png'), format="png", dpi=100)

        return fig, ax

def plot_tide_gauge(subplots, tide, frame_time, start_time, step_size):

    fig, axU1, axTG = subplots
    axU1.scatter(tide['x'], tide['y'], c='black', s=20, zorder=3)
    
    iterations = int((frame_time - start_time) // tide['time'])
    part_time = int(((frame_time - start_time) % tide['time']) / step_size)
    data = tide['data']
    years = (data[:, 1] - start_time)

    for run in range(iterations):
        ix1 = run * tide['entries']
        ix2 = (run + 1) * tide['entries'] + 1
        ix0 = ix1 - 1 if ix1 > 0 else 0
        axTG.plot(years[ix1:ix2] - years[ix1], data[ix1:ix2, 2] - data[ix0, 2], color='gray', alpha=0.5)

    if part_time == 0 and frame_time != start_time:
        iterations -= 1 # Extend red trend to end of full timespan when resetting
        part_time = tide['entries']
    ix1 = iterations * tide['entries']
    ix2 = (iterations * tide['entries']) + part_time + 1
    ix0 = ix1 - 1 if ix1 > 0 else 0

    axTG.plot(years[ix1:ix2] - years[ix1], data[ix1:ix2, 2] - data[ix0, 2], 'o-', color='red')
    axTG.set_xlim([0, tide['time']])
    axTG.set_ylim([-tide['ylim'], tide['ylim']])
    axTG.set_ylabel('Sea Level Change (m)')
    axTG.set_xlabel('Time (years)')
    return

