"""
Visualisation utilities for plotting New Zealand map backgrounds,
coastlines, hillshading, and GIS vector layers.

Provides functions to clip and overlay the NZ coastline shapefile,
render hillshade rasters, plot GIS line and polygon features (roads,
rivers, lakes, regional boundaries, Hikurangi boundary), format
WGS84 axis tick labels, and compose composite map backgrounds via
``plot_background``.
"""
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
    Clip the NZ coastline to a bounding box and trim polygon boundaries.

    Uses ``geopandas.clip`` to trim coastline polygons to the bounding
    box, then returns the individual polygon geometries.

    Parameters
    ----------
    x1 : int or float
        Bottom-left easting (NZTM metres, or longitude if ``wgs=True``).
    y1 : int or float
        Bottom-left northing (NZTM metres, or latitude if ``wgs=True``).
    x2 : int or float
        Top-right easting.
    y2 : int or float
        Top-right northing.
    wgs : bool, optional
        If ``True``, interpret coordinates as WGS84 longitude/latitude.
        Defaults to ``False``.
    coarse : bool, optional
        If ``True``, use the coarse Natural Earth coastline.
        Defaults to ``False``.
    fine : bool, optional
        If ``True``, use the fine 150 k topo coastline.
        Defaults to ``False``.

    Returns
    -------
    geopandas.GeoSeries
        Clipped coastline polygons in NZTM (EPSG:2193).
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
    Clip the NZ coastline to a bounding box using coordinate indexing.

    Uses the ``cx`` indexer rather than ``geopandas.clip``, so polygons
    that overlap the boundary are kept whole rather than trimmed.

    Parameters
    ----------
    x1 : int or float
        Bottom-left easting (NZTM metres, or longitude if ``wgs=True``).
    y1 : int or float
        Bottom-left northing (NZTM metres, or latitude if ``wgs=True``).
    x2 : int or float
        Top-right easting.
    y2 : int or float
        Top-right northing.
    wgs : bool, optional
        If ``True``, interpret coordinates as WGS84 longitude/latitude.
        Defaults to ``False``.
    coarse : bool, optional
        If ``True``, use the coarse Natural Earth coastline.
        Defaults to ``False``.
    fine : bool, optional
        If ``True``, use the fine 150 k topo coastline.
        Defaults to ``False``.

    Returns
    -------
    geopandas.GeoSeries
        Coastline polygons intersecting the bounding box, in NZTM.
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
    """
    Plot line features from a GIS vector file onto a matplotlib axis.

    Parameters
    ----------
    gis_file : str or pathlib.Path
        Path to the GIS file (shapefile, GeoPackage, etc.).
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    color : str
        Line colour.
    linewidth : int, optional
        Line width in points.  Defaults to 0.3.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` bounding box for clipping.
        If ``None``, all features are plotted.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    """
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(color=color, ax=ax, linewidth=linewidth, linestyle=linestyle)

def plot_gis_polygons(gis_file: Union[str, pathlib.Path], ax: plt.Axes, edgecolor: str, linewidth: int = 0.3, clip_bounds: list = None,
                      linestyle: str = "-", facecolor="none"):
    """
    Plot polygon features from a GIS vector file onto a matplotlib axis.

    Parameters
    ----------
    gis_file : str or pathlib.Path
        Path to the GIS file (shapefile, GeoPackage, etc.).
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    edgecolor : str
        Polygon edge colour.
    linewidth : int, optional
        Edge line width in points.  Defaults to 0.3.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` bounding box for clipping.
        If ``None``, all features are plotted.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    facecolor : str, optional
        Polygon fill colour.  Defaults to ``"none"`` (transparent).
    """
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(edgecolor=edgecolor, ax=ax, linewidth=linewidth, linestyle=linestyle, facecolor=facecolor)


def plot_highway_lines(ax: plt.Axes, color: str = "r", linewidth: int = 1., clip_bounds: list = None,
                  linestyle: str = "-"):
    """
    Overlay NZ state highway lines on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    color : str, optional
        Line colour.  Defaults to ``"r"``.
    linewidth : int, optional
        Line width in points.  Defaults to 1.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    """
    plot_gis_lines(roads, ax=ax, color=color, linewidth=linewidth, clip_bounds=clip_bounds, linestyle=linestyle)


def plot_river_lines(ax: plt.Axes, color: str = "b", linewidth: int = 0.3, clip_bounds: list = None,
                linestyle: str = "-"):
    """
    Overlay NZ major river lines on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    color : str, optional
        Line colour.  Defaults to ``"b"``.
    linewidth : int, optional
        Line width in points.  Defaults to 0.3.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    """
    plot_gis_lines(rivers, ax=ax, color=color, linewidth=linewidth, clip_bounds=clip_bounds, linestyle=linestyle)


def plot_boundary_polygons(ax: plt.Axes, edgecolor: str = "k", linewidth: int = 0.3, clip_bounds: list = None,
                 linestyle: str = "--", facecolor: str = "none"):
    """
    Overlay NZ regional boundary polygons on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    edgecolor : str, optional
        Edge colour.  Defaults to ``"k"``.
    linewidth : int, optional
        Edge line width in points.  Defaults to 0.3.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"--"``.
    facecolor : str, optional
        Fill colour.  Defaults to ``"none"`` (transparent).
    """
    plot_gis_polygons(regions, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)


def plot_lake_polygons(ax: plt.Axes, edgecolor: str = "b", linewidth: int = 0.3, clip_bounds: list = None,
                 linestyle: str = "-", facecolor: str = "b"):
    """
    Overlay NZ lake polygons on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    edgecolor : str, optional
        Edge colour.  Defaults to ``"b"``.
    linewidth : int, optional
        Edge line width in points.  Defaults to 0.3.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    facecolor : str, optional
        Fill colour.  Defaults to ``"b"`` (blue).
    """
    plot_gis_polygons(lakes, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)


def plot_hk_boundary(ax: plt.Axes, edgecolor: str = "r", linewidth: int = 0.1, clip_bounds: list = None,
                     linestyle: str = "-", facecolor: str = "0.8"):
    """
    Overlay the Hikurangi subduction zone clipping-area polygon.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    edgecolor : str, optional
        Edge colour.  Defaults to ``"r"``.
    linewidth : int, optional
        Edge line width in points.  Defaults to 0.1.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.
    linestyle : str, optional
        Matplotlib line-style string.  Defaults to ``"-"``.
    facecolor : str, optional
        Fill colour.  Defaults to ``"0.8"`` (light grey).
    """
    plot_gis_polygons(hk_boundary, ax=ax, edgecolor=edgecolor, linewidth=linewidth, clip_bounds=clip_bounds,
                      linestyle=linestyle, facecolor=facecolor)






def plot_coast(ax: plt.Axes, clip_boundary: list = None, edgecolor: str = "0.5", facecolor: str = 'none', linewidth: int = 0.3,
               trim_polygons=True, wgs: bool = False, coarse: bool = False, fine: bool = False):
    """
    Overlay the NZ coastline on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    clip_boundary : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` clipping bounds.  Defaults to
        the full NZ extent.
    edgecolor : str, optional
        Coastline edge colour.  Defaults to ``"0.5"`` (mid-grey).
    facecolor : str, optional
        Land fill colour.  Defaults to ``"none"`` (transparent).
    linewidth : int, optional
        Edge line width.  Defaults to 0.3.
    trim_polygons : bool, optional
        If ``True`` (default), use :func:`clip_coast_with_trim` to trim
        polygons to the bounding box; otherwise use :func:`clip_coast`.
    wgs : bool, optional
        If ``True``, use WGS84 (lon/lat) coordinates.
        Defaults to ``False``.
    coarse : bool, optional
        If ``True``, use the coarse Natural Earth coastline.
        Defaults to ``False``.
    fine : bool, optional
        If ``True``, use the fine 150 k topo coastline.
        Defaults to ``False``.

    Returns
    -------
    x1, y1, x2, y2 : float
        The clipping bounds actually used.
    """
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
    """
    Overlay a hillshaded bathymetry/topography raster on a matplotlib axis.

    Uses a combined 10 000-m resolution NIWA raster.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    alpha : float, optional
        Transparency of the hillshade overlay (0–1).  Defaults to 0.3.
    vertical_exaggeration : float, optional
        Vertical exaggeration for the light-source shading.
        Defaults to 0.01.
    cmap : LinearSegmentedColormap or None, optional
        Colourmap for the hillshade.  Defaults to ``plt.cm.gist_earth``.
    vmin : float, optional
        Minimum value for the colour scale.  Defaults to -10 000.
    vmax : float, optional
        Maximum value for the colour scale.  Defaults to 10 000.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` bounds for clipping the raster.
    """
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
    """
    Overlay a hillshaded NIWA NZTM raster on a matplotlib axis.

    Uses the high-resolution ``niwa_nztm.tif`` raster.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    alpha : float, optional
        Transparency of the hillshade overlay (0–1).  Defaults to 0.3.
    vertical_exaggeration : float, optional
        Vertical exaggeration for the light-source shading.
        Defaults to 0.01.
    clip_bounds : list or None, optional
        ``[x_min, y_min, x_max, y_max]`` bounds for clipping the raster.
    cmap : LinearSegmentedColormap or None, optional
        Colourmap for the hillshade.  Defaults to ``plt.cm.terrain``.
    vmin : float, optional
        Minimum value for the colour scale.  Defaults to -10 000.
    vmax : float, optional
        Maximum value for the colour scale.  Defaults to 10 000.
    """
    hillshade_name = "data/bathymetry/niwa_nztm.tif"
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
    Format axis tick labels as degree notation for WGS84 plots.

    Converts raw longitude/latitude tick values to degree strings,
    handling longitudes > 180° by converting to negative (west) values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose tick labels to format.
    xspacing : int, optional
        Longitude tick spacing in degrees.  Defaults to 5.
    yspacing : int, optional
        Latitude tick spacing in degrees.  Defaults to 5.
    y_only : bool, optional
        If ``True``, only format the y (latitude) axis.
        Defaults to ``False``.
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
                    slider_axis: bool = False, aotearoa: bool = True):
        """
        Compose a NZ map background figure with optional GIS overlays.

        Creates a ``subplot_mosaic`` figure with a main map panel and
        optional colourbar and slider axes.  Overlays the coastline,
        optional hillshade, lakes, rivers, highways, regional boundaries,
        and/or the Hikurangi clipping area.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.  Defaults to
            ``(6.4, 4.8)``.
        hillshading_intensity : float, optional
            Hillshade alpha (0–1).  If 0 (default), no hillshade is drawn.
        bounds : tuple, optional
            ``(x_min, y_min, x_max, y_max)`` map bounds.
        plot_rivers : bool, optional
            If ``True`` (default), overlay river lines.
        plot_lakes : bool, optional
            If ``True`` (default), overlay lake polygons.
        hillshade_fine : bool, optional
            If ``True``, use the high-resolution NIWA hillshade raster.
            Defaults to ``False``.
        plot_highways : bool, optional
            If ``True`` (default), overlay state highway lines.
        plot_boundaries : bool, optional
            If ``True``, overlay regional boundary polygons.
            Defaults to ``False``.
        subplots : tuple or None, optional
            ``(fig, ax)`` to draw onto.  A new figure is created if
            ``None``.
        pickle_name : str or None, optional
            If provided, pickle the ``(fig, ax)`` tuple to this path.
        hillshade_cmap : LinearSegmentedColormap, optional
            Colourmap for the hillshade.  Defaults to ``cm.terrain``.
        plot_edge_label : bool, optional
            If ``True`` (default), show axis tick labels.
        plot_hk : bool, optional
            If ``True``, overlay the Hikurangi clipping-area polygon.
            Defaults to ``False``.
        plot_fault_outlines : bool, optional
            Reserved for future use.  Defaults to ``True``.
        wgs : bool, optional
            If ``True``, use WGS84 coordinates.  Defaults to ``False``.
        land_color : str, optional
            Land fill colour.  Defaults to ``"antiquewhite"``.
        plot_sub_cbar : bool, optional
            If ``True``, add a subduction slip colourbar panel.
            Defaults to ``False``.
        sub_slip_max : float, optional
            Maximum subduction slip (m) for the colourbar.
            Defaults to 20.
        plot_crust_cbar : bool, optional
            If ``True``, add a crustal slip colourbar panel.
            Defaults to ``False``.
        crust_slip_max : float, optional
            Maximum crustal slip (m) for the colourbar.  Defaults to 10.
        subduction_cmap : LinearSegmentedColormap, optional
            Colourmap for the subduction colourbar.  Defaults to
            ``cm.plasma``.
        crust_cmap : LinearSegmentedColormap, optional
            Colourmap for the crustal colourbar.  Defaults to
            ``cm.viridis``.
        slider_axis : bool, optional
            If ``True``, add a slider and year-label axes row.
            Defaults to ``False``.
        aotearoa : bool, optional
            If ``True`` (default), draw the NZ coastline and GIS overlays.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The composed figure.
        ax : dict or matplotlib.axes.Axes
            Axis dict (from ``subplot_mosaic``) or the provided ``ax``.
        """
        if subplots is not None:
            fig, ax = subplots
            main_ax = ax
        else:
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


            if slider_axis:
                fig, ax = plt.subplot_mosaic(mosaic, gridspec_kw={"height_ratios": height_ratio,
                                                                  "width_ratios": width_ratio},
                                             layout="constrained")
                year_ax = ax["year"]
            else:
                fig, ax = plt.subplot_mosaic(mosaic,
                                             gridspec_kw={"width_ratios": width_ratio},
                                             layout="constrained")
            fig.set_size_inches(figsize)
            main_ax = ax["main_figure"]


        plot_bounds = list(bounds)
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

        if pickle_name is not None:
            with open(pickle_name, "wb") as pfile:
                pickle.dump((fig, ax), pfile)

        return fig, ax



