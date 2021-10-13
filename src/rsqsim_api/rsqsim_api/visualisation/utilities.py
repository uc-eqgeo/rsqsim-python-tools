from matplotlib import pyplot as plt
import geopandas as gpd
import pathlib
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from typing import Union
from rsqsim_api.io.array_operations import read_tiff
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib import pyplot as plt

import rioxarray

coast_shp_fine_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-150k.shp"
coast_shp_coarse_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-1500k.shp"
coast_shp_fine = pathlib.Path(__file__).parent / coast_shp_fine_name
coast_shp_coarse = pathlib.Path(__file__).parent / coast_shp_coarse_name

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
                         wgs: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """

    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    if wgs:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=4326)
    else:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=2193)
    coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)
    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = gpd.clip(coast_df, boundary)
    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries


def clip_coast(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float],
               wgs: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)
    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = coast_df.cx[x1:x2, y1:y2]

    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
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






def plot_coast(ax: plt.Axes, clip_boundary: list = None, edgecolor: str = "0.5", facecolor: str = "none", linewidth: int = 0.3,
               trim_polygons=True, wgs: bool = False):
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
        clipped_gs = clip_coast_with_trim(x1, y1, x2, y2, wgs=wgs)
    else:
        clipped_gs = clip_coast(x1, y1, x2, y2, wgs=wgs)

    clipped_gs.plot(ax=ax, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)

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






