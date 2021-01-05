from matplotlib import pyplot as plt
import geopandas as gpd
import pathlib
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from typing import Union
from rsqsim_api.io.array_operations import read_tiff
from matplotlib.colors import LightSource

coast_shp_fine_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-150k.shp"
coast_shp_coarse_name = "data/coastline/nz-coastlines-and-islands-polygons-topo-1500k.shp"
coast_shp_fine = pathlib.Path(__file__).parent / coast_shp_fine_name
coast_shp_coarse = pathlib.Path(__file__).parent / coast_shp_coarse_name
min_x1 = 800000
min_y1 = 4000000
max_x2 = 2200000
max_y2 = 6400000


def clip_coast_with_trim(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float]):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
    assert all(conditions), "Check coordinates"

    boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=2193)
    coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)
    trimmed_df = gpd.clip(coast_df, boundary)
    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries


def clip_coast(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float], ):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
    assert all(conditions), "Check coordinates"

    coast_df = gpd.GeoDataFrame.from_file(coast_shp_coarse)
    trimmed_df = coast_df.cx[x1:x2, y1:y2]

    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries


def plot_coast(ax: plt.Axes, clip_boundary: list = None, colors: str = "0.5", linewidth: int = 0.3,
               trim_polygons=True):
    if clip_boundary is None:
        x1, y1, x2, y2 = [min_x1, min_y1, max_x2, max_y2]
    else:
        assert isinstance(clip_boundary, list)
        assert len(clip_boundary) == 4
        x1, y1, x2, y2 = clip_boundary
    if trim_polygons:
        clipped_gs = clip_coast_with_trim(x1, y1, x2, y2)
    else:
        clipped_gs = clip_coast(x1, y1, x2, y2)
    for poly in clipped_gs.geometry:
        x, y = [np.array(a) for a in poly.exterior.xy]
        ax.plot(x, y, colors, linewidth=linewidth)

    return x1, y1, x2, y2


def plot_hillshade(ax, alpha, clip_boundary: list = None):
    hillshade_name = "data/bathymetry/niwa_combined_10000.tif"
    hillshade = pathlib.Path(__file__).parent / hillshade_name
    x, y, z = read_tiff(hillshade)

    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.hillshade(z), cmap='Greys', extent=[min(x), max(x), min(y), max(y)], alpha=alpha)
