import geopandas as gpd
import numpy as np
import os
from typing import Union
from netCDF4 import Dataset
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def read_grid(raster_file: str, nan_threshold: Union[float, int] = 100,
              window: Union[str, list, tuple, np.ndarray] = None,
              buffer: Union[float, int] = 100, return_meta: bool = False):
    """
    Read
    :param raster_file: file to read
    :param nan_threshold: numbers with an absolute value greater than this are set to NaN
    :param window:
    :param buffer:
    :return:
    """

    assert os.path.exists(raster_file)
    raster = rasterio.open(raster_file)
    no_data = raster.nodata


    if window is not None:
        if isinstance(window, str):
            assert os.path.exists(window)
            shp = gpd.GeoDataFrame.from_file(window)
            x1, y1, x2, y2 = np.array(shp.bounds)[0]

        else:
            assert len(window) == 4
            x1, y1, x2, y2 = window[:]

        assert all([x2 > x1, y2 > y1])
        buffer_value = 0 if buffer is None else buffer
        if x1 < raster.bounds.left:
            x1 = np.round(raster.bounds.left + 1)

        lowest = min([raster.bounds.bottom, raster.bounds.top])
        if y1 < lowest:
            y1 = np.round(lowest + -1)

        if raster.bounds.top < raster.bounds.bottom:
            window_object = rasterio.windows.from_bounds(x1 - buffer_value, y2 + buffer_value,
                                                         x2 + buffer_value, y1 - buffer_value,
                                                         transform=raster.transform)
        else:
            window_object = rasterio.windows.from_bounds(x1 - buffer_value, y1 - buffer_value,
                                                         x2 + buffer_value, y2 + buffer_value,
                                                         transform=raster.transform)

        z = raster.read(1, window=window_object)
        geo_transform = raster.window_transform(window_object)

    else:
        z = raster.read(1)
        # Get affine transform from raster
        geo_transform = raster.transform

    if return_meta:
        geo_meta = raster.meta.copy()
        geo_meta.update({"width": z.shape[1],
                         "height": z.shape[0],
                         "transform": geo_transform})

    else:
        geo_meta = None

    # extract origin, pixel width and pixel height from affine transform
    origin_x = geo_transform[2]
    origin_y = geo_transform[5]
    pixel_width = geo_transform[0]
    pixel_height = geo_transform[4]

    # Width and height of raster in pixels
    cols = z.shape[1]
    rows = z.shape[0]

    # Turn into X and Y coordinates
    x = origin_x + pixel_width * np.arange(cols)
    y = origin_y + pixel_height * np.arange(rows)

    # Set values above threshold to NaN
    if no_data != np.nan:
        z[z < -nan_threshold] = np.NaN
        z[z > nan_threshold] = np.NaN

    # Close raster
    raster.close()
    if return_meta:
        return x, y, z, geo_meta
    else:
        return x, y, z


def read_tiff(tiff: str, x_correct: float = None, y_correct: float = None, nan_threshold: Union[float, int] = 9000,
              make_y_ascending: bool = False, window: Union[str, list, tuple, np.ndarray] = None,
              buffer: Union[float, int]=100, return_meta: bool = False):
    """
    Read geotiff, rather than generic raster. Uses read_grid.
    :param tiff: Filename
    :param x_correct: Offset to get back to GMT convention
    :param y_correct: Generally same as x_correct
    :param make_y_ascending: flips y and z along y axis so that y[-1] > y[0]
    :return:
    """
    assert os.path.exists(tiff)

    # Read grid into x, y ,z
    if return_meta:
        x, y, z, geo_meta = read_grid(tiff, nan_threshold=nan_threshold, window=window, buffer=buffer,
                                      return_meta=return_meta)
    else:
        x, y, z = read_grid(tiff, nan_threshold=nan_threshold, window=window, buffer=buffer)
        geo_meta = None

    if make_y_ascending and y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    # If necessary, sort out problems with pixel vs. gridline registration (need to supply values)
    if x_correct is not None:
        x += x_correct
        print("Adjusting for GMT-Tiff difference")
    if y_correct is not None:
        y += y_correct

    if return_meta:
        return x, y, z, geo_meta
    else:
        return x, y, z


def read_gmt_grid(grd_file: str):
    """
    function to read in a netcdf4 format file. Rasterio is really slow to do it.
    :param grd_file:
    :return:
    """
    assert os.path.exists(grd_file)
    # Open dataset and check that x,y,z are variable names
    fid = Dataset(grd_file, "r")
    assert all([a in fid.variables for a in ("x", "y", "z")])
    xm, ym, zm = [fid.variables[a][:] for a in ("x", "y", "z")]
    fid.close()

    # Check arrays are either numpy arrays or masked arrays
    for a in (xm, ym, zm):
        assert any([isinstance(a, b) for b in (np.ma.MaskedArray, np.ndarray)])

    x = xm if isinstance(xm, np.ndarray) else xm.filled(fill_value=np.nan)
    y = ym if isinstance(ym, np.ndarray) else ym.filled(fill_value=np.nan)
    z = zm if isinstance(zm, np.ndarray) else zm.filled(fill_value=np.nan)

    return np.array(x), np.array(y), np.array(z)


def write_tiff(filename: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, epsg: int = 2193, reverse_y: bool = False,
               compress_lzw: bool = True):
    """
    Write x, y, z into geotiff format.
    :param filename:
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates: must have ny rows and nx columns
    :param epsg: Usually NZTM (2193)
    :param reverse_y: y starts at y_max and decreases
    :param compress_lzw: lzw compression
    :return:
    """
    # Check data have correct dimensions
    assert all([a.ndim == 1 for a in (x, y)])
    assert z.shape == (len(y), len(x))

    # Change into y-ascending format (reverse_y option changes it back later)
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    # To allow writing in correct format
    z = np.array(z, dtype=np.float64)

    # Calculate x and y spacing
    x_spacing = (max(x) - min(x)) / (len(x) - 1)
    y_spacing = (max(y) - min(y)) / (len(y) - 1)

    # Set affine transform from x and y
    if reverse_y:
        # Sometimes GIS prefer y values to descend.
        transform = rasterio.transform.Affine(x_spacing, 0., min(x), 0., -1 * y_spacing, max(y))
    else:
        transform = rasterio.transform.Affine(x_spacing, 0., min(x), 0., y_spacing, min(y))

    # create tiff profile (no. bands, data type etc.)
    profile = rasterio.profiles.DefaultGTiffProfile(count=1, dtype=np.float64, transform=transform, width=len(x),
                                                    height=len(y))

    # Set coordinate system if specified
    if epsg is not None:
        if epsg not in [2193, 4326, 32759, 32760, 27200]:
            print("EPSG:{:d} Not a recognised NZ coordinate system...".format(epsg))
            print("Writing anyway...")
        crs = rasterio.crs.CRS.from_epsg(epsg)
        profile["crs"] = crs

    # Add compression if desired
    if compress_lzw:
        profile["compress"] = "lzw"

    # Open raster file for writing
    fid = rasterio.open(filename, "w", **profile)
    # Write z to band one (depending whether y ascending/descending required).
    if reverse_y:
        fid.write(z[-1::-1], 1)
    else:
        fid.write(z, 1)
    # Close file
    fid.close()


def write_gmt_grd(x_array: np.ndarray, y_array: np.ndarray, mesh: np.ndarray, grid_name: str):
    """
    Write old style of GMT grid
    :param x_array:
    :param y_array:
    :param mesh: z values
    :param grid_name: Name of gmt grid. String, usually ending in .grd
    :return:
    """
    assert all([a.ndim == 1 for a in (x_array, y_array)])
    assert mesh.shape == (len(y_array), len(x_array))
    # Reverse y axis if necessary
    if y_array[0] > y_array[-1]:
        y_array = y_array[::-1]
        mesh = mesh[::-1, :]
    # Open NetCDF file
    fid = Dataset(grid_name, "w")
    # Create x- and y-dimensions
    _ = fid.createDimension('x', len(x_array))
    _ = fid.createDimension('y', len(y_array))
    # Create variables using these dimensions
    x_var = fid.createVariable('x', 'd', ('x',), zlib=True)
    y_var = fid.createVariable('y', 'd', ('y',), zlib=True)
    z_var = fid.createVariable('z', 'd', ('y', 'x'), zlib=True)
    # Set variable values
    x_var[:] = x_array.flatten()
    y_var[:] = y_array.flatten()
    z_var[:] = mesh
    # Close file
    fid.close()


def tiff2grd(tiff: str, grd: str, x_correct: float = None, y_correct: float = None,
             window: Union[list, tuple, int] = None, buffer: Union[float, int] = 100):
    """
    Helper function to convert geotiff to gmt grid
    :param tiff:
    :param grd:
    :param x_correct: sometimes necessary to correct for differences between pixel and gridline registration
    :param y_correct:
    :return:
    """
    assert os.path.exists(tiff)
    x, y, z = read_tiff(tiff, x_correct=x_correct, y_correct=y_correct, window=window, buffer=buffer)
    write_gmt_grd(x, y, z, grd)


def grd2tiff(grd: str, tiff: str, x_correct: float = None, y_correct: float = None):
    """
    Helper function: GMT grid to geotiff
    :param grd:
    :param tiff:
    :param x_correct:
    :param y_correct:
    :return:
    """
    assert os.path.exists(grd)
    # Read in grid
    x, y, z = read_gmt_grid(grd)
    # Correct if necessary
    if x_correct is not None:
        x_correct -= x_correct
        print("Adjusting for GMT-Tiff difference")
    if y_correct is not None:
        y -= y_correct
    write_tiff(tiff, x, y, z)


def array_to_gmt(array: np.ndarray, out_file: str):
    """

    :param array:
    :param out_file:
    :return:
    """
    assert array.shape[-1] >= 2, "Need 2 or more elements or columns"
    out_id = open(out_file, "w")
    if array.ndim == 1:
        out_str = "{:.4f} " * len(array) + "\n"
        out_id.write(out_str.format(*array))
    else:
        out_str = "{:.4f} " * len(array[0, :]) + "\n"
        for row in array:
            out_id.write(out_str.format(*row))
    out_id.close()


def clip_tiff(in_tiff: str, out_tiff: str, window: Union[str, list, tuple, int], buffer: Union[float, int] = 100):
    """

    :param in_tiff:
    :param out_tiff:
    :param window:
    :param buffer
    :return:
    """
    assert os.path.exists(in_tiff)

    x, y, z = read_tiff(in_tiff, window=window, buffer=buffer)

    write_tiff(out_tiff, x, y, z)


def reproject_tiff(in_raster: str, out_raster: str, dst_epsg: int = 4326,
                   window: Union[str, list, tuple, int] = None, buffer: Union[float, int] = 100, out_format="tiff"):
    """
    Copied off rasterio website
    :param in_raster:
    :param out_raster:
    :param dst_epsg:
    :param window
    :param buffer
    :param out_format
    :return:
    """

    assert out_format in ("tiff", "grd")
    dst_crs = "EPSG:{:d}".format(dst_epsg)
    x, y, z, src_meta = read_tiff(in_raster, return_meta=True, window=window, buffer=buffer)

    transform, width, height = calculate_default_transform(src_meta["crs"],
                                                           dst_crs,
                                                           src_meta["width"],
                                                           src_meta["height"], min(x), min(y), max(x), max(y))

    kwargs = src_meta.copy()
    kwargs.update({
                   'crs': dst_crs,
                   'transform': transform,
                   'width': width,
                   'height': height
                   })

    new_z = np.zeros((height, width))

    reproject(
        source=z,
        destination=new_z,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)

    origin_x = transform[2]
    origin_y = transform[5]
    pixel_width = transform[0]
    pixel_height = transform[4]

    # Width and height of raster in pixels
    cols = width
    rows = height

    # Turn into X and Y coordinates
    new_x = origin_x + pixel_width * np.arange(cols)
    new_y = origin_y + pixel_height * np.arange(rows)

    if out_format == "tiff":
        write_tiff(out_raster, new_x, new_y, new_z, epsg=dst_epsg)
    else:
        write_gmt_grd(new_x, new_y, new_z, out_raster)
