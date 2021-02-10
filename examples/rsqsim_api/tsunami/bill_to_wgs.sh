#!/bin/bash

for fname in for_bill/*nztm.grd
do
  number="${fname%nztm.grd}"
  echo $number
  gdal_translate $fname -of GTIFF -a_srs epsg:2193 -co compress=LZW ${number}nztm.tif
  gdalwarp ${number}nztm.tif -t_srs epsg:4326 --config CENTER_LONG 180 ${number}wgs.tif

  gdal_translate ${number}wgs.tif -co compress=LZW ${number}wgs_compressed.tif
  rm ${number}wgs.tif
  gdal_translate ${number}wgs_compressed.tif -of GMT ${number}wgs.grd
  gmt grdconvert ${number}wgs.grd+n9.96920996838686905e36 -G${number}wgs_compressed.grd

done
rm -f for_bill/*wgs.grd
