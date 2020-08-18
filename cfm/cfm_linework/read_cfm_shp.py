import geopandas as gpd
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shapely.geometry import LineString
from typing import Union

shp_file = "linework_test_polylines.shp"

shp_df = gpd.GeoDataFrame.from_file(shp_file)
sorted_df = shp_df.sort_values("Name")
sorted_wgs = shp_df.to_crs(epsg=4326)

g0 = sorted_wgs.geometry[0]
f0 = sorted_wgs.iloc[0]

opensha_element = ET.Element("OpenSHA")

required_values = ['Depth_Best', 'Depth_Max', 'Depth_Min', 'Dip_Best',
       'Dip_Dir', 'Dip_Max', 'Dip_Min', 'FZ_Name', 'Name', 'Number',
       'Qual_Code', 'Rake_Best', 'Rake_Max', 'Rake_Min', 'Sense_Dom',
       'Sense_Sec', 'Source1_1', 'Source2', 'SR_Best', 'SR_Max', 'SR_Min',
       'geometry']


def fault_model_xml(fault_info: pd.Series, section_id: int):
    
    tag_name = "i{:d}"
    attribute_dic = {"sectionId": "{:d}".format(section_id),
                     "sectionName": fault_info.Name}


def fault_trace_xml(geometry: LineString, section_name: str, z: Union[float, int] = 0):
    trace_element = ET.Element("FaultTrace", attrib={"name": section_name})
    ll_float_str = "{:.4f}"

    x, y = geometry.xy
    for x_i, y_i in zip(x, y):
        loc_element = ET.Element("Location", attrib={"Latitude": ll_float_str.format(y_i),
                                                     "Longitude": ll_float_str.format(x_i),
                                                     "Depth": ll_float_str.format(z)})
        trace_element.append(loc_element)

    return trace_element

test = fault_trace_xml(f0.geometry, "test")
opensha_element.append(test)

xml_dom = minidom.parseString(ET.tostring(opensha_element, encoding="UTF-8", xml_declaration=True))
pretty_xml_str = xml_dom.toprettyxml(indent="  ", encoding="utf-8")
with open("test2.xml", "wb") as fid:
    fid.write(pretty_xml_str)
