import pandas as pd
from pyproj import Transformer

trans = Transformer.from_crs(4326, 2193, always_xy=True)

data = pd.read_excel("site_subsurface_information_DRAFT08.xls", sheet_name=1, skiprows=5)
of_interest = data.iloc[:-2, [0, 1, 127, 128]]
of_interest.columns = ["ID", "Station", "Latitude", "Longitude"]
of_interest.Latitude *= -1

nztm_x, nztm_y = trans.transform(of_interest.Longitude, of_interest.Latitude)
of_interest["nztm_x"] = nztm_x
of_interest["nztm_y"] = nztm_y

of_interest.to_csv("geonet_sm_sites.csv", index=False)
