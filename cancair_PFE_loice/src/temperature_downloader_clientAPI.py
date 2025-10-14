from rasterio.transform import from_origin
from scipy.interpolate import griddata
from rasterio.transform import from_origin
import cdsapi
import os
import calendar


os.environ['REQUESTS_CA_BUNDLE'] = '*************************'("chemin vers le certificat .pem")

c = cdsapi.Client()

dataset = "reanalysis-era5-land"

years = ["2017", "2018", "2019"]
months = [f"{m:02d}" for m in range(1, 13)]

for year in years:
    for month in months:
        # Calculer le nombre de jours dans le mois (utile pour février et les mois à 30 jours)
        nb_days = calendar.monthrange(int(year), int(month))[1]
        days = [f"{d:02d}" for d in range(1, nb_days + 1)]

        filename = f"era5_land_{year}_{month}.nc"  # extension netcdf

        print(f"Téléchargement : {filename} ...")

        request = {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "skin_temperature",
                "surface_net_solar_radiation"
            ],
            "year": year,
            "month": month,
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "format": "netcdf",
            "download_format": "unarchived",
            "area": [49.5, 1.5, 48, 3.5]  
        }

        c.retrieve(dataset, request, filename)

print("Téléchargements terminés.")