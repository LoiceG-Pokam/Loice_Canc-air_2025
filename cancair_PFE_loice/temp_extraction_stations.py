# import pandas as pd
# import numpy as np
# from scipy.interpolate import griddata
# import rasterio
# from rasterio.transform import from_origin
# from rasterio.mask import mask
# import geopandas as gpd
# import os
# from tqdm import tqdm
# from pyproj import Transformer
# from shapely.geometry import mapping


# df_temp = pd.read_csv("data/data_temp/temperature_ile_de_france.csv", sep=";")

# #  Charger les données

# df = df_temp[(df_temp['temperature'] > -25) & (df_temp['temperature'] < 60)]

# #  Reprojection EPSG:4326 → EPSG:2154
# transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
# df[['x', 'y']] = df.apply(lambda row: pd.Series(transformer.transform(row['longitude'], row['latitude'])), axis=1)

# #  Charger le shapefile de l’Île-de-France en EPSG:2154
# gdf = gpd.read_file("data/shapefile/IDF.shp").to_crs("EPSG:2154")
# idf_geom = [mapping(geom) for geom in gdf.geometry]

# #  Créer le dossier
# os.makedirs("rasters_temporels", exist_ok=True)

# #  Grille 50 m
# res = 50
# x_min, y_min, x_max, y_max = gdf.total_bounds

# grid_x, grid_y = np.meshgrid(
#     np.arange(x_min, x_max, res),
#     np.arange(y_min, y_max, res)
# )

# #  Dates uniques
# unique_dates = sorted(df['date'].dt.date.unique())

# #  NODATA
# nodata_value = -9999

# #  Interpolation + masque
# with tqdm(total=len(unique_dates), desc="Interpolation avec masque IDF") as pbar:
#     for date in unique_dates:
#         pbar.set_description_str(f"Traitement : {date}")
#         df_day = df[df['date'].dt.date == date]
#         if len(df_day) < 5:
#             pbar.update(1)
#             continue

#         points = df_day[['x', 'y']].values
#         values = df_day['temperature'].values
#         grid_temp = griddata(points, values, (grid_x, grid_y), method='linear')

#         if grid_temp is None:
#             pbar.update(1)
#             continue

#         grid_temp = np.where(np.isnan(grid_temp), nodata_value, grid_temp)
#         grid_temp = np.clip(grid_temp, -25, 60)

#         transform = from_origin(x_min, y_max, res, res)
#         tmp_path = f"rasters_temporels/tmp_temp_{date}.tif"
#         out_path = f"rasters_temporels/temp_{date}.tif"

#         # ➕ Sauvegarde temporaire non-masquée
#         with rasterio.open(tmp_path, "w",
#                            driver="GTiff",
#                            height=grid_temp.shape[0],
#                            width=grid_temp.shape[1],
#                            count=1,
#                            dtype='float32',
#                            crs="EPSG:2154",
#                            transform=transform,
#                            nodata=nodata_value) as dst:
#             dst.write(grid_temp.astype('float32'), 1)

#         # Appliquer le masque
#         with rasterio.open(tmp_path) as src:
#             out_image, out_transform = mask(src, idf_geom, crop=True, nodata=nodata_value)
#             out_meta = src.meta.copy()
#             out_meta.update({
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform
#             })

#         # 💾 Raster final masqué
#         with rasterio.open(out_path, "w", **out_meta) as dest:
#             dest.write(out_image)

#         os.remove(tmp_path)  # Nettoyage
#         pbar.update(1)


import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
import geopandas as gpd
import os
from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import mapping

# Lire les données de température
df_temp = pd.read_csv("C:/temperature_ile_de_france.csv", sep=";")

df_temp["date"] = pd.to_datetime(df_temp['date'])

df_temp = df_temp[df_temp["date"].dt.year.isin([2019,2018,2019,2020])]



# Filtrer les valeurs aberrantes
df = df_temp[(df_temp['temperature'] > -25) & (df_temp['temperature'] < 60)].copy()

# Reprojection : EPSG:4326 → EPSG:2154
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
df[['x', 'y']] = df.apply(lambda row: pd.Series(transformer.transform(row['longitude'], row['latitude'])), axis=1)

# Charger le shapefile et convertir en EPSG:2154
gdf = gpd.read_file("data/shapefile/IDF.shp").to_crs("EPSG:2154")
idf_geom = [mapping(geom) for geom in gdf.geometry]

# Créer le dossier de sortie
output_dir = "R:/Direction_Data/0_Projets/Projet_CANCAIR/2025_Projet_Loice/temperatures_interpolated_stations"
os.makedirs(output_dir, exist_ok=True)

# Définir la grille 50m
res = 50
x_min, y_min, x_max, y_max = gdf.total_bounds
grid_x, grid_y = np.meshgrid(
    np.arange(x_min, x_max, res),
    np.arange(y_min, y_max, res)
)

# Extraire les dates uniques
unique_dates = sorted(df['date'].dt.date.unique())

# Valeur NoData
nodata_value = -9999

# Interpolation par date
with tqdm(total=len(unique_dates), desc="Interpolation avec masque IDF") as pbar:
    for date in unique_dates:
        pbar.set_description_str(f"Traitement : {date}")
        df_day = df[df['date'].dt.date == date]
        if len(df_day) < 3:
            pbar.update(1)
            continue

        points = df_day[['x', 'y']].values
        values = df_day['temperature'].values

        # Interpolation linéaire
        grid_linear = griddata(points, values, (grid_x, grid_y), method='linear')

        # Fallback avec nearest
        grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_temp = np.where(np.isnan(grid_linear), grid_nearest, grid_linear)

        # Nettoyage : clip puis remplacer les NaN
        grid_temp = np.clip(grid_temp, -25, 60)
        grid_temp = np.where(np.isnan(grid_temp), nodata_value, grid_temp)

        # Définir la transformation
        transform = from_origin(x_min, y_max, res, res)

        tmp_path = os.path.join(output_dir, f"tmp_temp_{date}.tif")
        out_path = os.path.join(output_dir, f"temp_{date}.tif")

        # Sauvegarde raster temporaire non masqué
        with rasterio.open(tmp_path, "w",
                           driver="GTiff",
                           height=grid_temp.shape[0],
                           width=grid_temp.shape[1],
                           count=1,
                           dtype='float32',
                           crs="EPSG:2154",
                           transform=transform,
                           nodata=nodata_value) as dst:
            dst.write(grid_temp.astype('float32'), 1)

        # Application du masque spatial (Île-de-France)
        with rasterio.open(tmp_path) as src:
            out_image, out_transform = mask(src, idf_geom, crop=True, nodata=nodata_value)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # Sauvegarde raster final masqué
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

        os.remove(tmp_path)  # Supprimer le raster temporaire
        pbar.update(1)
