import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
import geopandas as gpd
import os
from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

def idw_interpolation(points, values, grid_points, power=2, smoothing=0, max_distance=None, min_neighbors=3, max_neighbors=12):
    """
    Interpolation par pondération inverse de la distance (IDW) améliorée
    
    Parameters:
    -----------
    points : array-like, shape (n_samples, 2)
        Coordonnées des points de mesure [x, y]
    values : array-like, shape (n_samples,)
        Valeurs mesurées aux points
    grid_points : array-like, shape (n_grid, 2)
        Coordonnées des points de la grille
    power : float, default=2
        Puissance pour la pondération (plus élevé = influence plus locale)
    smoothing : float, default=0
        Paramètre de lissage pour éviter la singularité aux points de mesure
    max_distance : float, optional
        Distance maximale d'influence (au-delà, poids = 0)
    min_neighbors : int, default=3
        Nombre minimum de voisins pour l'interpolation
    max_neighbors : int, default=12
        Nombre maximum de voisins à considérer
    
    Returns:
    --------
    interpolated_values : array
        Valeurs interpolées sur la grille
    """
    
    # Calculer les distances entre tous les points de grille et les points de mesure
    distances = cdist(grid_points, points, metric='euclidean')
    
    # Initialiser le résultat
    interpolated = np.full(grid_points.shape[0], np.nan)
    
    for i in range(grid_points.shape[0]):
        dist_to_point = distances[i, :]
        
        # Appliquer la distance maximale si spécifiée
        if max_distance is not None:
            valid_mask = dist_to_point <= max_distance
            if not np.any(valid_mask):
                continue
            dist_to_point = dist_to_point[valid_mask]
            point_values = values[valid_mask]
        else:
            point_values = values
        
        # Sélectionner les k plus proches voisins
        if len(dist_to_point) > max_neighbors:
            nearest_indices = np.argsort(dist_to_point)[:max_neighbors]
            dist_to_point = dist_to_point[nearest_indices]
            point_values = point_values[nearest_indices]
        
        # Vérifier le nombre minimum de voisins
        if len(dist_to_point) < min_neighbors:
            continue
        
        # Éviter la division par zéro avec le paramètre de lissage
        dist_to_point = dist_to_point + smoothing
        
        # Calculer les poids IDW
        weights = 1.0 / (dist_to_point ** power)
        
        # Normaliser les poids
        weights = weights / np.sum(weights)
        
        # Calculer la valeur interpolée
        interpolated[i] = np.sum(weights * point_values)
    
    return interpolated

def adaptive_idw_parameters(points, extent, resolution):
    """
    Calcule des paramètres IDW adaptatifs basés sur la densité des stations
    """
    area = (extent[2] - extent[0]) * (extent[3] - extent[1])  # xmax-xmin, ymax-ymin
    density = len(points) / (area / 1e6)  # stations par km²
    
    # Ajuster les paramètres selon la densité
    if density > 5:  # Haute densité
        power = 2.5
        max_distance = 15000  # 15 km
        min_neighbors = 4
        max_neighbors = 8
    elif density > 2:  # Densité moyenne
        power = 2.0
        max_distance = 25000  # 25 km
        min_neighbors = 3
        max_neighbors = 10
    else:  # Faible densité
        power = 1.5
        max_distance = 50000  # 50 km
        min_neighbors = 2
        max_neighbors = 15
    
    return power, max_distance, min_neighbors, max_neighbors

# ===== SCRIPT PRINCIPAL =====

# Lire les données de température
df_temp = pd.read_csv("C:/temperature_ile_de_france.csv", sep=";")
df_temp["date"] = pd.to_datetime(df_temp['date'])
df_temp = df_temp[df_temp["date"].dt.year.isin([2018,2019])]

# Filtrer les valeurs aberrantes
df = df_temp[(df_temp['temperature'] > -25) & (df_temp['temperature'] < 60)].copy()

# Reprojection : EPSG:4326 → EPSG:2154
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
df[['x', 'y']] = df.apply(lambda row: pd.Series(transformer.transform(row['longitude'], row['latitude'])), axis=1)

# Charger le shapefile et convertir en EPSG:2154
gdf = gpd.read_file("data/shapefile/IDF.shp").to_crs("EPSG:2154")
idf_geom = [mapping(geom) for geom in gdf.geometry]

# Créer le dossier de sortie
output_dir = "R:/Direction_Data/0_Projets/Projet_CANCAIR/2025_Projet_Loice/temperatures_interpolated_IDW"
os.makedirs(output_dir, exist_ok=True)

# Définir la grille
res = 50
x_min, y_min, x_max, y_max = gdf.total_bounds

# Créer la grille de points
x_coords = np.arange(x_min, x_max, res)
y_coords = np.arange(y_min, y_max, res)
grid_x, grid_y = np.meshgrid(x_coords, y_coords)

# Aplatir la grille pour l'interpolation
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# Extraire les dates uniques
unique_dates = sorted(df['date'].dt.date.unique())

# Valeur NoData
nodata_value = -9999

print(f"Traitement de {len(unique_dates)} dates avec interpolation IDW adaptative")

# Interpolation par date avec IDW
with tqdm(total=len(unique_dates), desc="Interpolation IDW") as pbar:
    for date in unique_dates:
        pbar.set_description_str(f"IDW - {date}")
        df_day = df[df['date'].dt.date == date]
        
        if len(df_day) < 2:
            pbar.update(1)
            continue

        # Coordonnées et valeurs des stations
        points = df_day[['x', 'y']].values
        values = df_day['temperature'].values

        # Calculer les paramètres adaptatifs
        extent = [x_min, y_min, x_max, y_max]
        power, max_distance, min_neighbors, max_neighbors = adaptive_idw_parameters(points, extent, res)
        
        # Interpolation IDW
        grid_temp_flat = idw_interpolation(
            points=points,
            values=values,
            grid_points=grid_points,
            power=power,
            smoothing=10.0,  # 10m de lissage
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors
        )
        
        # Reshaper en grille 2D
        grid_temp = grid_temp_flat.reshape(grid_y.shape)
        
        # Appliquer les contraintes et gérer les NaN
        grid_temp = np.clip(grid_temp, -25, 60)
        grid_temp = np.where(np.isnan(grid_temp), nodata_value, grid_temp)
        
        # Définir la transformation
        transform = from_origin(x_min, y_max, res, res)
        
        tmp_path = os.path.join(output_dir, f"tmp_temp_idw_{date}.tif")
        out_path = os.path.join(output_dir, f"temp_idw_{date}.tif")
        
        # Sauvegarde raster temporaire
        with rasterio.open(tmp_path, "w",
                           driver="GTiff",
                           height=grid_temp.shape[0],
                           width=grid_temp.shape[1],
                           count=1,
                           dtype='float32',
                           crs="EPSG:2154",
                           transform=transform,
                           nodata=nodata_value,
                           compress='lzw') as dst:
            dst.write(grid_temp.astype('float32'), 1)

        # Application du masque spatial
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

        os.remove(tmp_path)
        pbar.update(1)

print(f"Interpolation IDW terminée. Rasters sauvegardés dans : {output_dir}")

# ===== FONCTION BONUS : VALIDATION CROISÉE =====

def cross_validation_idw(df_day, power=2, max_distance=25000, k_folds=5):
    """
    Validation croisée pour évaluer la qualité de l'interpolation IDW
    """
    from sklearn.model_selection import KFold
    
    points = df_day[['x', 'y']].values
    values = df_day['temperature'].values
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    errors = []
    
    for train_idx, test_idx in kf.split(points):
        train_points = points[train_idx]
        train_values = values[train_idx]
        test_points = points[test_idx]
        test_values = values[test_idx]
        
        # Interpoler aux points de test
        predicted = idw_interpolation(
            train_points, train_values, test_points,
            power=power, max_distance=max_distance
        )
        
        # Calculer l'erreur
        valid_mask = ~np.isnan(predicted)
        if np.any(valid_mask):
            mae = np.mean(np.abs(predicted[valid_mask] - test_values[valid_mask]))
            errors.append(mae)
    
    return np.mean(errors) if errors else np.nan

#Exemple d'utilisation de la validation croisée (décommenter pour tester)
test_date = unique_dates[0]
df_test = df[df['date'].dt.date == test_date]
if len(df_test) >= 10:
    mae = cross_validation_idw(df_test)
    print(f"Erreur moyenne absolue (validation croisée) : {mae:.2f}°C")