import xarray as xr
import numpy as np
import os
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm
import rioxarray as rxr
import geopandas as gpd
from rasterio.mask import mask
import rasterio

def aggregate_temperature_data(input_dir, output_dir, years=[2017, 2018, 2019]):
    """
    Agrège les données de température mensuelles en données annuelles
    
    Parameters:
    input_dir (str): Répertoire contenant les fichiers era5_land_YYYY_MM.nc
    output_dir (str): Répertoire de sortie pour les fichiers annuels
    years (list): Liste des années à traiter
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for year in years:
        print(f"Traitement de l'année {year}...")
        
        # Pattern pour trouver tous les fichiers de l'année
        pattern = os.path.join(input_dir, f"era5_land_{year}_*.nc")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"Aucun fichier trouvé pour l'année {year}")
            continue
            
        print(f"  Fichiers trouvés: {len(files)}")
        
        # Ouvrir et concaténer tous les fichiers de l'année
        datasets = []
        for file in tqdm(files, desc=f"Chargement des fichiers {year}", unit="fichier"):
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"    Erreur lors du chargement de {file}: {e}")
        
        if not datasets:
            print(f"Aucun dataset valide pour l'année {year}")
            continue
        
        # Concaténer le long de la dimension temporelle
        annual_ds = xr.concat(datasets, dim='time')
        
        # Trier par temps si nécessaire
        annual_ds = annual_ds.sortby('time')
        
        # Ajouter des attributs
        annual_ds.attrs['title'] = f'ERA5-Land data for {year}'
        annual_ds.attrs['description'] = f'Aggregated temperature data for year {year}'
        annual_ds.attrs['creation_date'] = pd.Timestamp.now().isoformat()
        
        # Sauvegarder le fichier annuel
        output_file = os.path.join(output_dir, f"era5_land_{year}_annual.nc")
        annual_ds.to_netcdf(output_file)
        print(f"  Sauvegardé: {output_file}")
        
        # Fermer les datasets pour libérer la mémoire
        annual_ds.close()
        for ds in datasets:
            ds.close()
        
        print(f"Année {year} terminée.\n")


def aggregate_air_quality_data(base_dir, output_dir, years=[2017, 2018, 2019]):
    """
    Agrège les données de qualité de l'air journalières en moyennes annuelles
    Structure attendue: base_dir/YYYY/POLLUANT_TYPE_IDF_YYYYMMDD.nc
    
    Polluants et valeurs spécifiques:
    - O3, NO2: maxJ
    - PM10, PM2.5: meanJ
    
    Parameters:
    base_dir (str): Répertoire de base contenant les dossiers par année
    output_dir (str): Répertoire de sortie
    years (list): Liste des années à traiter
    """
    # Configuration des polluants et leurs types de valeurs
    pollutant_config = {
        'O3': ['maxJ'],
        'NO2': ['maxJ'], 
        'PM10': ['meanJ'],
        'PM2.5': ['meanJ']
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        
        if not os.path.exists(year_dir):
            print(f"Dossier {year_dir} non trouvé, passage à l'année suivante...")
            continue
            
        print(f"Traitement de l'année: {year}")
        
        for pollutant, value_types in pollutant_config.items():
            print(f"  Polluant: {pollutant}")
            
            for value_type in value_types:
                print(f"    Type de valeur: {value_type}")
                
                # Pattern pour trouver tous les fichiers du polluant/valeur pour cette année
                # Format: POLLUANT_TYPE_IDF_YYYYMMDD.nc
                pattern = os.path.join(year_dir, f"{pollutant}_{value_type}_IDF_{year}*.nc")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    print(f"      Aucun fichier trouvé pour {pollutant}_{value_type}_{year}")
                    continue
                
                print(f"      Fichiers trouvés: {len(files)}")
                
                # Traitement optimisé par chunks pour éviter les problèmes de mémoire
                try:
                    print(f"      Traitement par chunks pour optimiser la mémoire...")
                    
                    # Traiter par plus petits lots de fichiers pour éviter le dépassement mémoire
                    chunk_size = 10  # Réduire à 10 fichiers à la fois
                    total_files = len(files)
                    
                    # Variables pour accumulation
                    accumulated_sum = None
                    total_count = 0
                    
                    # Barre de progression pour les chunks
                    chunk_progress = tqdm(range(0, total_files, chunk_size), 
                                        desc=f"Traitement {pollutant}_{value_type}_{year}", 
                                        unit="chunk")
                    
                    for i in chunk_progress:
                        chunk_files = files[i:i+chunk_size]
                        
                        # Traiter fichier par fichier dans le chunk pour optimiser la mémoire
                        chunk_sum = None
                        chunk_count = 0
                        
                        file_progress = tqdm(chunk_files, 
                                           desc=f"Chunk {i//chunk_size + 1}", 
                                           unit="fichier", 
                                           leave=False)
                        
                        for file in file_progress:
                            try:
                                # Charger avec chunking agressif
                                ds = xr.open_dataset(file, chunks={'x': 500, 'y': 500})
                                
                                # Squeeze pour éliminer la dimension temporelle s'il n'y en a qu'une
                                if 'time' in ds.sizes and ds.sizes['time'] == 1:
                                    ds = ds.squeeze('time', drop=True)
                                
                                # Accumuler directement sans concaténation
                                if chunk_sum is None:
                                    chunk_sum = ds.copy()
                                else:
                                    chunk_sum = chunk_sum + ds
                                
                                chunk_count += 1
                                ds.close()
                                
                            except Exception as e:
                                print(f"          Erreur fichier {os.path.basename(file)}: {e}")
                        
                        if chunk_sum is None:
                            continue
                        
                        # Accumuler les chunks
                        if accumulated_sum is None:
                            accumulated_sum = chunk_sum
                        else:
                            accumulated_sum = accumulated_sum + chunk_sum
                        
                        total_count += chunk_count
                        
                        # Fermer le chunk
                        chunk_sum.close()
                        
                        # Mettre à jour la description de la barre de progression
                        chunk_progress.set_postfix({
                            'fichiers_traités': total_count,
                            'chunk_size': chunk_count
                        })
                        
                        # Forcer le garbage collection pour libérer la mémoire
                        import gc
                        gc.collect()
                    
                    # Fermer la barre de progression des chunks
                    chunk_progress.close()
                    
                    if accumulated_sum is None or total_count == 0:
                        print(f"      Aucune donnée valide pour {pollutant}_{value_type}_{year}")
                        continue
                    
                    # Calculer la moyenne finale
                    print(f"      Calcul de la moyenne finale sur {total_count} fichiers...")
                    annual_mean = accumulated_sum / total_count
                    
                    # Ajouter des métadonnées
                    annual_mean.attrs['title'] = f'{pollutant} {value_type} annual mean for {year}'
                    annual_mean.attrs['description'] = f'Annual mean of {pollutant} {value_type} values for year {year}'
                    annual_mean.attrs['creation_date'] = pd.Timestamp.now().isoformat()
                    annual_mean.attrs['source_files_count'] = total_count
                    
                    # Sauvegarder directement dans le dossier de sortie
                    output_file = os.path.join(output_dir, 
                                             f"{pollutant}_{value_type}_IDF_{year}_annual_mean.nc")
                    
                    # Optimisation pour l'écriture du fichier avec gestion des gros datasets
                    print(f"      Optimisation pour l'écriture du fichier...")
                    
                    # Convertir en numpy array si nécessaire pour éviter les blocages dask
                    print(f"      Chargement des données en mémoire pour écriture...")
                    annual_mean = annual_mean.compute()  # Force le calcul complet
                    
                    # Chunks optimaux pour l'écriture
                    annual_mean = annual_mean.chunk({'x': 1000, 'y': 1000})
                    
                    # Écriture sans compression pour vitesse maximale
                    print(f"      Écriture en cours (sans compression pour vitesse)...")
                    try:
                        # Tentative 1: Écriture simple sans compression
                        annual_mean.to_netcdf(output_file, 
                                            engine='netcdf4',
                                            format='NETCDF4')
                        print(f"      ✅ Sauvegardé: {os.path.basename(output_file)}")
                        
                    except Exception as e1:
                        print(f"      Tentative 1 échouée: {e1}")
                        print(f"      Tentative 2: Écriture par blocs...")
                        
                        try:
                            # Tentative 2: Écriture par blocs manuels
                            import tempfile
                            temp_dir = tempfile.gettempdir()
                            temp_file = os.path.join(temp_dir, f"temp_{pollutant}_{value_type}_{year}.nc")
                            
                            # Écriture temporaire
                            annual_mean.to_netcdf(temp_file)
                            
                            # Copie vers destination finale
                            import shutil
                            shutil.move(temp_file, output_file)
                            print(f"      ✅ Sauvegardé via fichier temporaire: {os.path.basename(output_file)}")
                            
                        except Exception as e2:
                            print(f"      Tentative 2 échouée: {e2}")
                            print(f"      Tentative 3: Écriture avec scipy...")
                            
                            try:
                                # Tentative 3: Utiliser scipy netcdf
                                annual_mean.to_netcdf(output_file, 
                                                    engine='scipy',
                                                    format='NETCDF3_CLASSIC')
                                print(f"      ✅ Sauvegardé avec scipy: {os.path.basename(output_file)}")
                                
                            except Exception as e3:
                                print(f"      ❌ Toutes les tentatives d'écriture ont échoué:")
                                print(f"        - Netcdf4: {e1}")
                                print(f"        - Fichier temp: {e2}")  
                                print(f"        - Scipy: {e3}")
                                print(f"      Passage au fichier suivant...")
                    
                    # Fermer les datasets
                    annual_mean.close()
                    accumulated_sum.close()
                    
                    # Garbage collection final
                    gc.collect()
                    
                except Exception as e:
                    print(f"      Erreur lors de l'agrégation optimisée: {e}")
                    import traceback
                    traceback.print_exc()


def clip_to_idf_shapefile(input_file, output_file, shapefile_path):
    """
    Découpe un fichier NetCDF selon le shapefile de l'Île-de-France
    
    Parameters:
    input_file (str): Chemin vers le fichier NetCDF d'entrée
    output_file (str): Chemin vers le fichier NetCDF de sortie découpé
    shapefile_path (str): Chemin vers le shapefile de l'Île-de-France
    """
    try:
        print(f"    Découpage selon le shapefile IDF...")
        
        # Charger le shapefile
        idf_gdf = gpd.read_file(shapefile_path)
        
        # S'assurer que le CRS est correct (généralement EPSG:2154 pour l'IDF)
        if idf_gdf.crs is None:
            print(f"      Warning: CRS non défini dans le shapefile")
            idf_gdf = idf_gdf.set_crs('EPSG:2154')  # Lambert 93 pour la France
        
        # Charger le fichier NetCDF avec rioxarray
        ds = xr.open_dataset(input_file)
        
        # Vérifier si les coordonnées spatiales existent
        spatial_coords = []
        for coord in ['x', 'y', 'lon', 'lat', 'longitude', 'latitude']:
            if coord in ds.coords:
                spatial_coords.append(coord)
        
        if len(spatial_coords) < 2:
            print(f"      Erreur: Coordonnées spatiales non trouvées dans {input_file}")
            return False
        
        # Déterminer les noms des coordonnées
        if 'x' in ds.coords and 'y' in ds.coords:
            x_coord, y_coord = 'x', 'y'
        elif 'lon' in ds.coords and 'lat' in ds.coords:
            x_coord, y_coord = 'lon', 'lat'
        elif 'longitude' in ds.coords and 'latitude' in ds.coords:
            x_coord, y_coord = 'longitude', 'latitude'
        else:
            print(f"      Erreur: Format de coordonnées non reconnu")
            return False
        
        # Assigner un CRS au dataset s'il n'en a pas
        if not hasattr(ds, 'rio') or ds.rio.crs is None:
            # Essayer de déterminer le CRS selon les coordonnées
            x_vals = ds[x_coord].values
            y_vals = ds[y_coord].values
            
            if np.max(x_vals) > 180 or np.min(x_vals) < -180:
                # Probablement en mètres (Lambert 93)
                ds = ds.rio.write_crs('EPSG:2154')
            else:
                # Probablement en degrés (WGS84)
                ds = ds.rio.write_crs('EPSG:4326')
        
        # Reprojeter le shapefile dans le même CRS que le dataset
        idf_gdf_reprojected = idf_gdf.to_crs(ds.rio.crs)
        
        # Découper le dataset selon le shapefile
        print(f"      Découpage en cours...")
        clipped_ds = ds.rio.clip(idf_gdf_reprojected.geometry.values, 
                                ds.rio.crs, 
                                drop=True, 
                                invert=False)
        
        # Ajouter des métadonnées
        clipped_ds.attrs['clipped_region'] = 'Île-de-France'
        clipped_ds.attrs['shapefile_source'] = os.path.basename(shapefile_path)
        clipped_ds.attrs['clipping_date'] = pd.Timestamp.now().isoformat()
        
        # Sauvegarder le fichier découpé
        encoding = {}
        for var in clipped_ds.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 1}
        
        clipped_ds.to_netcdf(output_file, encoding=encoding)
        
        # Fermer les datasets
        ds.close()
        clipped_ds.close()
        
        print(f"      ✅ Découpé et sauvegardé: {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"      ❌ Erreur lors du découpage: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_clipping_batch(input_dir, output_dir, shapefile_path, pattern="*.nc"):
    """
    Découpe tous les fichiers NetCDF d'un dossier selon le shapefile
    
    Parameters:
    input_dir (str): Dossier contenant les fichiers NetCDF à découper
    output_dir (str): Dossier de sortie pour les fichiers découpés
    shapefile_path (str): Chemin vers le shapefile de l'Île-de-France
    pattern (str): Pattern pour sélectionner les fichiers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Trouver tous les fichiers NetCDF
    files = glob.glob(os.path.join(input_dir, pattern))
    
    if not files:
        print(f"Aucun fichier trouvé dans {input_dir} avec le pattern {pattern}")
        return
    
    print(f"\n=== DÉCOUPAGE SELON LE SHAPEFILE IDF ===")
    print(f"Fichiers à traiter: {len(files)}")
    print(f"Shapefile: {shapefile_path}")
    
    # Traiter chaque fichier avec barre de progression
    successful_clips = 0
    
    for file in tqdm(files, desc="Découpage des fichiers", unit="fichier"):
        filename = os.path.basename(file)
        
        # Créer le nom de fichier de sortie (ajouter "_IDF" avant l'extension)
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            output_filename = f"{name_parts[0]}_IDF.{name_parts[1]}"
        else:
            output_filename = f"{filename}_IDF"
        
        output_file = os.path.join(output_dir, output_filename)
        
        # Vérifier si le fichier de sortie existe déjà
        if os.path.exists(output_file):
            print(f"  Fichier déjà existant: {output_filename}")
            continue
        
        print(f"  Traitement: {filename}")
        
        if clip_to_idf_shapefile(file, output_file, shapefile_path):
            successful_clips += 1
        
    print(f"\n✅ Découpage terminé: {successful_clips}/{len(files)} fichiers traités avec succès")


def main():
    """
    Fonction principale pour exécuter les agrégations et le découpage
    """
    # Configuration des chemins - MODIFIEZ CES CHEMINS SELON VOTRE STRUCTURE
    temp_input_dir = r"H:\PFE Loice\Notebooks\data\data_temp\nc"  
    temp_output_dir = r"R:\Direction_Data\0_Projets\Projet_CANCAIR\2025_Projet_Loice\rasters_pollution_temp_annuel"  
    
    air_quality_base_dir = r"R:\Direction_Data\0_Projets\Projet_CANCAIR\data\airparif_dir\airparif\pollution"  
    air_quality_output_dir = r"R:\Direction_Data\0_Projets\Projet_CANCAIR\2025_Projet_Loice\rasters_pollution_temp_annuel"  
    
    # Dossier pour les fichiers découpés selon l'IDF
    idf_clipped_dir = r"R:\Direction_Data\0_Projets\Projet_CANCAIR\2025_Projet_Loice\rasters_pollution_temp_annuel_IDF"
    
    # Chemin vers le shapefile de l'Île-de-France - MODIFIEZ SELON VOTRE SHAPEFILE
    idf_shapefile = r"H:\PFE Loice\Notebooks\data\shapefile\IDF.shp"  # À MODIFIER
    
    # Années à traiter
    years = [2017, 2018, 2019]
    
    print("=== AGRÉGATION DES DONNÉES DE TEMPÉRATURE ===")
    aggregate_temperature_data(temp_input_dir, temp_output_dir, years)
    
    # print("\n=== AGRÉGATION DES DONNÉES DE QUALITÉ DE L'AIR ===")
    # aggregate_air_quality_data(air_quality_base_dir, air_quality_output_dir, years)

    print("\n=== DÉCOUPAGE SELON LE SHAPEFILE IDF ===")
    # Découper les fichiers de température
    print("\nDécoupage des fichiers de température:")
    process_clipping_batch(temp_output_dir, idf_clipped_dir, idf_shapefile, "era5_land_*_annual.nc")
    
    # Découper les fichiers de qualité de l'air
    # print("\nDécoupage des fichiers de qualité de l'air:")
    # process_clipping_batch(air_quality_output_dir, idf_clipped_dir, idf_shapefile, "*_annual_mean.nc")

    print("\n=== TRAITEMENT TERMINÉ ===")
    print(f"Fichiers complets dans: {temp_output_dir}")
    print(f"Fichiers découpés IDF dans: {idf_clipped_dir}")


# Fonction utilitaire pour examiner la structure des données
def examine_file_structure(file_path):
    """
    Examine la structure d'un fichier NetCDF
    """
    try:
        ds = xr.open_dataset(file_path)
        print(f"\nStructure du fichier: {file_path}")
        print("Dimensions:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
        
        print("\nVariables:")
        for var in ds.data_vars:
            print(f"  {var}: {ds[var].dims}")
        
        print("\nCoordonnées:")
        for coord in ds.coords:
            print(f"  {coord}: {ds[coord].dims}")
        
        ds.close()
        
    except Exception as e:
        print(f"Erreur lors de l'examen du fichier {file_path}: {e}")


def verify_resolution_conservation(input_file, output_file):
    """
    Vérifie que la résolution spatiale est conservée
    """
    try:
        ds_input = xr.open_dataset(input_file)
        ds_output = xr.open_dataset(output_file)
        
        print("=== VÉRIFICATION DE LA RÉSOLUTION ===")
        print(f"Fichier d'entrée: {input_file}")
        print(f"  Dimensions spatiales: {[dim for dim in ds_input.dims if dim != 'time']}")
        print(f"  Tailles: {[(dim, ds_input.dims[dim]) for dim in ds_input.dims if dim != 'time']}")
        
        print(f"\nFichier de sortie: {output_file}")
        print(f"  Dimensions spatiales: {[dim for dim in ds_output.dims if dim != 'time']}")
        print(f"  Tailles: {[(dim, ds_output.dims[dim]) for dim in ds_output.dims if dim != 'time']}")
        
        # Vérification des coordonnées
        for coord in ['longitude', 'latitude', 'lon', 'lat', 'x', 'y']:
            if coord in ds_input.coords and coord in ds_output.coords:
                input_coord = ds_input.coords[coord]
                output_coord = ds_output.coords[coord]
                are_equal = np.array_equal(input_coord.values, output_coord.values)
                print(f"  {coord}: {'✓ Identique' if are_equal else '✗ Différent'}")
        
        ds_input.close()
        ds_output.close()
        
    except Exception as e:
        print(f"Erreur lors de la vérification: {e}")


if __name__ == "__main__":
    # Décommentez cette ligne pour examiner la structure d'un fichier exemple
    # examine_file_structure("path/to/your/example/file.nc")
    
    # Exécuter le traitement principal
    main()