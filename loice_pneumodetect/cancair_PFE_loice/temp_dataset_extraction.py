import os 
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt

from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# 🔹 1. Ouvrir et combiner les fichiers NetCDF
def ouvrir_et_combiner_nc(dossier_nc, variable='t2m', chunks={'valid_time': 500}):
    fichiers = sorted([os.path.join(dossier_nc, f) for f in os.listdir(dossier_nc) if f.endswith('.nc')])
    if not fichiers:
        raise FileNotFoundError("❗ Aucun fichier .nc trouvé dans le dossier.")
    print("🔹 Lecture et concaténation des fichiers NetCDF...")
    try:
        ds = xr.open_mfdataset(
            fichiers,
            combine='by_coords',
            parallel=True,
            chunks=chunks,
            engine='netcdf4'
        )
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de l'ouverture : {e}")

    if variable not in ds:
        raise ValueError(f"❗ Variable '{variable}' non trouvée. Disponibles : {list(ds.data_vars)}")

    ds[variable] = ds[variable].astype('float32') - 273.15  # Kelvin -> Celsius

    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    elif 'time' not in ds.dims:
        raise KeyError("❗ Aucune dimension 'time' ou 'valid_time' trouvée.")
    
    return ds

# 🔹 2. Extraire une région géographique
def extraire_region(ds, lat_min=47.8, lat_max=49.4, lon_min=1.3, lon_max=3.7):
    return ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

# 🔹 3. Calcul de la moyenne journalière
def moyenne_journaliere(ds, variable='t2m'):
    ds = ds.sortby("time")
    return ds[variable].resample(time='1D').mean()

# 🔹 4. Charger les patients depuis CSV
def charger_patients(fichier_csv):
    df = pd.read_csv(fichier_csv)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x, df.y),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    return gdf

# 🔹 5. Filtrer les patients à l'intérieur de l'Île-de-France
def filtrer_patients_idf(gdf_patients, shapefile_idf):
    gdf_idf = gpd.read_file(shapefile_idf).to_crs("EPSG:2154")
    gdf_filtrés = gpd.sjoin(gdf_patients, gdf_idf, predicate="within", how="inner")
    print(f"✅ Patients conservés après filtre IDF : {len(gdf_filtrés)}")
    return gdf_filtrés

# 🔹 6. Associer température ↔ patients
def associer_temperature_patients_vectorise(gdf_patients, t2m_jour):
    gdf_patients_latlon = gdf_patients.to_crs("EPSG:4326")
    lats = xr.DataArray(gdf_patients_latlon.geometry.y.values, dims="points")
    lons = xr.DataArray(gdf_patients_latlon.geometry.x.values, dims="points")

    temperature_patients = []

    for date in tqdm(t2m_jour.time.values, desc="Interpolation des températures"):
        t2m_day = t2m_jour.sel(time=date)
        t2m_values = t2m_day.interp(latitude=lats, longitude=lons, method="nearest")

        df_temp = pd.DataFrame({
            "pseudo_provisoire": gdf_patients_latlon["pseudo_provisoire"].values,
            "date": pd.to_datetime(date),
            "temperature": t2m_values.values
        })
        temperature_patients.append(df_temp)

    df_temperature = pd.concat(temperature_patients, ignore_index=True)

    return df_temperature

# 🔹 7. Affichage NaN + enrichissement des valeurs NaN (VERSION MODIFIÉE)
def afficher_et_enrichir_nan(df_resultat, gdf_patients, t2m_jour, shapefile_idf, seuil_km=5):
    print("\n🔍 Analyse des valeurs NaN dans la température :")
    nb_nan = df_resultat["temperature"].isna().sum()
    pct_nan = nb_nan / len(df_resultat) * 100
    print(f"Nombre de NaN : {nb_nan}")
    print(f"Pourcentage de NaN : {pct_nan:.2f}%")
    print("\n🔎 Aperçu des 10 premières valeurs (y compris NaN) :")
    print(df_resultat.head(10))

    if nb_nan == 0:
        print("\n✅ Aucun NaN à enrichir.")
        return df_resultat

    print("\n🔧 Enrichissement des patients avec NaN...")

    # Extraire les lignes avec NaN
    df_nan = df_resultat[df_resultat["temperature"].isna()].copy()

    # Récupérer les géométries des patients correspondants (en EPSG:4326 pour KDTree)
    gdf_nan = gdf_patients[gdf_patients["pseudo_provisoire"].isin(df_nan["pseudo_provisoire"])].copy()
    gdf_nan = gdf_nan.to_crs("EPSG:4326")

    # Construction des points météo dans la même projection (EPSG:4326)
    lats = t2m_jour.latitude.values
    lons = t2m_jour.longitude.values
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    points_meteo = np.vstack([mesh_lon.ravel(), mesh_lat.ravel()]).T
    tree = cKDTree(points_meteo)

    distances = []
    temp_voisines = []
    coord_voisines = []
    dates_valides = []

    print("📍 Recherche du point météo le plus proche...")
    for idx, row in tqdm(df_nan.iterrows(), total=len(df_nan)):
        # Récupérer la géométrie du patient pour la ligne en cours
        pt = gdf_nan.loc[gdf_nan["pseudo_provisoire"] == row["pseudo_provisoire"]].geometry.iloc[0]
        x, y = pt.x, pt.y

        dist, idx_point = tree.query([x, y])
        closest_coord = points_meteo[idx_point]
        distances.append(dist)
        coord_voisines.append(closest_coord)

        date = pd.to_datetime(row["date"])
        if date in t2m_jour.time.values:
            temp = t2m_jour.sel(
                time=date,
                latitude=closest_coord[1],
                longitude=closest_coord[0],
                method="nearest"
            ).values.item()
            temp_voisines.append(temp)
            dates_valides.append(True)
        else:
            temp_voisines.append(np.nan)
            dates_valides.append(False)

    # Mise à jour des colonnes dans df_nan
    df_nan["x"] = gdf_nan.geometry.x
    df_nan["y"] = gdf_nan.geometry.y
    df_nan["temperature"] = temp_voisines
    df_nan["distance_point_meteo"] = distances
    df_nan["lat_point_plus_proche"] = [c[1] for c in coord_voisines]
    df_nan["lon_point_plus_proche"] = [c[0] for c in coord_voisines]
    df_nan["date_valide"] = dates_valides

    # Vérification d'appartenance à l'IDF
    gdf_idf = gpd.read_file(shapefile_idf).to_crs("EPSG:2154")
    gdf_nan_proj = gdf_nan.to_crs("EPSG:2154")

    # union_all() remplace unary_union
    union_idf = gdf_idf.geometry.unary_union if not hasattr(gdf_idf.geometry, "union_all") else gdf_idf.geometry.union_all()

    gdf_nan_proj["in_IDF"] = gdf_nan_proj.geometry.within(union_idf)

    # Créer une DataFrame temporaire avec pseudo_provisoire et in_IDF, index reset
    df_in_idf = gdf_nan_proj[["pseudo_provisoire", "in_IDF"]].copy()
    df_in_idf.set_index("pseudo_provisoire", inplace=True)

    # Mapper in_IDF dans df_nan selon pseudo_provisoire (clé de jointure)
    df_nan["in_IDF"] = df_nan["pseudo_provisoire"].map(df_in_idf["in_IDF"])

    # Pour info
    nb_in_idf = df_nan["in_IDF"].sum()
    print(f"✅ Parmi les NaN enrichis, {nb_in_idf} sont localisés en Île-de-France.")

    print("le nombre de patients correspondant aux NaN enrichis est :", df_nan["pseudo_provisoire"].nunique())

    patients_nan = df_nan.drop_duplicates(subset="pseudo_provisoire")


    # Remplacer dans df_resultat les lignes où temperature était NaN par les nouvelles valeurs enrichies
    df_resultat.update(df_nan)

    #Enregistrement des patients NAN
    patients_nan.to_csv("data/data_temp/patients_nan_enrichis.csv", index=False)
    print("✅ Enregistrement  des patients avec valeurs NAN terminé.")

    # Carte avec attente de fermeture
    print("🗺️ Génération de la carte des patients NaN enrichis et des points météo les plus proches...")
    print("⚠️  FERMEZ LA FENÊTRE DE LA CARTE POUR CONTINUER L'ENREGISTREMENT")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_idf.to_crs("EPSG:4326").plot(ax=ax, color='lightgrey', edgecolor='black')
    gdf_nan.plot(ax=ax, color='red', markersize=30, label='Patients NaN enrichis')

    df_coords = pd.DataFrame(coord_voisines, columns=["lon", "lat"])
    gdf_voisins = gpd.GeoDataFrame(df_coords, geometry=gpd.points_from_xy(df_coords["lon"], df_coords["lat"]), crs="EPSG:4326")
    gdf_voisins.plot(ax=ax, color='blue', markersize=10, label='Points météo les plus proches')

    plt.legend()
    plt.title("Patients avec NaN enrichis et leurs points météo les plus proches\n(Fermez cette fenêtre pour continuer)")
    
    # MÉTHODE 1 : Attendre explicitement la fermeture (recommandée)
    plt.show(block=True)  # block=True assure que le script attend
    
    # MÉTHODE 2 : Alternative avec gestionnaire d'événements (si méthode 1 ne fonctionne pas)
    # carte_fermee = False
    # def on_close(event):
    #     global carte_fermee
    #     carte_fermee = True
    #     print("✅ Carte fermée, continuation du traitement...")
    # 
    # fig.canvas.mpl_connect('close_event', on_close)
    # plt.show(block=False)
    # 
    # # Attendre que la carte soit fermée
    # import time
    # while not carte_fermee:
    #     plt.pause(0.1)  # Vérifier toutes les 100ms
    #     time.sleep(0.1)
    
    print("✅ Carte fermée, poursuite du traitement...")
    print("🚀 DÉBUT DE L'ENREGISTREMENT DES DONNÉES...")

    return df_resultat

# 🔹 8. Fonction pour sauvegarder avec tqdm
def sauvegarder_avec_progression(df, fichier_sortie, chunk_size=10000, description="Enregistrement"):
    """
    Sauvegarde un DataFrame par chunks avec barre de progression
    """
    print(f"\n💾 {description} : {fichier_sortie}")
    print(f"📊 Taille du DataFrame : {len(df):,} lignes")
    
    # Calculer le nombre de chunks
    nb_chunks = (len(df) - 1) // chunk_size + 1
    
    # Vider le fichier existant s'il existe
    if os.path.exists(fichier_sortie):
        os.remove(fichier_sortie)
    
    # Écriture avec progression
    with tqdm(total=nb_chunks, desc=f"💾 {description}", unit="chunk") as pbar:
        for i in range(nb_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            # Première écriture avec header, puis append
            if i == 0:
                chunk.to_csv(fichier_sortie, index=False, mode='w')
            else:
                chunk.to_csv(fichier_sortie, index=False, mode='a', header=False)
            
            pbar.update(1)
            pbar.set_postfix({"Lignes": f"{end_idx:,}/{len(df):,}"})
    
    print(f"✅ Fichier sauvegardé avec succès : {fichier_sortie}")

# 🔹 9. Agrégation mensuelle avec progression
def agregation_mensuelle(df_temp, fichier_sortie):
    print("\n📅 Début de l'agrégation mensuelle...")
    
    # Préparation des données
    print("🔄 Préparation des données temporelles...")
    df_temp["date"] = pd.to_datetime(df_temp["date"])
    df_temp["mois"] = df_temp["date"].dt.to_period("M")
    
    # Groupby avec progression
    print("📊 Calcul des moyennes mensuelles par patient...")
    
    # Grouper par patient et mois
    groupes = df_temp.groupby(["pseudo_provisoire", "mois"])
    
    # Calculer avec tqdm
    resultats = []
    with tqdm(total=len(groupes), desc="📈 Agrégation mensuelle", unit="groupe") as pbar:
        for (patient, mois), groupe in groupes:
            temp_moyenne = groupe["temperature"].mean()
            resultats.append({
                "pseudo_provisoire": patient,
                "mois": str(mois),
                "temperature_moyenne_mensuelle": temp_moyenne
            })
            pbar.update(1)
    
    # Créer le DataFrame final
    df_mensuel = pd.DataFrame(resultats)
    
    # Sauvegarder avec progression
    sauvegarder_avec_progression(
        df_mensuel, 
        fichier_sortie, 
        chunk_size=5000,
        description="Données mensuelles"
    )
    
    print(f"✅ Agrégation mensuelle terminée : {len(df_mensuel):,} lignes")
    return df_mensuel

# 🔹 10. Script principal (VERSION MODIFIÉE)
def main():
    dossier_nc = "data/data_temp/NC"  # adapter si besoin
    fichier_patients = "data/data_cleaned/patients_FR_IDF_geocoded_adultes_clinique_patho.csv"
    shapefile_idf = "data/shapefile/IDF.shp"
    sortie_csv = "R:/Direction_Data/0_Projets/Projet_CANCAIR/2025_Projet_Loice/Data/patients_temp_daily.csv"
    sortie_mensuelle = "R:/Direction_Data/0_Projets/Projet_CANCAIR/2025_Projet_Loice/Data/patients_temp_mensuel.csv"

    ds = ouvrir_et_combiner_nc(dossier_nc)
    ds_idf = extraire_region(ds)
    t2m_jour = moyenne_journaliere(ds_idf)

    gdf_patients = charger_patients(fichier_patients)
    gdf_filtrés = filtrer_patients_idf(gdf_patients, shapefile_idf)

    df_resultat = associer_temperature_patients_vectorise(gdf_filtrés, t2m_jour)

    # Affichage NaN + enrichissement
    df_resultat = afficher_et_enrichir_nan(df_resultat, gdf_filtrés, t2m_jour, shapefile_idf)

    print("\n" + "="*60)
    print("🚀 DÉBUT DE L'ENREGISTREMENT DES FICHIERS")
    print("="*60)
    
    # Sauvegarde des données daily avec progression
    print("\n📊 ÉTAPE 1/2 : Enregistrement des données journalières")
    sauvegarder_avec_progression(
        df_resultat, 
        sortie_csv, 
        chunk_size=10000,
        description="Données journalières"
    )

    print("\n📅 ÉTAPE 2/2 : Enregistrement des données mensuelles")
    # Agrégation mensuelle avec progression
    agregation_mensuelle(df_resultat, sortie_mensuelle)
    
    print("\n" + "="*60)
    print("🎉 ENREGISTREMENT TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print(f"📁 Fichier journalier : {sortie_csv}")
    print(f"📁 Fichier mensuel : {sortie_mensuelle}")

if __name__ == "__main__":
    main()