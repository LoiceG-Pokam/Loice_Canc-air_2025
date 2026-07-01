# -*- coding: utf-8 -*-

"""
Script d'extraction des concentrations de polluants pour chaque patient
Basé sur les coordonnées géographiques (lat, lon) et les dates
"""

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
import glob


class ExtracteurPollution:
    """
    Classe pour extraire les concentrations de polluants pour des patients
    à partir de fichiers NetCDF
    """
    
    def __init__(self, dossier_nc):
        """
        Initialise l'extracteur avec le dossier contenant les fichiers .nc
        
        Args:
            dossier_nc: Chemin vers le dossier contenant les fichiers NetCDF
        """
        self.dossier_nc = dossier_nc
        self.fichiers_nc = {}
        self.catalogue = {}
        
    def cataloguer_fichiers(self):
        """
        Catalogue tous les fichiers NetCDF disponibles par polluant, type et année
        """
        fichiers = glob.glob(os.path.join(self.dossier_nc, "*.nc"))
        
        for fichier in fichiers:
            nom = os.path.basename(fichier)
            
            # Parser le nom du fichier: INERIS.REANALYSED.DOMAIN.YEAR.POLLUTANT.TYPE.2gis.nc
            parties = nom.split('.')
            
            if len(parties) >= 6:
                domaine = parties[2]
                annee = int(parties[3])
                polluant = parties[4]
                type_mesure = parties[5]  # daymean, daymax
                
                # Clé principale par polluant et type
                cle_principale = f"{polluant}_{type_mesure}"
                
                if cle_principale not in self.catalogue:
                    self.catalogue[cle_principale] = {}
                
                # Stocker par année
                self.catalogue[cle_principale][annee] = {
                    'fichier': fichier,
                    'domaine': domaine,
                    'annee': annee,
                    'polluant': polluant,
                    'type': type_mesure
                }
        
        print(f"✅ {len(fichiers)} fichiers catalogués")
        for cle, annees in self.catalogue.items():
            annees_list = sorted(annees.keys())
            print(f"   • {cle}: {len(annees)} fichier(s) - années {min(annees_list)}-{max(annees_list)}")
        
        return self.catalogue
    
    def trouver_indices_grille(self, lat_patient, lon_patient, lats_grille, lons_grille):
        """
        Trouve les 4 points de grille les plus proches du patient pour interpolation
        
        Args:
            lat_patient: lat du patient
            lon_patient: lon du patient
            lats_grille: Array des lats de la grille
            lons_grille: Array des lons de la grille
            
        Returns:
            Dictionnaire avec les indices et distances
        """
        # Trouver l'indice le plus proche
        idx_lat = np.argmin(np.abs(lats_grille - lat_patient))
        idx_lon = np.argmin(np.abs(lons_grille - lon_patient))
        
        # Vérifier si le point est dans les limites
        if (lat_patient < lats_grille.min() or lat_patient > lats_grille.max() or
            lon_patient < lons_grille.min() or lon_patient > lons_grille.max()):
            return None
        
        return {
            'idx_lat': idx_lat,
            'idx_lon': idx_lon,
            'lat_grille': lats_grille[idx_lat],
            'lon_grille': lons_grille[idx_lon]
        }
    
    def interpoler_bilineaire(self, lat_patient, lon_patient, lats_grille, lons_grille, valeurs):
        """
        Interpolation bilinéaire pour obtenir la valeur exacte au point du patient
        
        Args:
            lat_patient: lat du patient
            lon_patient: lon du patient
            lats_grille: Array des lats
            lons_grille: Array des lons
            valeurs: Array 2D des valeurs (lat, lon)
            
        Returns:
            Valeur interpolée
        """
        # Créer l'interpolateur
        interpolateur = RegularGridInterpolator(
            (lats_grille, lons_grille), 
            valeurs,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Interpoler au point du patient
        valeur = interpolateur([lat_patient, lon_patient])
        
        return valeur[0]
    
    def extraire_pour_patient(self, pseudo_provisoire, lat, lon, date_debut, date_fin, 
                              polluants_config=None):
        """
        Extrait les concentrations pour un patient sur une période donnée
        
        Args:
            pseudo_provisoire: Identifiant du patient
            lat: lat du patient
            lon: lon du patient
            date_debut: Date de début (datetime ou string 'YYYY-MM-DD')
            date_fin: Date de fin (datetime ou string 'YYYY-MM-DD')
            polluants_config: Dict spécifiant les polluants et types
                             Ex: {'NO2': 'daymean', 'O3': 'daymax', 'PM10': 'daymean'}
                             Si None, utilise daymean pour tous
            
        Returns:
            DataFrame avec les concentrations journalières
        """
        # Configuration par défaut
        if polluants_config is None:
            polluants_config = {
                'NO2': 'daymean',
                'O3': 'daymax',  # Pour O3, daymax est plus pertinent
                'PM10': 'daymean',
                'PM25': 'daymean'
            }
        
        # Convertir les dates si nécessaire
        if isinstance(date_debut, str):
            date_debut = pd.to_datetime(date_debut)
        if isinstance(date_fin, str):
            date_fin = pd.to_datetime(date_fin)
        
        # Déterminer les années à traiter
        annees_necessaires = range(date_debut.year, date_fin.year + 1)
        
        resultats = []
        
        for polluant, type_mesure in polluants_config.items():
            cle = f"{polluant}_{type_mesure}"
            
            if cle not in self.catalogue:
                print(f"{cle} non disponible")
                continue
            
            print(f"📂 Traitement {polluant} ({type_mesure}) pour patient {pseudo_provisoire}...")
            
            # Traiter chaque année
            for annee in annees_necessaires:
                if annee not in self.catalogue[cle]:
                    print(f"     Année {annee} non disponible pour {polluant}")
                    continue
                
                info_fichier = self.catalogue[cle][annee]
                fichier = info_fichier['fichier']
                
                try:
                    with nc.Dataset(fichier, 'r') as ds:
                        # Lire les coordonnées
                        lats = ds.variables['lat'][:]
                        lons = ds.variables['lon'][:]
                        times = ds.variables['Times'][:]
                        
                        # Convertir les temps en dates
                        time_units = ds.variables['Times'].units
                        time_calendar = getattr(ds.variables['Times'], 'calendar', 'standard')
                        dates = nc.num2date(times, units=time_units, calendar=time_calendar)
                        dates = pd.to_datetime([pd.Timestamp(d) for d in dates])
                        
                        # Filtrer les dates pour cette année
                        mask_dates = (dates >= date_debut) & (dates <= date_fin)
                        indices_dates = np.where(mask_dates)[0]
                        
                        if len(indices_dates) == 0:
                            print(f"     Aucune date dans la période pour année {annee}")
                            continue
                        
                        # Trouver les indices de grille
                        indices = self.trouver_indices_grille(lat, lon, lats, lons)
                        
                        if indices is None:
                            print(f"    Patient hors de la grille")
                            continue
                        
                        # Lire la variable de pollution
                        var_pollution = ds.variables[polluant]
                        
                        # Extraire les valeurs pour chaque date
                        for idx_date in indices_dates:
                            date_actuelle = dates[idx_date]
                            
                            # Extraire la tranche 2D pour cette date
                            valeurs_2d = var_pollution[idx_date, :, :]
                            
                            # Vérifier si c'est un masked array
                            if hasattr(valeurs_2d, 'mask'):
                                valeurs_2d = np.ma.filled(valeurs_2d, np.nan)
                            
                            # Interpoler au point du patient
                            concentration = self.interpoler_bilineaire(
                                lat, lon, lats, lons, valeurs_2d
                            )
                            
                            resultats.append({
                                'pseudo_provisoire': pseudo_provisoire,
                                'date': date_actuelle,
                                'lat': lat,
                                'lon': lon,
                                'polluant': polluant,
                                'type_mesure': type_mesure,
                                'concentration': concentration,
                                'unite': 'ug/m3'
                            })
                        
                        print(f"   ✅ {len(indices_dates)} valeurs extraites pour {polluant} ({annee})")
                
                except Exception as e:
                    print(f"   ❌ Erreur pour {polluant} ({annee}): {e}")
        
        return pd.DataFrame(resultats)
    
    def extraire_pour_tous_patients(self, df_patients, date_debut, date_fin,
                                     polluants_config=None):
        """
        Extrait les concentrations pour tous les patients d'un DataFrame
        
        Args:
            df_patients: DataFrame avec colonnes 'pseudo_provisoire', 'lat', 'lon'
            date_debut: Date de début
            date_fin: Date de fin
            polluants_config: Dict spécifiant les polluants et types
                             Ex: {'NO2': 'daymean', 'O3': 'daymax'}
            
        Returns:
            DataFrame combiné avec toutes les extractions
        """
        if polluants_config is None:
            polluants_config = {
                'NO2': 'daymean',
                'O3': 'daymax',
                'PM10': 'daymean',
                'PM25': 'daymean'
            }
        
        tous_resultats = []
        
        print(f"\n🚀 Extraction pour {len(df_patients)} patients")
        print(f"📅 Période: {date_debut} à {date_fin}")
        print(f"🏭 Polluants: {', '.join([f'{p} ({t})' for p, t in polluants_config.items()])}")
        print("=" * 80)
        
        for idx, row in df_patients.iterrows():
            pseudo_provisoire = row['pseudo_provisoire']
            lat = row['lat']
            lon = row['lon']
            
            print(f"\n[{idx+1}/{len(df_patients)}] Patient {pseudo_provisoire} (lat={lat:.4f}, lon={lon:.4f})")
            
            df_patient = self.extraire_pour_patient(
                pseudo_provisoire, lat, lon, date_debut, date_fin, 
                polluants_config
            )
            
            if not df_patient.empty:
                tous_resultats.append(df_patient)
        
        if tous_resultats:
            df_final = pd.concat(tous_resultats, ignore_index=True)
            print(f"\n✅ Extraction terminée: {len(df_final)} mesures au total")
            return df_final
        else:
            print("\n⚠️  Aucune donnée extraite")
            return pd.DataFrame()
    
    def pivoter_resultats(self, df_resultats):
        """
        Transforme le DataFrame long en format wide (une colonne par polluant)
        
        Args:
            df_resultats: DataFrame en format long
            
        Returns:
            DataFrame pivoté
        """
        df_pivot = df_resultats.pivot_table(
            index=['pseudo_provisoire', 'date', 'lat', 'lon'],
            columns='polluant',
            values='concentration',
            aggfunc='first'
        ).reset_index()
        
        return df_pivot


# ==============================================================================
# EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Configuration des chemins
    DOSSIER_NC = r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Data_temp"
    FICHIER_PATIENTS = r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Data\patients_geocoded_france.csv"  # À ADAPTER
    FICHIER_SORTIE = r"concentrations_patients.csv"
    
    # # 2. Créer un DataFrame exemple de patients
    # # REMPLACER PAR VOS VRAIES DONNÉES
    # df_patients_exemple = pd.DataFrame({
    #     'pseudo_provisoire': ['P001', 'P002', 'P003'],
    #     'lat': [48.8566, 45.7640, 43.6047],  # Paris, Lyon, Marseille
    #     'lon': [2.3522, 4.8357, 1.4442]
    # })
    
    df_patients_exemple = pd.read_csv(FICHIER_PATIENTS, sep=';')
    # OU charger depuis un fichier CSV:
    # df_patients = pd.read_csv(FICHIER_PATIENTS)
    # Assurez-vous que le CSV a les colonnes: pseudo_provisoire, lat, lon
    
    # 3. Initialiser l'extracteur
    extracteur = ExtracteurPollution(DOSSIER_NC)
    extracteur.cataloguer_fichiers()
    
    # 4. Définir la période d'extraction (peut couvrir plusieurs années)
    date_debut = '2016-01-01'
    date_fin = '2018-12-31'  # Exemple sur 3 ans
    
    # 5. Configuration des polluants et types de mesures
    # Pour O3, on utilise daymax (plus pertinent épidémiologiquement)
    # Pour les autres, on utilise daymean
    config_polluants = {
        'NO2': 'daymean',
        'O3': 'daymax',      # ⚠️ daymax pour O3
        'PM10': 'daymean',
        'PM25': 'daymean'
    }
    
    # Ou si vous voulez aussi O3 en daymean, vous pouvez extraire les deux:
    # config_polluants_complet = {
    #     'NO2': 'daymean',
    #     'O3': 'daymax',
    #     'PM10': 'daymean',
    #     'PM25': 'daymean'
    # }
    
    # 6. Extraire les données pour tous les patients
    df_concentrations = extracteur.extraire_pour_tous_patients(
        df_patients=df_patients_exemple,
        date_debut=date_debut,
        date_fin=date_fin,
        polluants_config=config_polluants
    )
    
    # 7. Optionnel: Pivoter pour avoir une colonne par polluant
    if not df_concentrations.empty:
        df_pivot = extracteur.pivoter_resultats(df_concentrations)
        
        # 8. Sauvegarder les résultats
        df_concentrations.to_csv(FICHIER_SORTIE, index=False)
        df_pivot.to_csv(FICHIER_SORTIE.replace('.csv', '_pivot.csv'), index=False)
        
        print(f"\n💾 Résultats sauvegardés:")
        print(f"   • Format long : {FICHIER_SORTIE}")
        print(f"   • Format pivot: {FICHIER_SORTIE.replace('.csv', '_pivot.csv')}")
        
        # 9. Afficher un aperçu
        print("\n📊 Aperçu des données (format long):")
        print(df_concentrations.head(10))
        
        print("\n📊 Aperçu des données (format pivot):")
        print(df_pivot.head())
        
        # 10. Statistiques par polluant
        print("\n📈 Statistiques par polluant:")
        stats = df_concentrations.groupby(['polluant', 'type_mesure'])['concentration'].describe()
        print(stats)
        
        # 11. Vérifier la couverture temporelle par patient
        print("\n📅 Couverture temporelle par patient:")
        couverture = df_pivot.groupby('pseudo_provisoire').agg({
            'date': ['min', 'max', 'count']
        })
        print(couverture)
        
        # 12. Optionnel: Extraire aussi O3 daymean si nécessaire
        # Si vous avez besoin des deux types de mesures pour O3:
        # config_o3_mean = {'O3': 'daymean'}
        # df_o3_mean = extracteur.extraire_pour_tous_patients(
        #     df_patients=df_patients_exemple,
        #     date_debut=date_debut,
        #     date_fin=date_fin,
        #     polluants_config=config_o3_mean
        # )
        # # Combiner avec les autres données
        # df_total = pd.concat([df_concentrations, df_o3_mean], ignore_index=True)
    
    print("\n✅ Traitement terminé!")