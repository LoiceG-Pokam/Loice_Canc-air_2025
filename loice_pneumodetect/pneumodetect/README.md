# PneumoDetect — LUNG-CANC'AIR

Analyse spatiale et épidémiologique des associations entre pollution atmosphérique, proximité industrielle et cancer du poumon en Île-de-France.

---

## Table des matières

1. [Contexte scientifique](#1-contexte-scientifique)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Données requises](#3-données-requises)
4. [Installation et prérequis](#4-installation-et-prérequis)
5. [Pipeline d'analyse — ordre d'exécution](#5-pipeline-danalyse--ordre-dexécution)
6. [Description détaillée des notebooks](#6-description-détaillée-des-notebooks)
7. [Modules Python (`src/`)](#7-modules-python-src)
8. [Pseudonymisation et confidentialité](#8-pseudonymisation-et-confidentialité)
9. [Principaux résultats](#9-principaux-résultats)
10. [Référence & Contact](#10-référence--contact)

---

## 1. Contexte scientifique

Ce projet s'inscrit dans le cadre du projet national **LUNG-CANC'AIR**, utilisant les donnees de la base poumon de Curie.

**Question de recherche :** Existe-t-il une association spatiale et/ou temporelle entre l'exposition à la pollution atmosphérique de fond (PM2.5, PM10, NO2, O3), la proximité aux sites industriels classés (ICPE Seveso), et la survenue de mutations oncogéniques dans les cancers du poumon chez les non-fumeurs en Île-de-France ?

**Population étudiée :**
- 3 404 patients atteints de cancer du poumon diagnostiqués entre 2018 et 2023
- Région Île-de-France uniquement
- Données cliniques, génomiques (panel de mutations) et géographiques, socio-demographiques

**Principaux polluants :** PM2.5, PM10, NO2, O3 (données INERIS, résolution 4k m × 4km)

**Analyses réalisées :**
- Régressions logistiques et de Poisson (exposition × mutation)
- Autocorrélation spatiale globale et locale (Moran's I, LISA)
- Moran's I bivarié (co-localisation mutations × Seveso)
- Analyses de sensibilité (strates de densité, corrections pour tests multiples)

---

## 2. Structure du dépôt

Structure effective après migration complète (juin 2026) :

```
loice_pneumodetect/                 ← Racine du dépôt git
│
├── pneumodetect/                   ← Code source et notebooks d'analyse
│   ├── 1-Restructuration_pseudonimisation.ipynb
│   ├── 3-a-description_cohorte.ipynb
│   ├── 3-b-ajout_radon.ipynb
│   ├── 4-analyse_air_final_v11.ipynb
│   ├── 5-analyse_autocorrelation.ipynb
│   ├── 5-6-analyse_spatiale_complete.ipynb
│   ├── 6-analyses_robustesse.ipynb
│   │
│   ├── geocodage/                  ← Notebooks de géocodage et analyse biais
│   │   ├── 2-geocodage.ipynb
│   │   ├── 01_textbiais_identification.ipynb
│   │   ├── 02_hotspot_identification.ipynb
│   │   ├── 03_ref_biaised_integration.ipynb
│   │   ├── 04_results_analysis.ipynb
│   │   ├── requirements.txt
│   │   ├── addok_bd/               ← Base Addok locale (gitignorée)
│   │   └── adresses-addok-france.ndjson.gz  ← BAN nationale (gitignorée)
│   │
│   ├── src/                        ← Modules Python réutilisables
│   │   ├── lungcancair_analyses.py
│   │   ├── lungcancair_engine_.py
│   │   ├── moran_spatial_analysis.py
│   │   ├── visualisations_cohorte.py
│   │   ├── visualisations_stats.py
│   │   ├── batch_fix.py
│   │   ├── filter_ndjson.py
│   │   └── addok_conf.py
│   │
│   └── html/                       ← Rendus HTML des notebooks (gitignorés)
│
├── data/                           ← ⚠ DONNÉES — gitignorées (RGPD)
│   ├── 0_brut/
│   │   ├── clinique/               (pneumodetectV3.xlsx, NIPs_adresses.xlsx, …)
│   │   ├── geospatial/             (cartes_geologiques/, icpe/, tmja/)
│   │   └── socio_eco/              (edi2021-iris-fm.xlsx, …)
│   ├── 1_patients/                 (patients_geocoded_*.csv, patients_radon_*.csv)
│   ├── 2_exposition/               (patients_exposition_metrique_*.csv)
│   ├── 3_analyses/                 (donnees_enrichies.csv, patients_clinics_air_metrics.csv)
│   └── 4_confidentiel/             ← ⚠ NIPs, adresses — accès DPO uniquement
│       ├── NIP_pneumodetect.csv
│       ├── table_correspondance_nip_pseudo.csv
│       └── … (16 fichiers)
│
├── geocodage/                      ← Config Addok + BAN IDF (données gitignorées)
│   ├── addok.conf
│   ├── addok_readme.md
│   ├── README - Moteur Addok.pdf
│   └── ban_idf/                    ← Fichiers BAN IDF (gitignorés — volumineux)
│       └── adresses-addok-11.ndjson
│
├── figures/                        ← Figures exportées par les notebooks
│   ├── descriptives/               (pyramide âges, tabac, mutations, expositions…)
│   │   ├── NO2/, O3/, PM10/, PM25/, geographie/
│   ├── statistiques/               (forest plots, heatmaps, comparaisons A1–A4)
│   └── resultats_moran/            (scatter Moran, LISA, bivarié…)
│
├── archives/                       ← Travaux antérieurs (non utilisés dans l'analyse)
│   ├── cancair_stage_ete_2024/
│   └── cancair_PFE_loice/
│
├── projet_qgis/                    ← Projet QGIS de visualisation cartographique
├── token_lungcancair/              ← Tokens d'accès (gitignorés)
├── litterature/                    ← Articles de référence (gitignorés)
├── .gitignore
└── README.md
```

> **Script externe d'extraction :** Le script principal d'extraction des données de qualité de l'air et de température est situé **en dehors** de ce dépôt :
> `R:\Direction_Data\0_Projets\Projet_CANCAIR\Canc_air_stage_ete_Loice\Airparif\main_extraction.py`
> Il lit les fichiers NetCDF INERIS/AIRPARIF, extrait les valeurs par patient, et alimente la base PostgreSQL. Voir [section 3.2](#32-données-de-pollution-atmosphérique-ineris--airparif) et [section 6 — main_extraction.py](#main_extractionpy-script-externe) pour les détails.

> **Données externes (R:) :** Les fichiers NetCDF INERIS/AIRPARIF et la base PostgreSQL sont sur le réseau Curie — voir [section 3.2](#32-données-de-pollution-atmosphérique-ineris--airparif) pour les chemins complets.

---

## 3. Données requises

Les données **ne sont pas incluses dans ce dépôt** pour des raisons de confidentialité (données de santé pseudonymisées). Elles doivent être obtenues auprès des sources indiquées ci-dessous.

### 3.1 Données cliniques (accès restreint)

| Fichier | Description | Source |
|---------|-------------|--------|
| `cohorte_brute_*.xlsx` | Données brutes patients (NIP, adresses, mutations) | Registre cancer IDF (accès sur demande) |
| `cohorte_finale_avec_adresses.csv` | 3 404 patients pseudonymisés — généré par notebook 1 | Produit par le pipeline |
| `patients_geocoded_idf.csv` | 2 993 patients géocodés (IDF) — généré par notebook 2 | Produit par le pipeline |
| `patients_radon_score_final.csv` | 1 682 patients avec score radon — généré par notebook 3b | Produit par le pipeline |

> **Important :** La table de correspondance `NIP ↔ pseudo_provisoire` doit être stockée **séparément** du reste des analyses et n'est pas incluse dans ce dépôt.

### 3.2 Données de pollution atmosphérique (INERIS / AIRPARIF)

| Fichier / Ressource | Description | Source |
|---------------------|-------------|--------|
| `*.nc` (NetCDF) — INERIS | Réanalyse journalière PM2.5, PM10, NO2, O3 (**2008–2023**), format `INERIS.REANALYSED.FRA0X.YYYY.POLLUANT.daymean.2gis.nc` | INERIS PREV'AIR / base interne |
| `*.nc` (NetCDF) — AIRPARIF | Données journalières + chroniques IDF, format `POLLUANT_maxJ_IDF_YYYYMMDD.nc` | Base AIRPARIF interne |
| Table PostgreSQL `pneumodetect_temp_air_ineris` | Séries temporelles pollution + température par patient (produit par `main_extraction.py`) | `sandbox-dev-postgres-data.curie.net` / base `sandbox` |
| `pneumodetect_temp_air_ineris.csv` | Export CSV de la table PostgreSQL | Produit par `main_extraction.py` / export manuel |

**Localisation des fichiers sources :**

| Source | Chemin réseau (R:) |
|--------|-------------------|
| NetCDF INERIS (2008–2023) | `R:\Direction_Data\0_Projets\Projet_CANCAIR\pneumodetect\data_air\` |
| NetCDF AIRPARIF (journalier + chronique) | `R:\Direction_Data\0_Projets\Projet_CANCAIR\data\airparif_dir\airparif\` |
| Température (parquet intermédiaire) | `R:\Direction_Data\0_Projets\Projet_CANCAIR\Canc_air_stage_ete_Loice\Airparif\output\patients_temp_daily.parquet` |
| Script d'extraction | `R:\Direction_Data\0_Projets\Projet_CANCAIR\Canc_air_stage_ete_Loice\Airparif\main_extraction.py` |

**Couverture INERIS :**
- **Temporelle :** 2008–2023 (réanalyse journalière, 1 fichier `.nc` par an et par polluant)
- **Spatiale :** France métropolitaine, résolution **0,00781° × 0,01563°** (~869 m × 1 199 m)
- **Polluants :** PM2.5, PM10, NO2, O3 (daymean) + O3 (daymax)

**Couverture AIRPARIF :**
- **Temporelle :** données journalières et chroniques IDF
- **Spatiale :** Île-de-France uniquement
- **Polluants :** PM2.5, PM10, NO2, O3 (journalier + exposition chronique)

#### Pipeline d'extraction air + température

```
[NetCDF INERIS]   [NetCDF AIRPARIF]   [temp_dataset_extraction.py]
  R:\...\data_air\   R:\...\airparif\        (parquet intermédiaire)
        │                 │                         │
        └────────┬─────────┘                        │
                 ▼                                  │
         main_extraction.py                         │
    (détection auto source,                         │
     extraction vectorisée,                         │
     checkpoints, années manquantes)                │
                 │                                  │
                 └──────────────┬───────────────────┘
                                ▼
               PostgreSQL: pneumodetect_temp_air_ineris
               (sandbox-dev-postgres-data.curie.net)
                                │
                                ▼
               pneumodetect_temp_air_ineris.csv
```

**Colonnes produites :**

| Colonne | Description | Source |
|---------|-------------|--------|
| `pm10` | Particules ≤10 µm (µg/m³, daymean) | INERIS |
| `pm25` | Particules ≤2,5 µm (µg/m³, daymean) | INERIS |
| `no2` | Dioxyde d'azote (µg/m³, daymean) | INERIS |
| `o3` | Ozone (µg/m³, daymean) | INERIS |
| `o3_maxj` | Ozone maximum journalier (µg/m³) | INERIS |
| `pm10_chron`, `pm25_chron`, `no2_chron`, `o3_chron` | Exposition chronique (AIRPARIF uniquement) | AIRPARIF |
| `temperature` | Température quotidienne (°C) | Script dédié |

### 3.3 Données géospatiales (libres)

| Fichier / Source | Description | Téléchargement |
|------------------|-------------|----------------|
| Fonds de carte IGN (CONTOURS-IRIS) | Limites IRIS Île-de-France | [IGN Géoplateforme](https://geoservices.ign.fr/) |
| BRGM — carte géologique 1/50 000 | 8 495 polygones géologiques IDF | [Infoterre BRGM](https://infoterre.brgm.fr/) |
| ICPE Basias/Géorisques | Sites Seveso SH (37) et SB (70) IDF | [Géorisques](https://www.georisques.gouv.fr/) |
| INSEE — Indice de Défavorisation (EDI) | Quintiles EDI 2021 par IRIS | [INSEE](https://www.insee.fr/) |

### 3.4 API de géocodage

Le notebook 2 utilise une instance locale d'**Addok** pour le géocodage des adresses françaises.

```bash
# Lancer un serveur Addok local 

pour le faire lire le readme de lancement addok "README - Moteur Addok.pdf" situé dans le repertoire "Loice_Canc-air_2025\loice_pneumodetect\geocodage"

addok serve
```

Un serveur Addok distant accessible en réseau peut aussi être configuré dans `src/addok_conf.py`.

---

## 3.5 Organisation des données — Data Management

Les données sont réparties sur deux emplacements :
- **`H:\PFE Loice\...\loice_pneumodetect\data\`** — fichiers de travail locaux (gitignorés)
- **`R:\Direction_Data\0_Projets\Projet_CANCAIR\`** — données sources NetCDF et base PostgreSQL

> **Règle absolue :** Aucune donnée patient ne doit jamais être versionnée sur git. Le dossier `data/` est intégralement dans `.gitignore`.

### Hiérarchie effective (`data/` local — H:)

```
data/                                  ← gitignorée — JAMAIS dans git
│
├── 0_brut/                            ← Données sources — NE JAMAIS MODIFIER
│   ├── clinique/                      ← CONFIDENTIEL
│   │   ├── pneumodetectV3.xlsx        (données cliniques + mutations brutes)
│   │   ├── NIPs_adresses.xlsx         (adresses nominatives — ⚠ SENSIBLE)
│   │   ├── BasePOUMON-PNEUMODETECTDONNEES_V1.csv
│   │   ├── base_poumon_adresses_V1.csv
│   │   ├── data_dictionnary_base_poumon.csv
│   │   ├── données base poumon.xlsx
│   │   ├── pneumodetect 18-12-2025.xlsx
│   │   └── pneumodetect avec date diag.xlsx
│   │
│   ├── geospatial/
│   │   ├── cartes_geologiques/        (GEO050K_HARM_075..095 — BRGM)
│   │   ├── icpe/                      (icpe_idf.shp + icpe.geojson — Géorisques)
│   │   └── tmja/                      (TMJA_RRNc_2024.shp — trafic routier)
│   │
│   └── socio_eco/
│       ├── edi2021-iris-fm.xlsx       (Indice de Défavorisation — INSEE)
│       ├── type_histo_groupe.csv
│       └── classification_histologique_revueCB.xlsx
│
├── 1_patients/                        ← CONFIDENTIEL — pipeline patients
│   ├── patients_geocoded_france.csv   (sortie notebook 2 — tous patients)
│   ├── patients_geocoded_idf.csv      (sortie notebook 2 — IDF uniquement)
│   ├── patients_geocoded_clean_idf_2018_2023.csv
│   ├── patients_geocoded_clean_idf_2018_2023_groups.csv
│   ├── patients_radon_lithologie.csv  (sortie notebook 3b)
│   ├── patients_radon_score_final.csv (sortie notebook 3b)
│   ├── patients_exclus.csv
│   ├── patient_chunk_0.csv
│   └── patient_chunk_0_geocoded.csv
│
├── 2_exposition/                      ← Données d'exposition calculées
│   ├── patients_exposition_metrique_after_2018.csv
│   └── patients_exposition_metrique_after_2018_2023.csv
│
├── 3_analyses/                        ← Table de travail principale
│   ├── donnees_enrichies.csv
│   ├── patients_clinics_air_metrics.csv
│   └── patients_clinics_air_metrics_groupes.csv
│
└── 4_confidentiel/                    ← ⚠ JAMAIS versionné — accès DPO uniquement
    ├── NIP_pneumodetect.csv           (liste des NIPs — clé nominative)
    ├── table_correspondance_nip_pseudo.csv  (clé de dépseudonymisation)
    ├── cohorte_finale_avec_adresses.csv
    ├── df_pseudo_provisoire_adresses.csv
    ├── donnees_pseudonymisees.csv
    ├── nips_communs.csv
    ├── nips_sans_adresse.csv
    ├── nips_sans_info_clinique.csv
    ├── patients_adresse_incomplete.csv
    ├── patients_adresses_incompletes.csv
    ├── patients_exclus_sans_adresse.csv
    ├── patients_sans_adresse.csv
    ├── patients_sans_date_diagnostic.csv
    ├── pseudo_provisoires_adresses.csv
    ├── table_correspondance.csv
    ├── donnees_temp_air.csv
    └── tracking_parcours_patients.csv
```

### Données externes — réseau R: (non déplacées, en lecture seule)

| Ressource | Chemin réseau |
|-----------|---------------|
| NetCDF INERIS (2008–2023) | `R:\Direction_Data\0_Projets\Projet_CANCAIR\pneumodetect\data_air\` |
| NetCDF AIRPARIF | `R:\Direction_Data\0_Projets\Projet_CANCAIR\data\airparif_dir\airparif\` |
| Parquet température | `R:\...\Canc_air_stage_ete_Loice\Airparif\output\patients_temp_daily.parquet` |
| Script extraction | `R:\...\Canc_air_stage_ete_Loice\Airparif\main_extraction.py` |
| Base PostgreSQL | `sandbox-dev-postgres-data.curie.net` → base `sandbox`, table `pneumodetect_temp_air_ineris` |

### Flux de données par notebook

```
pneumodetectV3.xlsx + NIPs_adresses.xlsx   [0_brut/clinique/]
        │
        ▼ notebook 1
cohorte_finale_avec_adresses.csv            [4_confidentiel/]
table_correspondance_nip_pseudo.csv         [4_confidentiel/]
        │
        ▼ notebook 2
patients_geocoded_france.csv                [1_patients/]
patients_geocoded_idf.csv                   [1_patients/]
patients_geocoded_clean_idf_2018_2023.csv   [1_patients/]
        │
        ├── notebook 3a → stats descriptives (pas de fichier sortie dédié)
        │
        └── notebook 3b
                └── patients_radon_lithologie.csv      [1_patients/]
                └── patients_radon_score_final.csv     [1_patients/]
        │
        ├── [EXTERNE] main_extraction.py
        │       └── pneumodetect_temp_air_ineris.csv   [2_exposition/]
        │
        ▼ notebook 4
pneumodetect_cohorte_idf_ineris_data.csv    [3_analyses/]
patients_geocoded_clean_idf_2018_2023_groups.csv  [1_patients/]
patients_exposition_metrique_after_2018.csv [2_exposition/]
        │
        ▼ notebooks 5 / 5-6
synthese_moran.csv, patients_moran.csv      [5_sorties/resultats_moran/]
cartes, forest plots (PNG)                  [5_sorties/figures/]
        │
        ▼ notebook 6
résultats robustesse (CSV + PNG)            [5_sorties/]
```

### Règles de gestion

| Règle | Détail |
|-------|--------|
| Jamais de modification des bruts | `0_brut/` est en lecture seule — toute correction se fait dans le notebook |
| Séparation confidentiel / analytique | `4_confidentiel/` ne contient que les tables avec NIP ou adresses non géocodées |
| Chemin unique par fichier | Chaque fichier intermédiaire n'existe qu'en un seul endroit (pas de copies locales `H:` désynchronisées) |
| Nommage daté pour les exports | Préfixer les sorties avec `YYYYMMDD_` si plusieurs versions doivent coexister |
| `.gitignore` couvre tout `data/` | Le dossier de données ne doit jamais apparaître dans `git status` |

---

## 4. Installation et prérequis

### 4.1 Python

Python **3.10+** recommandé (testé sur 3.12).

### 4.2 Dépendances

Installer les paquets requis :

```bash
pip install pandas numpy scipy scikit-learn statsmodels \
            geopandas shapely pyproj fiona \
            esda libpysal pymannkendall \
            xarray netCDF4 h5netcdf \
            matplotlib seaborn plotly \
            jupyter notebook ipykernel
```

Ou avec un fichier `requirements.txt` (à créer si besoin) :

```bash
pip install -r requirements.txt
```

### 4.3 Configuration du `sys.path`

Chaque notebook commence par ajouter `src/` au chemin Python. Vérifiez que le chemin est correct pour votre environnement :

```python
import sys
sys.path.append("../src")   # ou le chemin absolu vers pneumodetect/src/
```

### 4.4 Accès base de données (optionnel)

Certaines étapes initialement extraites d'une base **PostgreSQL** du registre. Les fichiers CSV intermédiaires peuvent être utilisés directement si les extractions ont déjà été faites.

---

## 5. Pipeline d'analyse — ordre d'exécution

Les notebooks doivent être exécutés **dans l'ordre numérique**. Chaque notebook produit des fichiers CSV utilisés par le suivant.

```
[Données brutes registre]
        │
        ▼
1-Restructuration_pseudonimisation.ipynb
        │  → cohorte_finale_avec_adresses.csv (3 404 patients)
        ▼
2-geocodage.ipynb
        │  → patients_geocoded_idf.csv (2 993 patients IDF)
        ▼
3-a-description_cohorte.ipynb          3-b-ajout_radon.ipynb
        │  (stats descriptives)                │  → patients_radon_score_final.csv
        │                                      │
        └──────────────┬───────────────────────┘
                       │
   ┌───────────────────┘
   │
   │  [Étape externe — à exécuter avant les notebooks suivants]
   ▼
main_extraction.py  (R:\...\Airparif\main_extraction.py)
   │  Entrée : patients_geocoded_idf.csv + fichiers NetCDF INERIS/AIRPARIF
   │  → PostgreSQL: pneumodetect_temp_air_ineris (pollution + température)
   │  → pneumodetect_temp_air_ineris.csv (export depuis PostgreSQL)
   ▼
4-analyse_air_final_v11.ipynb
                       │  (régressions logistiques/Poisson)
                       ▼
          5-analyse_autocorrelation.ipynb
          5-6-analyse_spatiale_complete.ipynb
                       │  (Moran's I, LISA, bivarié)
                       ▼
          6-analyses_robustesse.ipynb
                       │  (sensibilité, tests multiples)
```

---

## 6. Description détaillée des notebooks

### `1-Restructuration_pseudonimisation.ipynb`
**Objectif :** Nettoyage, restructuration et pseudonymisation complète de la cohorte brute.

- Fusion des fichiers d'entrée (données cliniques, génomiques, adresses)
- Attribution d'un identifiant unique `pseudo_provisoire` (entier de 1 à 3 404)
- Suppression définitive des NIPs (identifiants nominatifs)
- Standardisation des variables : histologie, stade, mutations, tabac
- **Sortie :** `cohorte_finale_avec_adresses.csv` (3 404 lignes, ~28 colonnes)

**Variables clés produites :**

| Variable | Description |
|----------|-------------|
| `pseudo_provisoire` | Identifiant unique pseudonymisé |
| `sexe` | M / F |
| `age_diagnostic` | Âge en années |
| `date_diagnostic` | Date du diagnostic |
| `stade` | I, II, III, IV, Non disponible |
| `histologie` | Adénocarcinome, Carcinome épidermoïde, etc. |
| `statut_tabac` | fumeur / non-fumeur |
| `pa` | Paquets-années |
| `EGFR`, `KRAS`, `BRAF`, … | Statut mutatif (0/1) pour 12+ gènes |
| `groupe_A`, `groupe_B`, … | Groupes de mutation (A=NF conducteurs, B=autres, C=NF, D=fumeurs) |

---

### `2-geocodage.ipynb`
**Objectif :** Géocodage des adresses en coordonnées GPS.

- Interrogation de l'API Addok (batch ou requête unitaire)
- Filtre géographique : Île-de-France uniquement (bbox)
- 9 patients non géocodés (adresses étrangères ou incomplètes)
- **Sortie :** `patients_geocoded_idf.csv` (2 993 patients, + colonnes `latitude`, `longitude`)

---

### `3-a-description_cohorte.ipynb`
**Objectif :** Description statistique complète de la cohorte.

- Pyramide des âges par sexe
- Distribution des stades et histologies
- Prévalence des mutations par gène et par groupe
- Exposition au tabac (paquets-années, catégories)
- Statistiques descriptives univariées et bivariées

---

### `3-b-ajout_radon.ipynb`
**Objectif :** Attribution d'un potentiel radon à chaque patient.

- Intersection spatiale adresses patients × carte géologique BRGM (1/50 000)
- Correspondance des formations géologiques IDF avec leur teneur en uranium (ppm U)
- Calcul d'un score radon standardisé (0–100)
- Formations à risque élevé : argiles plastiques sparnaciennnes (~4,0 ppm), argiles à silex
- **Sortie :** `patients_radon_score_final.csv` (1 682 patients)

---

### `main_extraction.py` (script externe)
**Chemin :** `R:\Direction_Data\0_Projets\Projet_CANCAIR\Canc_air_stage_ete_Loice\Airparif\main_extraction.py`

**Objectif :** Extraction en masse des données de qualité de l'air (INERIS ou AIRPARIF) et de température depuis les fichiers NetCDF vers PostgreSQL.

- **Détection automatique de la source** (INERIS ou AIRPARIF selon la structure des fichiers `.nc`)
- **Détection dynamique des années manquantes** : compare les années en base avec les fichiers disponibles et ne traite que ce qui est absent
- Extraction spatiale vectorisée : association patient × pixel le plus proche (lat/lon WGS84 pour INERIS, Lambert pour AIRPARIF)
- Traitement par batches avec **système de checkpoints** (reprise possible après interruption)
- **Module 1 — Pollution** : PM10, PM25, NO2, O3, O3-max journaliers → table `{table_name}_pollution`
- **Module 2 — Température** : lecture depuis `patients_temp_daily.parquet` (produit par `temp_dataset_extraction.py`) → table `{table_name}_temperature`
- **Module 3 — Fusion** : jointure pollution × température → table finale `pneumodetect_temp_air_ineris`

**Entrées :**

| Entrée | Chemin |
|--------|--------|
| Fichiers NetCDF INERIS | `R:\Direction_Data\0_Projets\Projet_CANCAIR\pneumodetect\data_air\` |
| Patients géocodés | `R:\Direction_Data\0_Projets\Projet_CANCAIR\pneumodetect\donnees patients\Data\patients_geocoded_idf.csv` |
| Données température (parquet) | `R:\...\Airparif\output\patients_temp_daily.parquet` |

**Sortie :** table PostgreSQL `pneumodetect_temp_air_ineris` dans la base `sandbox`.

**Exécution :**
```bash
cd R:\Direction_Data\0_Projets\Projet_CANCAIR\Canc_air_stage_ete_Loice\Airparif
python main_extraction.py
```

Le script demande confirmation avant tout traitement et permet de reprendre depuis un checkpoint en cas d'interruption.

---

### `4-analyse_air_final_v11.ipynb`
**Objectif :** Analyse principale — association pollution × mutations.

- Jointure cohorte géocodée + expositions air + score radon
- Régressions logistiques (outcome binaire mutation oui/non) et de Poisson (mutations rares)
- Ajustement sur : âge, sexe, tabac (paquets-années), EDI, densité de population
- Estimation des Odds Ratios par tertile/quartile d'exposition
- Fenêtres temporelles : 3, 5 et 10 ans avant le diagnostic
- Tests d'interaction sexe × exposition
- **Population d'analyse :** ~1 682 patients avec données air complètes

---

### `5-analyse_autocorrelation.ipynb`
**Objectif :** Autocorrélation spatiale (Moran's I).

- Moran's I global univarié (mutations NF, expositions air)
- LISA (Local Indicators of Spatial Association) : identification des clusters locaux HH/LL/HL/LH
- **Résultat clé :** I = -0,0081, p = 0,224 pour les mutations NF (pas de clustering global significatif)
- Cartographie des hotspots par arrondissement / commune

---

### `5-6-analyse_spatiale_complete.ipynb`
**Objectif :** Analyse spatiale complète et bivarié.

- Moran's I bivarié : co-localisation mutations NF × proximité Seveso SH
- **Résultat clé :** I = 0,0164, p = 0,023 (association spatiale significative)
- Identification des 64 patients en zone HH (mutations ET forte proximité Seveso)
- Visualisations cartographiques (cartes choroplèthes, scatter de Moran)

---

### `6-analyses_robustesse.ipynb`
**Objectif :** Tests de sensibilité et robustesse des résultats.

- Stratification par densité de population (rural / intermédiaire / urbain)
- Analyse par mutation individuelle (EGFR, KRAS, BRAF, ROS1, HER2, etc.)
- Variation du rayon de voisinage (1, 2, 3, 5, 10 km)
- Corrections pour tests multiples : Bonferroni et FDR (Benjamini-Hochberg)
- Sous-groupes : sexe, âge (<65 / ≥65), statut tabagique

---

## 7. Modules Python (`src/`)

| Module | Rôle principal |
|--------|---------------|
| `lungcancair_engine_.py` | Moteur d'analyse : `AnalyseConfig`, `run_analyse()`, `run_dose_reponse()` |
| `lungcancair_analyses.py` | Bibliothèque complète : description cohorte, régressions, visualisations, tests statistiques |
| `moran_spatial_analysis.py` | Autocorrélation spatiale : `charger_patients()`, `moran_global_univarie()`, `lisa_local()`, `moran_bivarie()` |
| `visualisations_cohorte.py` | Graphiques descriptifs : pyramide des âges, profils mutationnels, distributions tabac |
| `visualisations_stats.py` | Graphiques statistiques : forest plots, courbes ROC, scatter de Moran |
| `batch_fix.py` | Corrections de données : normalisation adresses, alignement temporel, valeurs manquantes |
| `filter_ndjson.py` | Filtrage GeoJSON/NDJSON pour les données ICPE |
| `addok_conf.py` | Configuration du géocodeur Addok (URL API, règles de normalisation) |

**Exemple d'utilisation du module spatial :**

```python
import sys
sys.path.append("src/")
from moran_spatial_analysis import charger_patients, moran_global_univarie, lisa_local

patients = charger_patients("patients_geocoded_idf.csv")
resultat = moran_global_univarie(patients, variable="groupe_A", rayon_km=3)
print(resultat)

lisa = lisa_local(patients, variable="groupe_A", rayon_km=3)
lisa.plot()
```

---

## 8. Pseudonymisation et confidentialité

Ce projet traite des **données de santé pseudonymisées** (catégorie sensible, RGPD Art. 9).

- Toutes les données nominatives (NIP, nom, prénom) ont été **supprimées** avant toute analyse
- Le seul identifiant utilisé est `pseudo_provisoire` (entier de 1 à 3 404)
- La table de correspondance `NIP ↔ pseudo_provisoire` est stockée **dans un espace sécurisé séparé**, non inclus dans ce dépôt
- Les adresses brutes ne sont **pas versionnées** dans ce dépôt
- Les fichiers de données patients ne doivent **jamais** être poussés sur un dépôt public

> Toute utilisation des données nécessite une autorisation préalable du registre et/ou du DPO de l'établissement.

---

## 9. Principaux résultats

> Ces résultats sont préliminaires (état : juin 2026) et ne constituent pas une publication.

| Analyse | Résultat |
|---------|---------|
| Association PM2.5 × mutations NF (régression) | OR modéré, signal plus consistant que PM10 / NO2 / O3 |
| Moran's I global — mutations NF | I = -0,0081, p = 0,224 (pas de clustering global) |
| Moran's I bivarié — mutations NF × Seveso SH | **I = 0,0164, p = 0,023** (co-localisation significative) |
| Patients en zone HH | **64 patients** en cluster haute mutation + haute proximité Seveso |
| Robustesse (rayons 1–10 km) | Signal stable pour Seveso SH, variable pour SB |
| Tests multiples (Bonferroni, 7 tests) | Seuil α corrigé = 0,0071 ; Moran bivarié reste significatif |

**Interprétation :** Les résultats suggèrent une **co-localisation spatiale** entre les mutations oncogéniques des non-fumeurs et la proximité aux sites Seveso à haut seuil, sans établir de causalité. Des analyses complémentaires (modèles INLA, analyse de médiation, données d'émissions) sont nécessaires.

---

## 10. Référence & Contact

**Auteure :** Loice Graciane Pokam Bopda  
**Encadrement :** Clemence Basse 
**Période :** 2025 – 2026  
**Cadre :** Projet LungCancair

Pour toute question sur les données ou les méthodes, contacter :
mail : pokamblg@gmail.com
tel: 33 758966513


---

*Ce README a été rédigé pour permettre à toute personne extérieure au projet de comprendre, reproduire et étendre les analyses réalisées dans le cadre de LUNG-CANC'AIR / PneumoDetect.*
