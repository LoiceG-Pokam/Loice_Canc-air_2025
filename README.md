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

Ce projet est réalisé dans le cadre d'une analyse statistique du projet pneumodetect IDF. Il s'inscrit dans le projet **LUNG-CANC'AIR**.

**Question de recherche :** Existe-t-il une association spatiale et/ou temporelle entre l'exposition à la pollution atmosphérique de fond (PM2.5, PM10, NO2, O3), la proximité aux sites industriels classés (ICPE Seveso), et la survenue de mutations dans les cancers du poumon chez les non-fumeurs en Île-de-France ?

**Population étudiée :**
- 3 404 patients atteints de cancer du poumon diagnostiqués entre 2008 et 2023
- Région Île-de-France uniquement
- Données cliniques, génomiques (panel de mutations) et géographiques, socio demographiques, 

**Principaux polluants :** PM2.5, PM10, NO2, O3 (données INERIS, résolution ~4 km × 4 km)

**Analyses réalisées :**
- Régressions logistiques et de Poisson (exposition × mutation)
- Autocorrélation spatiale globale et locale (Moran's I, LISA)
- Moran's I bivarié (co-localisation mutations × Seveso)
- Analyses de sensibilité (strates de densité, corrections pour tests multiples)

---

## 2. Structure du dépôt

```
loice_pneumodetect/
│
├── pneumodetect/                   ← Dossier principal du projet
│   ├── 1-Restructuration_pseudonimisation.ipynb
│   ├── 2-geocodage.ipynb
│   ├── 3-a-description_cohorte.ipynb
│   ├── 3-b-ajout_radon.ipynb
│   ├── 4-analyse_air_final_v11.ipynb
│   ├── 5-6-analyse_spatiale_complete.ipynb
│   ├── EXTRACTION_AIR.ipynb        ← Extraction données NetCDF air
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
│   └── html/                       ← Rendus HTML des notebooks (résultats)
│
├── figures/                        ← Figures exportées
├── projet_qgis/                    ← Projet QGIS de visualisation cartographique
├── token_lungcancair/              ← Tokens d'accès (ne pas versionner)
└── README.md                       ← Ce fichier
```

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

### 3.2 Données de pollution atmosphérique (INERIS)

| Fichier | Description | Source |
|---------|-------------|--------|
| `*.nc` (NetCDF) | Réanalyse journalière PM2.5, PM10, NO2, O3 (2016–2023) | [INERIS PREV'AIR](https://www.prevair.org/) ou base INERIS interne |
| `pneumodetect_temp_air_ineris.csv` | Séries temporelles par patient — généré par `EXTRACTION_AIR.ipynb` | Produit par le pipeline |

Résolution spatiale : **0,00781° × 0,01563°** (~869 m × 1 199 m). Couverture : France métropolitaine.

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
# Lancer un serveur Addok local (si non disponible en réseau)
# Voir https://addok.readthedocs.io/
addok serve
```

Un serveur Addok distant accessible en réseau peut aussi être configuré dans `src/addok_conf.py`.

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
                       ▼
          EXTRACTION_AIR.ipynb  (sur données NetCDF INERIS)
                       │  → pneumodetect_temp_air_ineris.csv
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

### `EXTRACTION_AIR.ipynb`
**Objectif :** Extraction des données de pollution atmosphérique depuis les fichiers NetCDF INERIS.

- Lecture des fichiers `.nc` (PM2.5, PM10, NO2, O3) journaliers 2008–2023
- Association spatiale patient × pixel INERIS le plus proche (lat/lon)
- Calcul des expositions cumulées sur des fenêtres glissantes (3, 5, 10 ans)
- Statistiques Mann-Kendall (tendance temporelle)
- **Sortie :** `pneumodetect_temp_air_ineris.csv`

> Ce notebook est **indépendant** et peut être exécuté en parallèle des notebooks 3a/3b.

---

### `4-analyse_air_final.ipynb`
**Objectif :** Analyse principale — association pollution × mutations.

- Jointure cohorte géocodée + expositions air + score radon
- Régressions logistiques (outcome binaire mutation oui/non) et de Poisson (mutations rares)
- Ajustement sur : âge, sexe, tabac (paquets-années), EDI, densité de population
- Estimation des Odds Ratios par tertile/quartile d'exposition
- Fenêtres temporelles : 3, 5 et 10 ans avant le diagnostic
- Tests d'interaction sexe × exposition
- **Population d'analyse :** ~1 682 patients avec données air complètes

---

### `5-6-analyse_spatiale_complete.ipynb`
**Objectif :** Analyse spatiale complète et bivarié.

- Moran's I bivarié : co-localisation mutations NF × proximité Seveso SH
- **Résultat clé :** I = 0,0164, p = 0,023 (association spatiale significative)
- Identification des 64 patients en zone HH (mutations ET forte proximité Seveso)
- Visualisations cartographiques (cartes choroplèthes, scatter de Moran)


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

**Auteure :** Loice Graciane  Pokam Bopda 
**Encadrement :** (à compléter selon l'établissement)  
**Période :** 2025 – 2026  
**Cadre :** Projet LungCancair — Santé publique / Biostatistiques

Pour toute question sur les données ou les méthodes, contacter le responsable du registre cancer Île-de-France.

---

*Ce README a été rédigé pour permettre à toute personne extérieure au projet de comprendre, reproduire et étendre les analyses réalisées dans le cadre de LUNG-CANC'AIR / PneumoDetect.*
