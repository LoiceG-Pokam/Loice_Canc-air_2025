"""
═══════════════════════════════════════════════════════════════════════════════
LUNG-CANC'AIR — Bibliothèque complète
═══════════════════════════════════════════════════════════════════════════════
Ce fichier contient TOUT le code analytique du projet.
Le notebook se réduit à : chemins + CFG + appels de fonctions.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict
from tqdm import tqdm
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
import pymannkendall as mk
import geopandas as gpd

plt.style.use('seaborn-v0_8-whitegrid')

# ── Constantes internes ────────────────────────────────────────────────────────
POLLUANTS_DEFAUT   = ['PM25', 'PM10', 'NO2', 'O3']
FENETRE_MAX_MOIS   = 120
POLLUANT_PRINCIPAL = 'PM25'
SEUIL_STABILITE    = 0.2
NB_MOIS_MIN_MK     = 24
MUTATIONS_NF       = ['EGFR', 'ALK', 'ROS1', 'RET', 'NTRK', 'ERBB2', 'MET']
COULEURS = {'AB':'#2E86AB','CD':'#52B788','ACBD':'#7B2D8B',
            'EGFR':'#2E86AB','MET':'#52B788','ALK':'#E76F51',
            'ERBB2':'#7B2D8B','ROS1':'#888888'}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHARGEMENT & CONSTRUCTION DE df_final
# ═══════════════════════════════════════════════════════════════════════════════


def detecter_vars_cumul(df_final, cfg=None):
    """
    Détecte les colonnes cumul disponibles selon la fenêtre et les polluants retenus.
    Si cfg fourni, filtre selon cfg['polluants_retenus'].
    Si 'IPG' est dans polluants_retenus, ajoute 'IPG' à la liste retournée.
    """
    polluants = cfg.get('polluants_retenus', ['PM25','PM10','O3']) if cfg else ['PM25','PM10','O3']
    cols = []
    for pol in polluants:
        if pol == 'IPG':
            # IPG n'a pas de colonne _cumul_ — c'est directement la colonne 'IPG'
            if 'IPG' in df_final.columns:
                cols.append('IPG')
            continue
        matches = sorted([c for c in df_final.columns
                   if c.startswith(f'{pol}_cumul_') and 'pct' not in c and 'manq' not in c])
        if matches:
            cols.append(matches[0])
    print(f"Variables cumul détectées ({polluants}) : {cols}")
    return cols


def charger_donnees(chemins):
    """
    Charge les données brutes depuis les chemins définis dans CHEMINS.
    Retourne (data, df_clinique).
    """
    data = (
        pd.read_csv(chemins['pollution'], sep=";", dtype=str, keep_default_na=False)
        .rename(columns={"O3_maxJ": "O3"})
    )
    data = data.sort_values(['pseudo_provisoire','date']).reset_index(drop=True)
    data['date']              = data['date'].values.astype('datetime64[ns]')
    data['pseudo_provisoire'] = data['pseudo_provisoire'].astype(int)
    for col in ['PM25','PM10','NO2','O3']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    df_clinique = pd.read_csv(chemins['clinique'], dtype=str, keep_default_na=False)
    df_clinique['date_diagnostic']   = df_clinique['date_diagnostic'].values.astype('datetime64[ns]')
    df_clinique['pseudo_provisoire'] = df_clinique['pseudo_provisoire'].astype(int)
    for col in ['age_diagnostic','paquet_annee']:
        if col in df_clinique.columns:
            df_clinique[col] = pd.to_numeric(df_clinique[col], errors='coerce')

    print(f"✅ data        : {data['pseudo_provisoire'].nunique():,} patients")
    print(f"✅ df_clinique : {df_clinique['pseudo_provisoire'].nunique():,} patients")
    return data, df_clinique


def _calculer_tendance_centrale(serie, nom):
    s = serie[~np.isnan(serie)]
    if len(s) == 0:
        return {f'{nom}_moyenne':np.nan,f'{nom}_mediane':np.nan,
                f'{nom}_max':np.nan,f'{nom}_min':np.nan,f'{nom}_ecart_moy_med':np.nan}
    moy,med = np.mean(s),np.median(s)
    return {f'{nom}_moyenne':moy,f'{nom}_mediane':med,
            f'{nom}_max':np.max(s),f'{nom}_min':np.min(s),f'{nom}_ecart_moy_med':moy-med}

def _calculer_variabilite(serie, nom):
    s = serie[~np.isnan(serie)]
    if len(s) == 0:
        return {f'{nom}_std':np.nan,f'{nom}_cv':np.nan,f'{nom}_iqr':np.nan}
    std,moy = np.std(s),np.mean(s)
    q75,q25 = np.percentile(s,[75,25])
    return {f'{nom}_std':std,f'{nom}_cv':std/moy if moy>0 else np.nan,f'{nom}_iqr':q75-q25}

def _calculer_exposition_cumulee(serie, nom, fenetre_mois=120):
    s = serie[~np.isnan(serie)]
    n_total,n_valides = len(serie),len(s)
    pct = (n_total-n_valides)/n_total*100 if n_total>0 else np.nan
    suffix = f"{fenetre_mois}m"
    if n_valides == 0:
        return {f"{nom}_cumul_{suffix}":np.nan, f"{nom}_cumul_pct_manq":np.nan}
    return {f"{nom}_cumul_{suffix}":np.nansum(serie), f"{nom}_cumul_pct_manq":pct}

def _calculer_pct_seuils(serie, nom, seuils):
    s = serie[~np.isnan(serie)]
    if len(s) == 0:
        return {f'{nom}_pct_sup{seuil}':np.nan for seuil in seuils}
    return {f'{nom}_pct_sup{seuil}':round(float((s>seuil).sum())/len(s)*100,2) for seuil in seuils}

def _calculer_tendance_mm_mk(df_patient, pol, col_date='date'):
    nan_result = {f'{pol}_mm365_sen_pente':np.nan,f'{pol}_mm365_mk_tau':np.nan,
                  f'{pol}_mm365_mk_tendance':np.nan}
    if pol not in df_patient.columns:
        return nan_result
    serie = (df_patient[[col_date,pol]].copy()
             .assign(**{col_date:lambda d:pd.to_datetime(d[col_date])})
             .set_index(col_date)[pol].sort_index())
    serie_lissee = serie.rolling(window=365,center=True,min_periods=180).mean().dropna()
    if len(serie_lissee) < NB_MOIS_MIN_MK:
        return nan_result
    try:
        res = mk.original_test(serie_lissee.values.astype(float))
    except Exception:
        return nan_result
    pente_an = float(res.slope*365)
    tendance = 'montée' if pente_an>SEUIL_STABILITE else ('descente' if pente_an<-SEUIL_STABILITE else 'stable')
    return {f'{pol}_mm365_sen_pente':round(pente_an,4),
            f'{pol}_mm365_mk_tau':round(float(res.Tau),4),
            f'{pol}_mm365_mk_tendance':tendance}

def _calculer_IPG(df_expo):
    pols_ipg = [p for p in POLLUANTS_DEFAUT if p != POLLUANT_PRINCIPAL]
    df = df_expo.copy(); cols_z = []
    for pol in pols_ipg:
        col_moy,col_z = f'{pol}_moyenne',f'z_{pol}'
        if col_moy not in df.columns: continue
        mu,sigma = df[col_moy].mean(),df[col_moy].std()
        df[col_z] = np.nan if (sigma==0 or pd.isna(sigma)) else (df[col_moy]-mu)/sigma
        cols_z.append(col_z)
    df['IPG'] = df[cols_z].mean(axis=1) if cols_z else np.nan
    return df


def calculer_variables_air(data, df_clinique, cfg):
    """
    Calcule toutes les variables d'exposition pour chaque patient.
    cfg doit contenir :
        'seuils_pct'       : {'PM25':[...],'PM10':[...],'O3':[...]}
        'fenetre_ans'      : années rétrospectives (défaut=10)
        'polluants_calcul' : polluants à calculer (défaut=tous les 4)
                             Ex: ['PM25','O3'] pour ne calculer que ces deux
    Colonnes créées selon la fenêtre :
        fenetre_ans=3  → PM25_cumul_36m
        fenetre_ans=10 → PM25_cumul_120m (défaut)
    """
    seuils_pct       = cfg.get('seuils_pct', {'PM25':[5,10,15,25,35],'PM10':[35,45,50,60,80,90],'O3':[100,120,180]})
    fenetre_ans      = cfg.get('fenetre_ans', 10)
    fenetre_mois     = int(fenetre_ans * 12)
    polluants_calcul = cfg.get('polluants_calcul', POLLUANTS_DEFAUT)
    print(f"Fenêtre : {fenetre_ans} an(s) ({fenetre_mois} mois) | Polluants calculés : {polluants_calcul}")

    patients_communs = set(data['pseudo_provisoire'].unique()) & set(df_clinique['pseudo_provisoire'].unique())
    print(f"Patients communs : {len(patients_communs)}")

    df_poll = data[data['pseudo_provisoire'].isin(patients_communs)].reset_index(drop=True)
    df_clin = df_clinique[df_clinique['pseudo_provisoire'].isin(patients_communs)].reset_index(drop=True)
    groupes_poll = df_poll.groupby('pseudo_provisoire')

    resultats, exclus = [], []
    for pseudo in tqdm(df_clin['pseudo_provisoire'].unique(), desc='Patients'):
        date_diag  = pd.Timestamp(df_clin.loc[df_clin['pseudo_provisoire']==pseudo,'date_diagnostic'].iloc[0])
        date_debut = date_diag - relativedelta(months=fenetre_mois)
        if pseudo not in groupes_poll.groups:
            exclus.append(pseudo); continue
        df_pat = df_poll.loc[groupes_poll.groups[pseudo]]
        df_pat = df_pat[(df_pat['date']>=date_debut)&(df_pat['date']<date_diag)].copy()
        if len(df_pat) == 0:
            exclus.append(pseudo); continue
        m = {'pseudo_provisoire':pseudo,'date_diagnostic':date_diag,'fenetre_mois':fenetre_mois}
        for pol in polluants_calcul:
            if pol not in df_pat.columns: continue
            serie = df_pat[pol].values
            m.update(_calculer_tendance_centrale(serie,pol))
            m.update(_calculer_variabilite(serie,pol))
            m.update(_calculer_exposition_cumulee(serie,pol,fenetre_mois))
            m.update(_calculer_tendance_mm_mk(df_pat,pol))
            if pol in seuils_pct:
                m.update(_calculer_pct_seuils(serie,pol,seuils_pct[pol]))
        resultats.append(m)

    df_air = _calculer_IPG(pd.DataFrame(resultats))
    if exclus: print(f"⚠️  {len(exclus)} patients exclus")
    cols_cumul = [c for c in df_air.columns if '_cumul_' in c and 'pct' not in c and 'manq' not in c]
    print(f"✅ df_air : {len(df_air)} patients × {len(df_air.columns)} variables")
    print(f"   Fenêtre : {fenetre_ans} an(s) | Colonnes cumul : {cols_cumul[:6]}")
    return df_air


def ajouter_socioeco(df_air, df_clinique, chemin_edi):
    """Fusionne les variables socio-économiques (EDI2021)."""
    df_socio = pd.read_excel(chemin_edi, sheet_name='EDI2021_IRIS')
    df_air = (df_air
              .merge(df_clinique[['pseudo_provisoire','CODE_IRIS']], on='pseudo_provisoire', how='left')
              .merge(df_socio[['IRIS','EDI2021','quintileEDI2021']], left_on='CODE_IRIS', right_on='IRIS', how='left'))
    n = df_air['EDI2021'].notna().sum()
    print(f"✅ EDI2021 : {n}/{len(df_air)} patients ({n/len(df_air)*100:.1f}%)")
    return df_air


def ajouter_routier(df_air, df_clinique, cfg):
    """
    Calcule les variables routières (dist_RN_m, indice_trafic, proche_RN_{X}m).
    cfg doit contenir 'shapefile_tmja' et 'buffer_rn_m'.
    """
    from shapely.geometry import Point
    TMJA_COL    = 'tmja'
    buffer_rn_m = cfg.get('buffer_rn_m', 500)
    shapefile   = cfg['shapefile_tmja']

    routes = gpd.read_file(shapefile)
    if routes.crs.to_epsg() != 2154:
        routes = routes.to_crs("EPSG:2154")
    routes[TMJA_COL] = (routes[TMJA_COL].astype(str).str.strip()
                        .str.replace('\u00a0','',regex=False).str.replace(' ','',regex=False)
                        .str.replace(',','.',regex=False).replace('',float('nan')))
    routes[TMJA_COL] = pd.to_numeric(routes[TMJA_COL], errors='coerce')

    df_geo = df_clinique[['pseudo_provisoire','x','y']].copy()
    df_geo['x'] = pd.to_numeric(df_geo['x'],errors='coerce')
    df_geo['y'] = pd.to_numeric(df_geo['y'],errors='coerce')
    gdf_patients = gpd.GeoDataFrame(df_geo,
        geometry=gpd.points_from_xy(df_geo['x'],df_geo['y']),crs='EPSG:4326').to_crs('EPSG:2154')

    gdf_joined = gpd.sjoin_nearest(gdf_patients, routes[[TMJA_COL,'geometry']],
        how='left', distance_col='dist_RN_m').pipe(lambda d: d[~d.index.duplicated(keep='first')])
    gdf_joined['dist_RN_m']       = gdf_joined['dist_RN_m'].round(1)
    gdf_joined                    = gdf_joined.rename(columns={TMJA_COL:'TMJA_proche'})
    gdf_joined['TMJA_proche_log'] = np.log1p(gdf_joined['TMJA_proche'])
    gdf_joined['indice_trafic']   = (gdf_joined['TMJA_proche_log']/np.log1p(gdf_joined['dist_RN_m']+1)).round(4)
    col_proche = f'proche_RN_{buffer_rn_m}m'
    gdf_joined[col_proche]        = (gdf_joined['dist_RN_m'] < buffer_rn_m).astype(int)

    cols = ['pseudo_provisoire','dist_RN_m','TMJA_proche','TMJA_proche_log',col_proche,'indice_trafic']
    df_air = df_air.merge(gdf_joined[cols], on='pseudo_provisoire', how='left')
    print(f"✅ Routier : dist_RN_m, indice_trafic, {col_proche}")
    return df_air, gdf_patients


def ajouter_icpe(df_air, gdf_patients, cfg):
    """
    Calcule les variables ICPE dynamiquement.
    cfg doit contenir :
        'icpe_path'    : chemin vers le shapefile ICPE
        'rayons_icpe_m': liste de rayons (ex: [3000, 5000])
        'icpe_types'   : dict {'SH':'SH','SB':'SB','NS':'NS'} (vide = total uniquement)
    """
    icpe_path    = cfg['icpe_path']
    rayons       = cfg.get('rayons_icpe_m', [3000, 5000])
    icpe_types   = cfg.get('icpe_types', {'NS':'NS','SB':'SB','SH':'SH'})

    icpe = gpd.read_file(icpe_path).to_crs('EPSG:2154')
    print(f"ICPE chargées : {len(icpe)}")
    print(icpe['seveso'].value_counts(dropna=False).to_string())

    # Sous-ensembles
    icpe_subsets = {lbl: icpe[icpe['seveso']==val].reset_index(drop=True)
                    for lbl,val in icpe_types.items()}

    cols_produites = []

    # Distance aux sites les plus proches (SH et SB toujours)
    gdf_pts = gdf_patients[['pseudo_provisoire','geometry']].copy()
    for type_label in ['SH','SB']:
        col_dist = f'dist_ICPE_{type_label}_m'
        # Utiliser le sous-ensemble si disponible, sinon filtrer depuis icpe
        subset = icpe_subsets.get(type_label)
        if subset is None or len(subset) == 0:
            subset = icpe[icpe['seveso'] == type_label].reset_index(drop=True)
        if len(subset) > 0:
            j = gpd.sjoin_nearest(gdf_pts, subset[['geometry']], how='left', distance_col=col_dist)
            j = j[~j.index.duplicated(keep='first')]
            gdf_patients[col_dist] = j[col_dist].round(1).values
            print(f"  {col_dist} — médiane : {gdf_patients[col_dist].median():.0f} m")
        else:
            gdf_patients[col_dist] = float('nan')
        cols_produites.append(col_dist)

    # Comptage par rayon
    for rayon in rayons:
        lbl = f'{rayon//1000}km' if rayon%1000==0 else f'{rayon}m'
        print(f"\n── Rayon {lbl} ──")
        buf = gdf_patients[['pseudo_provisoire','geometry']].copy()
        buf = buf.set_geometry(buf.geometry.buffer(rayon))

        # Total
        col_tot = f'nb_ICPE_total_{lbl}'
        j = gpd.sjoin(buf, icpe[['geometry']], how='left', predicate='contains')
        nb = j.groupby('pseudo_provisoire').size().reset_index(name=col_tot)
        gdf_patients = gdf_patients.merge(nb, on='pseudo_provisoire', how='left')
        gdf_patients[col_tot] = gdf_patients[col_tot].fillna(0).astype(int)
        cols_produites.append(col_tot)
        print(f"  {col_tot:<32} médiane={gdf_patients[col_tot].median():.0f} max={gdf_patients[col_tot].max():.0f}")

        # Par type
        for type_label, subset in icpe_subsets.items():
            col_t = f'nb_ICPE_{type_label}_{lbl}'
            if len(subset) > 0:
                j2 = gpd.sjoin(buf, subset[['geometry']], how='left', predicate='contains')
                nb2 = j2.groupby('pseudo_provisoire').size().reset_index(name=col_t)
                gdf_patients = gdf_patients.merge(nb2, on='pseudo_provisoire', how='left')
                gdf_patients[col_t] = gdf_patients[col_t].fillna(0).astype(int)
            else:
                gdf_patients[col_t] = 0
            cols_produites.append(col_t)
            print(f"  {col_t:<32} médiane={gdf_patients[col_t].median():.0f} max={gdf_patients[col_t].max():.0f}")

    cols_merge = ['pseudo_provisoire'] + cols_produites
    df_air = df_air.merge(gdf_patients[cols_merge], on='pseudo_provisoire', how='left')
    print(f"\n✅ {len(cols_produites)} variables ICPE : {cols_produites}")
    return df_air


def ajouter_radon(df_air, df_clinique, chemin_radon, cfg=None):
    """
    Fusionne le score radon géologique sur pseudo_provisoire.

    Le CSV attendu (ex: patients_radon_score_final.csv) doit contenir :
        - pseudo_provisoire : identifiant patient
        - radon_score       : score continu 0-100 basé sur la géologie
        - NOTATION          : code formation géologique (optionnel)
        - DESCR             : description formation géologique (optionnel)

    cfg peut contenir :
        'radon_var'      : nom exact de la colonne score radon
                           (défaut : auto-détection sur 'radon' dans le nom)
        'geo_notation'   : nom de la colonne formation géologique
                           (défaut : 'NOTATION' si présente)

    Variables créées dans df_air :
        radon_score    : score géologique continu (0–100)
        radon_bin      : binaire — 1 si radon_score > médiane de la cohorte
        radon_quartile : Q1 à Q4 selon la distribution de la cohorte
                         (utile pour l'analyse dose-réponse)
        formation_geo  : code de la formation géologique (si disponible)

    Note scientifique :
        Ce score est calculé à partir des teneurs en uranium des formations
        géologiques du Bassin parisien. Les valeurs sont plus faibles que dans
        les zones granitiques (Bretagne, Massif Central), ce qui est cohérent
        avec un rôle de confondant MINEUR dans cette cohorte.
    """
    df_radon = pd.read_csv(chemin_radon, dtype=str, keep_default_na=False)
    df_radon['pseudo_provisoire'] = df_radon['pseudo_provisoire'].astype(int)

    # ── Détection de la colonne score radon ───────────────────────────────
    radon_var = cfg.get('radon_var') if cfg else None
    if radon_var is None:
        candidates = [c for c in df_radon.columns
                      if 'radon' in c.lower() and c != 'pseudo_provisoire']
        radon_var = candidates[0] if candidates else None
    if radon_var is None:
        print("⚠️  Colonne radon non trouvée dans le CSV — vérifier le fichier.")
        return df_air

    # ── Détection de la colonne formation géologique ──────────────────────
    # Priorité : cfg['geo_notation'], puis 'NOTATION', puis 'DESCR'
    geo_var = cfg.get('geo_notation') if cfg else None
    if geo_var is None:
        for candidate in ['NOTATION', 'notation', 'formation', 'DESCR']:
            if candidate in df_radon.columns:
                geo_var = candidate
                break

    # ── Sélection et renommage des colonnes utiles ─────────────────────────
    cols_merge = ['pseudo_provisoire', radon_var]
    if geo_var:
        cols_merge.append(geo_var)

    df_r = df_radon[cols_merge].copy().rename(
        columns={radon_var: 'radon_score', geo_var: 'formation_geo'} if geo_var
        else {radon_var: 'radon_score'}
    )
    df_r['radon_score'] = pd.to_numeric(df_r['radon_score'], errors='coerce')

    # ── Variables dérivées ─────────────────────────────────────────────────
    med = df_r['radon_score'].median()
    df_r['radon_bin'] = (df_r['radon_score'] > med).astype(float)

    # Quartiles : Q1 = faible exposition, Q4 = forte exposition
    # duplicates='drop' gère les ex-aequo fréquents dans ce type de score
    df_r['radon_quartile'] = pd.qcut(
        df_r['radon_score'], q=4,
        labels=['Q1_faible', 'Q2', 'Q3', 'Q4_élevé'],
        duplicates='drop'
    ).astype(str)

    # ── Fusion sur df_air ──────────────────────────────────────────────────
    df_air = df_air.merge(df_r, on='pseudo_provisoire', how='left')

    n_ok  = df_air['radon_score'].notna().sum()
    n_tot = len(df_air)
    print(f"✅ Radon — score géologique continu (0-100)")
    print(f"   Disponible : {n_ok}/{n_tot} patients ({n_ok/n_tot*100:.1f}%)")
    print(f"   Médiane cohorte = {df_air['radon_score'].median():.1f} | "
          f"Moyenne = {df_air['radon_score'].mean():.1f} | "
          f"Max = {df_air['radon_score'].max():.1f}")
    print(f"   radon_bin (> médiane) : {int(df_air['radon_bin'].sum())} patients exposés")
    print(f"   Distribution quartiles :")
    for q, n in df_air['radon_quartile'].value_counts(dropna=False).sort_index().items():
        print(f"      {q} : {n} patients")
    if geo_var:
        print(f"   Formations géologiques distinctes : {df_air['formation_geo'].nunique()}")
    print("⚠️  Note : cohorte en Bassin parisien → radon attendu comme confondant MINEUR")
    return df_air


def explorer_distribution_radon(df):
    """
    Visualise la distribution du score radon géologique par groupe.

    Produit 3 figures :
      1. Boxplots du score radon continu (0-100) par groupe (A/B, C/D, A+C/B+D)
         + test Mann-Whitney. C'est la figure principale car radon_score est continu.
      2. Distribution des quartiles de radon par groupe — barplot en % + Chi²
         (permet de voir si certains groupes sont sur-représentés en Q4)
      3. Top formations géologiques par groupe (si formation_geo disponible)
    """
    if 'radon_score' not in df.columns:
        print("⚠️  Colonne 'radon_score' absente — lancer ajouter_radon() d'abord.")
        return

    groupes = [
        ('groupe_AB',    'Groupe A',   'Groupe B',   '#2E86AB', '#A8DADC', 'A vs B — Mutations NF'),
        ('groupe_CD',    'Groupe C',   'Groupe D',   '#52B788', '#E76F51', 'C vs D — Non-fumeurs vs Fumeurs'),
        ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', '#7B2D8B', '#BBBBBB', 'A+C vs B+D'),
    ]

    # ── Figure 1 : Boxplots score radon continu ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Score radon géologique (0–100) par groupe', fontsize=13, fontweight='bold')

    for ax, (col_g, g1, g2, c1, c2, titre) in zip(axes, groupes):
        df_sub = df[df[col_g].isin([g1, g2])].dropna(subset=['radon_score'])
        if len(df_sub) == 0:
            ax.set_visible(False); continue
        s1 = df_sub.loc[df_sub[col_g] == g1, 'radon_score']
        s2 = df_sub.loc[df_sub[col_g] == g2, 'radon_score']
        _, p = mannwhitneyu(s1, s2, alternative='two-sided')
        p_txt  = '<0.001' if p < 0.001 else f'{p:.3f}'
        couleur_sig = 'red' if p < 0.05 else 'gray'
        palette = {g1: c1, g2: c2}
        sns.boxplot(data=df_sub, x=col_g, y='radon_score',
                    palette=palette, order=[g1, g2], ax=ax, width=0.5)
        ax.set_title(f'{titre}\np = {p_txt}', fontsize=10, fontweight='bold',
                     color=couleur_sig)
        ax.set_xlabel('')
        ax.set_ylabel('Score radon (0–100)')
        # Annotation médianes
        for g, c in [(g1, c1), (g2, c2)]:
            med = df_sub.loc[df_sub[col_g] == g, 'radon_score'].median()
            ax.text(0.5, med, f' Méd={med:.1f}', va='center', fontsize=8, color='black')
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ── Figure 2 : Distribution quartiles par groupe ─────────────────────
    if 'radon_quartile' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Distribution quartiles radon par groupe (%)',
                     fontsize=13, fontweight='bold')
        ordre_q = sorted(df['radon_quartile'].dropna().unique())

        for ax, (col_g, g1, g2, c1, c2, titre) in zip(axes, groupes):
            df_sub = df[df[col_g].isin([g1, g2])].dropna(subset=['radon_quartile'])
            if len(df_sub) == 0:
                ax.set_visible(False); continue

            tab = pd.crosstab(df_sub['radon_quartile'], df_sub[col_g])
            try:
                from scipy.stats import chi2_contingency
                _, p, _, _ = chi2_contingency(tab.values)
                p_txt = '<0.001' if p < 0.001 else f'{p:.3f}'
            except Exception:
                p_txt = 'NA'

            tab_pct = tab.div(tab.sum(axis=0), axis=1) * 100
            x     = np.arange(len(ordre_q))
            width = 0.35
            vals_g1 = [tab_pct.loc[q, g1] if q in tab_pct.index
                       and g1 in tab_pct.columns else 0 for q in ordre_q]
            vals_g2 = [tab_pct.loc[q, g2] if q in tab_pct.index
                       and g2 in tab_pct.columns else 0 for q in ordre_q]

            ax.bar(x - width/2, vals_g1, width, color=c1, label=g1, edgecolor='white')
            ax.bar(x + width/2, vals_g2, width, color=c2, label=g2, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(ordre_q, fontsize=8, rotation=15)
            ax.set_ylabel('% du groupe')
            ax.set_title(f'{titre}\nChi² p = {p_txt}', fontsize=10, fontweight='bold',
                         color='red' if p_txt not in ['NA'] and
                         (p_txt == '<0.001' or (p_txt != 'NA' and float(p_txt) < 0.05))
                         else 'gray')
            ax.legend(fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()

    # ── Figure 3 : Top formations géologiques ────────────────────────────
    if 'formation_geo' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Top 8 formations géologiques par groupe (%)',
                     fontsize=13, fontweight='bold')
        for ax, (col_g, g1, g2, c1, c2, titre) in zip(axes, groupes):
            df_sub = df[df[col_g].isin([g1, g2])].dropna(subset=['formation_geo'])
            if len(df_sub) == 0:
                ax.set_visible(False); continue
            tab = pd.crosstab(df_sub['formation_geo'], df_sub[col_g])
            # Garder les 8 formations les plus fréquentes
            top8 = tab.sum(axis=1).nlargest(8).index
            tab_pct = tab.loc[top8].div(tab.sum(axis=0), axis=1) * 100
            try:
                from scipy.stats import chi2_contingency
                _, p, _, _ = chi2_contingency(tab.values)
                p_txt = '<0.001' if p < 0.001 else f'{p:.3f}'
            except Exception:
                p_txt = 'NA'
            tab_pct.plot(kind='barh', ax=ax, color=[c1, c2], edgecolor='white', width=0.6)
            ax.set_title(f'{titre}\nChi² p = {p_txt}', fontsize=10, fontweight='bold')
            ax.set_xlabel('% du groupe')
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True, axis='x', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()


def analyse_sensibilite_radon(df, cfg):
    """
    Analyse de sensibilité : l'effet des polluants atmosphériques persiste-t-il
    après ajustement pour le radon ?

    Pour chaque groupe (A/B, C/D, A+C/B+D), lance deux modèles :
      - Modèle 1 (sans radon) : polluants + covariables cliniques
      - Modèle 2 (avec radon) : polluants + covariables cliniques + radon_categorie

    Compare les OR des polluants entre les deux modèles et calcule
    le % de changement de l'OR → si faible (< 10%), l'effet pollution
    est robuste au radon.

    Retourne un dict avec les deux tableaux et la comparaison.
    """
    if 'radon_score' not in df.columns:
        print("⚠️  'radon_score' absent — lancer ajouter_radon() d'abord.")
        return None

    GROUPES = [
        ('AB',   'groupe_AB',    'Groupe A',   'Groupe B',   True,  '#2E86AB', 'A vs B — Mutations NF'),
        ('CD',   'groupe_CD',    'Groupe C',   'Groupe D',   False, '#52B788', 'C vs D — Non-fumeurs vs Fumeurs'),
        ('ACBD', 'groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', False, '#7B2D8B', 'A+C vs B+D'),
    ]

    resultats = {}

    print("\n" + "█"*65)
    print("ANALYSE DE SENSIBILITÉ — AJUSTEMENT RADON")
    print("█"*65)
    print("Question : Les effets des polluants persistent-ils après ajustement pour le radon ?")
    print()

    for key, col_g, g1, g2, paquet, couleur, titre in GROUPES:
        print(f"\n{'='*60}\n{titre}\n{'='*60}")

        d = df[df[col_g].isin([g1, g2])].copy()
        d['outcome'] = (d[col_g] == g1).astype(int)

        # Variables du modèle de base (sans radon)
        vars_base = _vars_modele(cfg, inclure_paquet=paquet)
        vars_base = [v for v in vars_base if v in d.columns]

        # Variables du modèle avec radon — dédoublonner pour éviter les
        # colonnes dupliquées qui causent "truth value of a Series is ambiguous"
        vars_radon = list(dict.fromkeys(vars_base + ['radon_score']))
        vars_radon = [v for v in vars_radon if v in d.columns]

        # ── Modèle 1 : sans radon ──────────────────────────────────────────
        d1 = d[vars_base + ['outcome']].dropna()
        if len(d1) < 30 or d1['outcome'].sum() < 5:
            print(f"⚠️  Effectif insuffisant"); continue

        scaler = StandardScaler()
        # Reconstruction explicite du DataFrame standardisé pour éviter les
        # colonnes dupliquées dues à l'assignation en slice pandas
        arr1 = scaler.fit_transform(d1[vars_base])
        d1_std = pd.DataFrame(arr1, columns=vars_base, index=d1.index)
        d1_std['outcome'] = d1['outcome'].values
        vars_ok1 = [v for v in vars_base if float(d1_std[v].std()) > 1e-8]
        X1 = sm.add_constant(d1_std[vars_ok1]); y1 = d1_std['outcome']
        try:
            mod1 = sm.Logit(y1, X1).fit(disp=False)
            tab1 = _tableau_OR(mod1, vars_ok1)
        except Exception as e:
            print(f"  ⚠️  Modèle sans radon échoué : {e}"); continue

        # ── Modèle 2 : avec radon ──────────────────────────────────────────
        d2 = d[vars_radon + ['outcome']].dropna()
        if len(d2) < 30 or d2['outcome'].sum() < 5:
            print(f"⚠️  Effectif insuffisant après dropna avec radon"); continue

        scaler2 = StandardScaler()
        arr2 = scaler2.fit_transform(d2[vars_radon])
        d2_std = pd.DataFrame(arr2, columns=vars_radon, index=d2.index)
        d2_std['outcome'] = d2['outcome'].values
        vars_ok2 = [v for v in vars_radon if float(d2_std[v].std()) > 1e-8]
        X2 = sm.add_constant(d2_std[vars_ok2]); y2 = d2_std['outcome']
        try:
            mod2 = sm.Logit(y2, X2).fit(disp=False)
            tab2 = _tableau_OR(mod2, vars_ok2)
        except Exception as e:
            print(f"  ⚠️  Modèle avec radon échoué : {e}"); continue

        # ── Comparaison OR ─────────────────────────────────────────────────
        auc1 = roc_auc_score(y1, mod1.predict(X1))
        auc2 = roc_auc_score(y2, mod2.predict(X2))

        print(f"\n  AUC sans radon  = {auc1:.3f}")
        print(f"  AUC avec radon  = {auc2:.3f}  (Δ = {auc2-auc1:+.3f})")

        # Variables d'exposition (polluants) présentes dans les deux modèles
        vars_exp = [v for v in vars_ok1 if any(pol in v for pol in ['PM25','PM10','NO2','O3','IPG'])]

        rows_comp = []
        for var in vars_exp:
            if var not in tab1.set_index('Variable').index: continue
            if var not in tab2.set_index('Variable').index: continue
            or1 = tab1.set_index('Variable').loc[var, 'OR']
            or2 = tab2.set_index('Variable').loc[var, 'OR']
            p1  = tab1.set_index('Variable').loc[var, 'p-value']
            p2  = tab2.set_index('Variable').loc[var, 'p-value']
            delta_pct = (or2 - or1) / or1 * 100 if or1 != 0 else float('nan')
            robuste = '✅ Robuste' if abs(delta_pct) < 10 else ('⚠️ Δ modéré' if abs(delta_pct) < 20 else '❌ Confondant fort')
            rows_comp.append({
                'Variable'       : var,
                'OR sans radon'  : round(or1, 3),
                'p sans radon'   : p1,
                'OR avec radon'  : round(or2, 3),
                'p avec radon'   : p2,
                'Δ OR (%)'       : round(delta_pct, 1),
                'Robustesse'     : robuste,
            })

        # OR du radon lui-même
        if 'radon_score' in tab2.set_index('Variable').index:
            r_row = tab2.set_index('Variable').loc['radon_score']
            print(f"\n  Effet radon : OR = {r_row['OR']:.3f} | p = {r_row['p-value']} {r_row['Sig']}")

        df_comp = pd.DataFrame(rows_comp)
        print(f"\n  ── Comparaison OR polluants (sans vs avec radon) ──")
        try:
            from IPython.display import display; display(df_comp)
        except Exception:
            print(df_comp.to_string(index=False))

        # Visualisation forest plot comparatif
        if len(df_comp) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(df_comp)*0.8 + 1)))
            ax.axvline(x=1, color='black', linestyle='--', lw=1.2)
            for i, row in df_comp.reset_index().iterrows():
                # OR sans radon (gris)
                or_s = tab1.set_index('Variable').loc[row['Variable'], 'OR']
                ic_lo_s = tab1.set_index('Variable').loc[row['Variable'], 'IC 95% inf']
                ic_hi_s = tab1.set_index('Variable').loc[row['Variable'], 'IC 95% sup']
                ax.plot([ic_lo_s, ic_hi_s], [i + 0.15, i + 0.15],
                        color='#AAAAAA', lw=2, label='Sans radon' if i == 0 else '')
                ax.plot(or_s, i + 0.15, 's', color='#AAAAAA', markersize=8)
                # OR avec radon (couleur)
                or_r = tab2.set_index('Variable').loc[row['Variable'], 'OR']
                ic_lo_r = tab2.set_index('Variable').loc[row['Variable'], 'IC 95% inf']
                ic_hi_r = tab2.set_index('Variable').loc[row['Variable'], 'IC 95% sup']
                ax.plot([ic_lo_r, ic_hi_r], [i - 0.15, i - 0.15],
                        color=couleur, lw=2, label='Avec radon' if i == 0 else '')
                ax.plot(or_r, i - 0.15, 'o', color=couleur, markersize=8)
                # Annotation Δ
                delta_str = f"Δ={row['Δ OR (%)']:+.1f}%"
                ax.text(max(ic_hi_s, ic_hi_r) * 1.05, i,
                        delta_str, va='center', fontsize=8,
                        color='green' if abs(row['Δ OR (%)']) < 10 else 'red')

            ax.set_yticks(range(len(df_comp)))
            ax.set_yticklabels(df_comp['Variable'].tolist(), fontsize=9)
            ax.set_xscale('log')
            ax.set_xlabel('Odds Ratio (IC 95%)')
            ax.set_title(f'Sensibilité au radon — {titre}\n□ Sans radon  ○ Avec radon',
                         fontweight='bold')
            handles = [
                mpatches.Patch(color='#AAAAAA', label='Sans radon'),
                mpatches.Patch(color=couleur,   label='Avec radon'),
            ]
            ax.legend(handles=handles, fontsize=9)
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Résumé verbal
        n_robuste = (df_comp['Robustesse'] == '✅ Robuste').sum()
        n_total   = len(df_comp)
        print(f"\n  ── Résumé ──")
        print(f"  {n_robuste}/{n_total} variables d'exposition robustes après ajustement radon (|ΔOR| < 10%)")
        if n_robuste == n_total:
            print("  ✅ L'effet des polluants atmosphériques est ROBUSTE après ajustement pour le radon.")
        elif n_robuste > n_total / 2:
            print("  ⚠️  L'effet des polluants est PARTIELLEMENT robuste — le radon est un confondant modéré.")
        else:
            print("  ❌ L'effet des polluants change fortement avec le radon — confondant fort.")

        resultats[key] = {
            'modele_sans_radon': mod1,
            'modele_avec_radon': mod2,
            'tableau_sans_radon': tab1,
            'tableau_avec_radon': tab2,
            'comparaison': df_comp,
            'auc_sans': auc1,
            'auc_avec': auc2,
        }

    return resultats


def construire_df_final(data, df_clinique, df_air):
    """Fusionne df_air avec df_clinique → df_final avec groupes."""
    df_final = df_clinique.merge(df_air, on=['pseudo_provisoire','date_diagnostic'], how='inner')

    # Colonnes cumul créées selon la fenêtre et les polluants
    cols_cumul = [c for c in df_final.columns if '_cumul_' in c and 'pct' not in c and 'manq' not in c]
    print(f"\n── Colonnes cumul dans df_final : {cols_cumul}")

    # Groupes
    mask_A = pd.Series(False, index=df_final.index)
    for mut in MUTATIONS_NF:
        mask_A |= porte_mutation(df_final, mut)

    df_final['groupe_AB']    = np.where(mask_A,'Groupe A','Groupe B')
    df_final['groupe_CD']    = np.where(df_final['paquet_annee']==0,'Groupe C','Groupe D')
    mask_AC = mask_A | (df_final['paquet_annee']==0)
    df_final['groupe_AC_BD'] = np.where(mask_AC,'Groupe A+C','Groupe B+D')
    df_final['sexe_bin']     = df_final['sexe'].astype(str).str.strip().str.lower().map(
        {'masculin':0,'feminin':1,'féminin':1})

    print(f"\n✅ df_final : {len(df_final)} patients × {len(df_final.columns)} variables")
    for col in ['groupe_AB','groupe_CD','groupe_AC_BD']:
        print(f"  {col}: {dict(df_final[col].value_counts())}")

    cols_spatial = [c for c in df_final.columns if any(k in c for k in ['ICPE','RN_','trafic'])]
    print(f"\nVariables spatiales ({len(cols_spatial)}) : {cols_spatial}")
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def porte_mutation(df, mut):
    col = f'mutation_{mut}'
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().str.lower().isin(
        ['1','true','oui','yes','positive','positif','pos','muté','mute','detected','present'])


def _filtrer(df, cfg):
    d = df.copy()
    if cfg.get('filtre_sexe'):
        d = d[d['sexe'].astype(str).str.strip().str.lower() == cfg['filtre_sexe'].lower()]
    if cfg.get('filtre_fumeur') is True:
        d = d[d['paquet_annee'] > 0]
    elif cfg.get('filtre_fumeur') is False:
        d = d[d['paquet_annee'] == 0]
    if cfg.get('filtre_histologie') and 'histologie_groupe' in d.columns:
        d = d[d['histologie_groupe'].str.lower() == cfg['filtre_histologie'].lower()]
    for col,val in cfg.get('filtre_custom',{}).items():
        if col in d.columns:
            d = d[d[col]==val]
    return d.reset_index(drop=True)


def _vars_modele(cfg, inclure_paquet=True):
    """
    Construit la liste des variables du modèle depuis CFG.
    Filtre automatiquement selon cfg['polluants_retenus'].
    """
    polluants = cfg.get('polluants_retenus', ['PM25','PM10','NO2','O3'])
    seen, vars_mod = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); vars_mod.append(v)

    # Cumul — garder uniquement les polluants retenus
    for v in cfg.get('cumul',[]):
        pol = v.split('_')[0]  # ex: 'PM25' depuis 'PM25_cumul_36m'
        if pol in polluants: add(v)

    # Tendance — garder uniquement les polluants retenus
    for v in cfg.get('tendance',[]):
        pol = v.split('_')[0]
        if pol in polluants: add(v)

    # % du temps — garder uniquement les polluants retenus
    if 'PM25' in polluants:
        for s in cfg.get('pct_pm25',[]): add(f'PM25_pct_sup{s}')
    if 'PM10' in polluants:
        for s in cfg.get('pct_pm10',[]): add(f'PM10_pct_sup{s}')
    if 'NO2' in polluants:
        for s in cfg.get('pct_no2',[]): add(f'NO2_pct_sup{s}')
    if 'O3' in polluants:
        for s in cfg.get('pct_o3',[]): add(f'O3_pct_sup{s}')

    # Variables cliniques et contextuelles (indépendantes du polluant)
    if cfg.get('inclure_age',True): add('age_diagnostic')
    if inclure_paquet and cfg.get('inclure_paquet',True): add('paquet_annee')
    if cfg.get('inclure_sexe',True): add('sexe_bin')
    if cfg.get('inclure_edi',True): add('quintileEDI2021')
    if cfg.get('inclure_trafic',True): add('indice_trafic')
    # Radon — covariable contextuelle (ajustement confondant)
    if cfg.get('inclure_radon', False): add('radon_score')
    # IPG — Indice de Pollution Global (composite z-score PM25+PM10+NO2+O3)
    # S'active dès que inclure_ipg=True, indépendamment de polluants_retenus
    if cfg.get('inclure_ipg', False): add('IPG')
    for v in cfg.get('icpe',[]): add(v)
    return vars_mod


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATION VISUELLE
# ═══════════════════════════════════════════════════════════════════════════════

def explorer_distribution_polluants(df):
    """Boxplots des cumuls par groupe pour les 4 polluants."""
    pols = []
    for p in ['PM25','PM10','NO2','O3']:
        matches = [c for c in df.columns if c.startswith(f'{p}_cumul_') and 'pct' not in c and 'manq' not in c]
        if matches: pols.append((matches[0], f'{p} (μg/m³·j)'))
    groupes = [
        ('groupe_AB','Groupe A','Groupe B','#2E86AB','#A8DADC','A vs B — Mutations NF'),
        ('groupe_CD','Groupe C','Groupe D','#52B788','#E76F51','C vs D — Non-fumeurs vs Fumeurs'),
        ('groupe_AC_BD','Groupe A+C','Groupe B+D','#7B2D8B','#BBBBBB','A+C vs B+D'),
    ]
    for col_g,g1,g2,c1,c2,titre in groupes:
        fig,axes = plt.subplots(1,len(pols),figsize=(5*len(pols),5))
        fig.suptitle(titre,fontsize=13,fontweight='bold')
        if len(pols)==1: axes=[axes]
        df_sub = df[df[col_g].isin([g1,g2])]
        for ax,(col,label) in zip(axes,pols):
            d = df_sub[[col_g,col]].dropna()
            s1,s2 = d.loc[d[col_g]==g1,col],d.loc[d[col_g]==g2,col]
            _,p = mannwhitneyu(s1,s2,alternative='two-sided')
            p_txt = f'p = {"<0.001" if p<0.001 else f"{p:.3f}"}'
            sns.boxplot(data=d,x=col_g,y=col,palette={g1:c1,g2:c2},order=[g1,g2],ax=ax,width=0.5)
            ax.set_title(f'{label}\n{p_txt}',fontsize=10,fontweight='bold',color='red' if p<0.05 else 'gray')
            ax.set_xlabel(''); ax.set_ylabel('μg/m³·j')
        plt.tight_layout(); plt.show()


def explorer_distribution_pct(df):
    """Boxplots des variables % du temps par groupe."""
    cols_pct = [c for c in df.columns if '_pct_sup' in c]
    if not cols_pct:
        print("⚠️ Aucune variable % trouvée."); return
    groupes = [
        ('groupe_AB','Groupe A','Groupe B',{'Groupe A':'#2E86AB','Groupe B':'#A8DADC'},'A vs B'),
        ('groupe_CD','Groupe C','Groupe D',{'Groupe C':'#52B788','Groupe D':'#E76F51'},'C vs D'),
        ('groupe_AC_BD','Groupe A+C','Groupe B+D',{'Groupe A+C':'#7B2D8B','Groupe B+D':'#BBBBBB'},'A+C vs B+D'),
    ]
    for col_g,g1,g2,palette,titre in groupes:
        ncols = 4
        nrows = max(1,int(np.ceil(len(cols_pct)/ncols)))
        fig,axes = plt.subplots(nrows,ncols,figsize=(20,nrows*4))
        axes_flat = np.array(axes).flatten()
        fig.suptitle(f'Distribution % du temps — {titre}',fontsize=13,fontweight='bold')
        df_sub = df[df[col_g].isin([g1,g2])]
        for ax,col in zip(axes_flat,cols_pct):
            d = df_sub[[col_g,col]].dropna()
            if len(d)<5: ax.set_visible(False); continue
            _,p = mannwhitneyu(d.loc[d[col_g]==g1,col],d.loc[d[col_g]==g2,col],alternative='two-sided')
            p_txt = '<0.001' if p<0.001 else f'{p:.3f}'
            sns.boxplot(data=d,x=col_g,y=col,palette=palette,order=[g1,g2],ax=ax,width=0.5)
            ax.set_title(f'{col}\np = {p_txt}',fontsize=9,fontweight='bold',color='red' if p<0.05 else 'gray')
            ax.set_xlabel(''); ax.set_ylabel('% du temps')
        for ax in axes_flat[len(cols_pct):]: ax.set_visible(False)
        plt.tight_layout(); plt.show()


def explorer_tendances(df):
    """Barplots de la distribution des tendances MM365+MK par polluant."""
    COULEURS_T = {'montée':'#E76F51','stable':'#A8DADC','descente':'#2E86AB'}
    ORDRE = ['montée','stable','descente']
    pols_dispo = [p for p in POLLUANTS_DEFAUT if f'{p}_mm365_mk_tendance' in df.columns]

    # Figure 1 — cohorte entière
    fig,axes = plt.subplots(1,len(pols_dispo),figsize=(5*len(pols_dispo),5))
    fig.suptitle('Distribution des tendances MM365 + Mann-Kendall (Cohorte entière)',fontsize=13,fontweight='bold')
    if len(pols_dispo)==1: axes=[axes]
    for ax,pol in zip(axes,pols_dispo):
        col = f'{pol}_mm365_mk_tendance'
        total = df[col].notna().sum()
        counts = df[col].value_counts().reindex(ORDRE,fill_value=0)
        bars = ax.bar(ORDRE,counts.values,color=[COULEURS_T[t] for t in ORDRE],
                      edgecolor='white',linewidth=1.5,width=0.6)
        for bar,n in zip(bars,counts.values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+total*0.01,
                    f'{n}\n({n/total*100:.1f}%)',ha='center',va='bottom',fontsize=10,fontweight='bold')
        ax.set_title(pol,fontsize=13,fontweight='bold')
        ax.set_ylabel('Nombre de patients')
        ax.set_ylim(0,counts.max()*1.25)
        ax.set_xticklabels(['↑ Montée','→ Stable','↓ Descente'],fontsize=11)
        ax.grid(True,axis='y',alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.show()

    # Figure 2 — par groupe
    groupes_vis = [
        ('groupe_AB','Groupe A','Groupe B','A vs B'),
        ('groupe_CD','Groupe C','Groupe D','C vs D'),
        ('groupe_AC_BD','Groupe A+C','Groupe B+D','A+C vs B+D'),
    ]
    for pol in pols_dispo:
        col = f'{pol}_mm365_mk_tendance'
        fig,axes = plt.subplots(1,3,figsize=(18,5))
        fig.suptitle(f'Tendance {pol} par groupe',fontsize=13,fontweight='bold')
        for ax,(col_g,g1,g2,titre) in zip(axes,groupes_vis):
            df_sub = df[df[col_g].isin([g1,g2])]
            tab = (df_sub.groupby([col_g,col]).size().unstack(fill_value=0).reindex(columns=ORDRE,fill_value=0))
            groupes = tab.index.tolist()
            x,larg = np.arange(len(groupes)),0.22
            for j,tendance in enumerate(ORDRE):
                vals = tab[tendance].values if tendance in tab.columns else [0]*len(groupes)
                totaux = tab.sum(axis=1).values
                pcts = [v/t*100 if t>0 else 0 for v,t in zip(vals,totaux)]
                offsets = x+(j-1.5+0.5)*larg
                bars = ax.bar(offsets,vals,width=larg,color=COULEURS_T[tendance],
                              edgecolor='white',linewidth=1,label=tendance)
                for bar,v,p in zip(bars,vals,pcts):
                    if v>0:
                        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,
                                f'{v}\n({p:.0f}%)',ha='center',va='bottom',fontsize=8)
            ax.set_title(titre,fontsize=11,fontweight='bold')
            ax.set_ylabel('Nombre de patients')
            ax.set_xticks(x); ax.set_xticklabels(groupes,fontsize=10)
            ax.set_ylim(0,tab.values.max()*1.35)
            ax.legend(title='Tendance',fontsize=9)
            ax.grid(True,axis='y',alpha=0.3); ax.spines[['top','right']].set_visible(False)
            try:
                from scipy.stats import chi2_contingency
                _,p,_,_ = chi2_contingency(tab.values)
                p_txt = '<0.001' if p<0.001 else f'{p:.3f}'
                ax.set_xlabel(f'Chi² p = {p_txt}',fontsize=10,color='red' if p<0.05 else 'gray',fontweight='bold')
            except Exception: pass
        plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SEUILS CRITIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def chercher_seuils_critiques(df, cfg):
    """Cherche les seuils critiques d'exposition pour les 3 groupes."""
    # Filtrer les seuils selon polluants_retenus
    all_seuils = cfg.get('seuils_pct', {'PM25':[5,10,15,25,35],'PM10':[35,45,50,60,80,90],'O3':[100,120,180]})
    polluants  = cfg.get('polluants_retenus', list(all_seuils.keys()))
    seuils_test = {pol: seuils for pol,seuils in all_seuils.items() if pol in polluants}
    print(f"Polluants testés : {list(seuils_test.keys())}")
    GROUPES = [
        ('groupe_AB','Groupe A','Groupe B','A vs B — Mutations NF'),
        ('groupe_CD','Groupe C','Groupe D','C vs D — Non-fumeurs vs Fumeurs'),
        ('groupe_AC_BD','Groupe A+C','Groupe B+D','A+C vs B+D'),
    ]
    resultats = []
    print("="*65+"\nRECHERCHE DU SEUIL CRITIQUE\n"+"="*65)
    for col_g,g1,g2,titre in GROUPES:
        print(f"\n── {titre} ──")
        df_sub = df[df[col_g].isin([g1,g2])].copy()
        df_sub['outcome'] = (df_sub[col_g]==g1).astype(int)
        for pol,seuils in seuils_test.items():
            for seuil in seuils:
                col = f'{pol}_pct_sup{seuil}'
                if col not in df_sub.columns: continue
                df_m = df_sub[['outcome',col]].dropna()
                s1,s2 = df_m.loc[df_m['outcome']==1,col].values,df_m.loc[df_m['outcome']==0,col].values
                if len(s1)<5 or len(s2)<5: continue
                _,p = mannwhitneyu(s1,s2,alternative='two-sided')
                auc = roc_auc_score(df_m['outcome'],df_m[col])
                direction = '↑ g1 plus exposé' if auc>0.5 else '↓ g1 moins exposé'
                resultats.append({'col_g':col_g,'Groupe':titre,'Polluant':pol,'Seuil':seuil,
                    'Variable':col,'Médiane g1':round(float(np.median(s1)),1),
                    'Médiane g2':round(float(np.median(s2)),1),'p-value':p,
                    'p_fmt':'<0.001' if p<0.001 else f'{p:.3f}','AUC':round(auc,3),
                    'Direction':direction,'Sig':'✅' if p<0.05 else '—'})
                print(f"  {col:<25} | p={('<0.001' if p<0.001 else f'{p:.3f}'):<6} | AUC={auc:.3f} | {direction} {'✅' if p<0.05 else '—'}")
    df_res = pd.DataFrame(resultats)

    # Visualisation
    for pol in seuils_test:
        pol_label = {'PM25':'PM2.5','PM10':'PM10','O3':'O3'}.get(pol,pol)
        pol_color = {'PM25':'#2E86AB','PM10':'#E76F51','O3':'#52B788'}.get(pol,'#888888')
        fig,axes = plt.subplots(2,3,figsize=(20,10))
        fig.suptitle(f'Pouvoir discriminant — {pol_label}',fontsize=13,fontweight='bold')
        for col_idx,(col_g,g1,g2,titre) in enumerate(GROUPES):
            df_g = df_res[(df_res['col_g']==col_g)&(df_res['Polluant']==pol)].sort_values('Seuil')
            ax_p,ax_auc = axes[0,col_idx],axes[1,col_idx]
            if len(df_g)==0: ax_p.set_visible(False);ax_auc.set_visible(False);continue
            log_p = -np.log10(df_g['p-value'].clip(lower=1e-10))
            bars = ax_p.bar([f'>{s}' for s in df_g['Seuil']],log_p,
                           color=[pol_color if p<0.05 else '#DDDDDD' for p in df_g['p-value']],
                           alpha=0.85,edgecolor='white',linewidth=1.5)
            for bar,row in zip(bars,df_g.itertuples()):
                if row._8<0.05:
                    ax_p.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,'★',ha='center',fontsize=14,color=pol_color)
            ax_p.axhline(y=-np.log10(0.05),color='red',linestyle='--',lw=1.5,alpha=0.7)
            ax_p.set_title(f'{titre}\n-log10(p)',fontsize=10,fontweight='bold')
            ax_p.set_xlabel(f'Seuil {pol_label}'); ax_p.set_ylabel('-log10(p)'); ax_p.grid(True,axis='y',alpha=0.3)
            aucs_d = np.abs(df_g['AUC'].values-0.5)
            bars2 = ax_auc.bar([f'>{s}' for s in df_g['Seuil']],aucs_d,
                               color=[pol_color if p<0.05 else '#DDDDDD' for p in df_g['p-value']],
                               alpha=0.85,edgecolor='white',linewidth=1.5)
            for bar,row in zip(bars2,df_g.itertuples()):
                ax_auc.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                            '↑' if row.AUC>0.5 else '↓',ha='center',fontsize=12,
                            color=pol_color if row._8<0.05 else '#999999')
            ax_auc.set_title(f'{titre}\n|AUC-0.5|',fontsize=10,fontweight='bold')
            ax_auc.set_xlabel(f'Seuil {pol_label}'); ax_auc.set_ylabel('|AUC-0.5|'); ax_auc.grid(True,axis='y',alpha=0.3)
        plt.tight_layout(); plt.show()

    # ── Correction Benjamini-Hochberg (FDR) ──────────────────────────────────
    from statsmodels.stats.multitest import multipletests
    if len(df_res) > 0:
        _, pvals_corr, _, _ = multipletests(df_res['p-value'].values, method='fdr_bh')
        df_res['p_adj_BH'] = pvals_corr
        df_res['p_adj_fmt'] = ['<0.001' if p<0.001 else f'{p:.3f}' for p in pvals_corr]
        df_res['Sig_BH']   = ['✅' if p<0.05 else '—' for p in pvals_corr]
        sig_raw = (df_res['Sig']=='✅').sum()
        sig_bh  = (df_res['Sig_BH']=='✅').sum()
        print(f"\n{'='*65}\nCORRECTION BENJAMINI-HOCHBERG\n{'='*65}")
        print(f"  Significatifs avant BH : {sig_raw} / {len(df_res)}")
        print(f"  Significatifs après BH : {sig_bh} / {len(df_res)}")
        if sig_bh < sig_raw:
            print(f"  ⚠️  {sig_raw-sig_bh} seuil(s) perdent leur significativité")

    # ── Seuils retenus après BH ───────────────────────────────────────────────
    seuils_retenus = {pol:[] for pol in seuils_test}
    for pol in seuils_test:
        df_pol = df_res[df_res['Polluant']==pol]
        col_sig = 'Sig_BH' if 'Sig_BH' in df_res.columns else 'Sig'
        df_sig = df_pol[df_pol[col_sig]=='✅']

        # ── RÈGLE : 1 seul seuil par polluant = le plus significatif ──────────
        if len(df_sig) > 0:
            # Garder uniquement le seuil avec p-value minimale
            best_sig = df_sig.loc[df_sig['p-value'].idxmin()]
            seuils_sig = [int(best_sig['Seuil'])]
            if len(df_sig) > 1:
                exclus = [int(s) for s in df_sig['Seuil'].tolist() if int(s) != seuils_sig[0]]
                print(f"  ℹ️  {pol} : {len(df_sig)} seuils sig. → retenu >{seuils_sig[0]} "
                      f"(p={best_sig['p-value']:.3f}) | exclus : {exclus}")
        else:
            best = df_pol.loc[df_pol['p-value'].idxmin()] if len(df_pol)>0 else None
            if best is not None:
                seuils_sig=[int(best['Seuil'])]
                print(f"  ⚠️  {pol} : fallback meilleur brut (>{int(best['Seuil'])})")
            else:
                seuils_sig=[]
        seuils_retenus[pol] = seuils_sig
    print("\n── Seuils retenus après BH ──")
    for pol,seuils in seuils_retenus.items():
        print(f"  {pol:<8} : {seuils if seuils else '— aucun'}")
    return df_res, seuils_retenus


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CORRÉLATION & FAMD
# ═══════════════════════════════════════════════════════════════════════════════


def _firth_regression(X, y, max_iter=100, tol=1e-6):
    """Régression logistique de Firth. Corrige biais EPV faible."""
    from scipy import stats
    X=np.array(X,dtype=float); y=np.array(y,dtype=float)
    n,p=X.shape; beta=np.zeros(p)
    for _ in range(max_iter):
        pi=np.clip(1/(1+np.exp(-X@beta)),1e-10,1-1e-10)
        W=np.diag(pi*(1-pi)); XtWX=X.T@W@X
        try: H_inv=np.linalg.inv(XtWX)
        except: H_inv=np.linalg.pinv(XtWX)
        try:
            Wh=np.diag(np.sqrt(pi*(1-pi)))
            h=np.diag(Wh@X@H_inv@X.T@Wh)
        except: h=np.zeros(n)
        U_star=X.T@(y-pi+h*(0.5-pi))
        delta=H_inv@U_star; beta_new=beta+delta
        if np.max(np.abs(delta))<tol: beta=beta_new; break
        beta=beta_new
    pi=np.clip(1/(1+np.exp(-X@beta)),1e-10,1-1e-10)
    XtWX=X.T@np.diag(pi*(1-pi))@X
    try: var_cov=np.linalg.inv(XtWX)
    except: var_cov=np.linalg.pinv(XtWX)
    return beta, var_cov

def _run_regression_firth(d, vars_ok, titre, couleur):
    """Lance Firth sur DataFrame préparé."""
    from scipy import stats
    scaler=StandardScaler(); d_std=d.copy()
    d_std[vars_ok]=scaler.fit_transform(d[vars_ok])
    vars_ok=[v for v in vars_ok if d_std[v].std()>1e-8]
    X_arr=np.column_stack([np.ones(len(d_std))]+[d_std[v].values for v in vars_ok])
    y_arr=d_std['outcome'].values
    try: beta,var_cov=_firth_regression(X_arr,y_arr)
    except Exception as e: print(f"⚠️  Firth échoué : {e}"); return None,None,None
    se=np.sqrt(np.diag(var_cov)); z=beta/(se+1e-10); pval=2*(1-stats.norm.cdf(np.abs(z)))
    rows=[{'Variable':nom,'OR':round(float(np.exp(b)),3),
           'IC 95% inf':round(float(np.exp(b-1.96*s)),3),'IC 95% sup':round(float(np.exp(b+1.96*s)),3),
           'p-value':'<0.001' if p<0.001 else f'{p:.3f}','p_num':float(p),
           'Sig':'✅' if p<0.05 else '—','Méthode':'Firth'}
          for nom,b,s,p in zip(vars_ok,beta[1:],se[1:],pval[1:])]
    tab=pd.DataFrame(rows)
    try:
        from IPython.display import display; display(tab.drop(columns=['p_num','Méthode']))
    except: print(tab.drop(columns=['p_num','Méthode']).to_string(index=False))
    pi_hat=1/(1+np.exp(-X_arr@beta)); auc=roc_auc_score(y_arr,pi_hat)
    fig,ax=plt.subplots(figsize=(8,max(4,len(vars_ok)*0.4)))
    ax.axvline(x=1,color='black',linestyle='--',lw=1.2)
    for i,row in tab.iterrows():
        c=couleur if row['Sig']=='✅' else '#AAAAAA'
        ax.plot([row['IC 95% inf'],row['IC 95% sup']],[i,i],color=c,lw=2)
        ax.plot(row['OR'],i,'o',color=c,markersize=8)
        if row['Sig']=='✅':
            ax.text(row['IC 95% sup']*1.05,i,f"p={row['p-value']}",va='center',fontsize=8,color=couleur,fontweight='bold')
    ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab['Variable'].tolist(),fontsize=9)
    ax.set_xscale('log'); ax.set_xlabel('OR — Firth (IC 95%)')
    ax.set_title(f'Forest plot Firth — {titre}\nAUC={auc:.3f}',fontweight='bold')
    ax.grid(True,axis='x',alpha=0.3); plt.tight_layout(); plt.show()
    print(f"AUC={auc:.3f} [Firth]")
    return None, tab, auc


def heatmap_correlation(df, vars_grouped):
    """vars_grouped : dict {label: [colonnes]}"""
    all_vars = [v for vlist in vars_grouped.values() for v in vlist if v in df.columns]
    if not all_vars: print("⚠️ Aucune variable."); return
    df_corr = df[all_vars].apply(pd.to_numeric,errors='coerce').corr(method='spearman')
    mask = np.triu(np.ones_like(df_corr,dtype=bool),k=1)
    fig,ax = plt.subplots(figsize=(max(10,len(all_vars)*0.8),max(8,len(all_vars)*0.8)))
    sns.heatmap(df_corr,mask=mask,annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-1,vmax=1,
                ax=ax,square=True,cbar_kws={'label':'r de Spearman'},linewidths=0.3,annot_kws={"size":8})
    ax.set_title(f'Corrélation de Spearman (n={df[all_vars].dropna().shape[0]} patients)',
                 fontsize=12,fontweight='bold')
    couleurs_sec = ['#2E86AB','#E76F51','#52B788','#7B2D8B','#888888','#F59E0B']
    idx = 0
    sections = []
    for i,(label,vlist) in enumerate(vars_grouped.items()):
        n = len([v for v in vlist if v in df.columns])
        if n == 0: continue
        c = couleurs_sec[i % len(couleurs_sec)]
        ax.axhline(y=idx, color=c, lw=2)
        ax.axvline(x=idx, color=c, lw=2)
        sections.append((idx, n, label, c))
        idx += n

    # Labels de section placés dans la marge gauche, hors de la heatmap
    n_vars = len(all_vars)
    ax_pos = ax.get_position()
    for idx_s, n, label, c in sections:
        # Position verticale centrée sur la section
        y_frac = ax_pos.y0 + ax_pos.height * (1 - (idx_s + n / 2) / n_vars)
        fig.text(
            ax_pos.x0 - 0.02,   # juste à gauche de l'axe
            y_frac,
            label,
            ha='right', va='center',
            fontsize=9, color=c, fontweight='bold',
            transform=fig.transFigure
        )

    plt.tight_layout()
    # Agrandir la marge gauche pour les labels de section
    fig.subplots_adjust(left=max(0.25, ax_pos.x0))
    plt.show()


def run_famd(df, vars_cont, vars_cat, n_components=5):
    """Lance la FAMD et retourne (famd, coords, var_exp, df_famd)."""
    try: import prince
    except ImportError:
        import subprocess; subprocess.run(['pip','install','prince','--quiet']); import prince
    from matplotlib.patches import Ellipse

    vars_cont_ok = [v for v in vars_cont if v in df.columns]
    vars_cat_ok  = [v for v in vars_cat  if v in df.columns]
    df_famd = df[vars_cont_ok+vars_cat_ok].copy()
    for v in vars_cont_ok: df_famd[v] = pd.to_numeric(df_famd[v],errors='coerce')
    for v in vars_cat_ok:  df_famd[v] = df_famd[v].astype(str)
    df_famd = df_famd.dropna()
    print(f"FAMD : {len(df_famd)} patients × {len(df_famd.columns)} variables")

    famd = prince.FAMD(n_components=n_components,n_iter=10,random_state=42).fit(df_famd)
    coords = famd.row_coordinates(df_famd)
    coords.columns = [f'Dim{i+1}' for i in range(coords.shape[1])]
    var_exp = famd.eigenvalues_summary
    print("\n── Variance expliquée ──"); print(var_exp.head())

    def pct_var(i):
        val = var_exp.iloc[i]["% of variance"]
        return float(str(val).replace('%','').strip()) if isinstance(val,str) else float(val)*(100 if float(val)<=1 else 1)

    # Visualisation
    groupes_vis = [
        ('groupe_AB',{'Groupe A':'#2E86AB','Groupe B':'#A8DADC'},'A vs B'),
        ('groupe_CD',{'Groupe C':'#52B788','Groupe D':'#E76F51'},'C vs D'),
        ('groupe_AC_BD',{'Groupe A+C':'#7B2D8B','Groupe B+D':'#BBBBBB'},'A+C vs B+D'),
    ]
    fig,axes = plt.subplots(1,3,figsize=(22,6))
    fig.suptitle('FAMD — Plan factoriel Dim1 × Dim2',fontsize=13,fontweight='bold')
    for ax,(col_g,palette,titre) in zip(axes,groupes_vis):
        if col_g not in df_famd.columns: ax.set_visible(False); continue
        vals = df_famd[col_g].values
        for g,c in palette.items():
            m = vals==g
            ax.scatter(coords.loc[m,'Dim1'],coords.loc[m,'Dim2'],c=c,label=g,alpha=0.5,s=20)
            x,y = coords.loc[m,'Dim1'].values,coords.loc[m,'Dim2'].values
            if len(x)<5: continue
            cov = np.cov(x,y); vals_e,vecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*vecs[:,-1][::-1]))
            w,h = 2*2.0*np.sqrt(vals_e)
            ax.add_patch(Ellipse(xy=(x.mean(),y.mean()),width=w,height=h,angle=angle,
                                  edgecolor=c,fc='None',lw=2,linestyle='--'))
        ax.set_xlabel(f'Dim1 ({pct_var(0):.1f}%)'); ax.set_ylabel(f'Dim2 ({pct_var(1):.1f}%)')
        ax.set_title(titre,fontsize=11,fontweight='bold'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()

    print("\n── Top 10 variables — Dim1 ──"); contrib = famd.column_contributions_
    print(contrib.sort_values(0,ascending=False).head(10))
    print("\n── Top 10 variables — Dim2 ──")
    print(contrib.sort_values(1,ascending=False).head(10))
    return famd, coords, var_exp, df_famd


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TABLEAUX DE CARACTÉRISTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _n_pct(series, valeur=None):
    s = pd.to_numeric(series,errors='coerce') if valeur is None else series
    if valeur is not None:
        n = (s==valeur).sum(); tot = len(s)
        return f'{n} ({n/tot*100:.1f}%)' if tot>0 else 'NA'
    s = s.dropna()
    return f'{s.median():.0f} [{s.min():.0f}–{s.max():.0f}]' if len(s)>0 else 'NA'

def _p_chi2(df,col,col_g,g1,g2):
    sub = df[df[col_g].isin([g1,g2])][[col,col_g]].dropna()
    if len(sub)==0: return 'NA'
    table = pd.crosstab(sub[col],sub[col_g])
    if table.shape[0]<2 or table.shape[1]<2: return 'NA'
    try:
        _,p,_,exp = chi2_contingency(table)
        if (exp<5).any() and table.shape==(2,2): _,p = fisher_exact(table)
        return '<0.001' if p<0.001 else f'{p:.3f}'
    except: return 'NA'

def _p_mwu(df,col,col_g,g1,g2):
    s1=pd.to_numeric(df.loc[df[col_g]==g1,col],errors='coerce').dropna()
    s2=pd.to_numeric(df.loc[df[col_g]==g2,col],errors='coerce').dropna()
    if len(s1)<2 or len(s2)<2: return 'NA'
    try: _,p=mannwhitneyu(s1,s2,alternative='two-sided'); return '<0.001' if p<0.001 else f'{p:.3f}'
    except: return 'NA'

def tableaux_caracteristiques(df):
    """Génère les 3 tableaux de caractéristiques."""
    def tableau(df_f,col_g,g1,g2,lbl_g1,lbl_g2,titre,legende=''):
        df_sub = df_f[df_f[col_g].isin([g1,g2])].copy()
        n_tot,n_g1,n_g2 = len(df_sub),(df_sub[col_g]==g1).sum(),(df_sub[col_g]==g2).sum()
        lbl_tot = f'Total (n={n_tot})'
        rows=[]
        def sep(l): rows.append({'Caractéristique':l,lbl_tot:'',lbl_g1:'',lbl_g2:'','p-value':''})
        def cat_h(col): rows.append({'Caractéristique':col,lbl_tot:'',lbl_g1:'',lbl_g2:'','p-value':_p_chi2(df_sub,col,col_g,g1,g2)})
        def cat_r(col,val,lbl): rows.append({'Caractéristique':f'  {lbl}',lbl_tot:_n_pct(df_sub[col],val),lbl_g1:_n_pct(df_sub.loc[df_sub[col_g]==g1,col],val),lbl_g2:_n_pct(df_sub.loc[df_sub[col_g]==g2,col],val),'p-value':''})
        def cont_r(col,lbl): rows.append({'Caractéristique':lbl,lbl_tot:_n_pct(df_sub[col]),lbl_g1:_n_pct(df_sub.loc[df_sub[col_g]==g1,col]),lbl_g2:_n_pct(df_sub.loc[df_sub[col_g]==g2,col]),'p-value':_p_mwu(df_sub,col,col_g,g1,g2)})
        def mut_r(mut,lbl):
            col=f'mutation_{mut}'
            if col not in df_sub.columns: return
            tot=porte_mutation(df_sub,mut); g1s=porte_mutation(df_sub.loc[df_sub[col_g]==g1],mut); g2s=porte_mutation(df_sub.loc[df_sub[col_g]==g2],mut)
            nt,n1,n2=tot.sum(),g1s.sum(),g2s.sum(); d1,d2=(df_sub[col_g]==g1).sum(),(df_sub[col_g]==g2).sum()
            rows.append({'Caractéristique':f'  {lbl}',lbl_tot:f'{nt} ({nt/n_tot*100:.1f}%)',lbl_g1:f'{n1} ({n1/d1*100:.1f}%)' if d1>0 else 'NA',lbl_g2:f'{n2} ({n2/d2*100:.1f}%)' if d2>0 else 'NA','p-value':''})

        rows.append({'Caractéristique':'N',lbl_tot:str(n_tot),lbl_g1:str(n_g1),lbl_g2:str(n_g2),'p-value':''})
        df_sub['sexe']=df_sub['sexe'].astype(str).str.strip().str.lower()
        cat_h('sexe'); cat_r('sexe','feminin','Femme'); cat_r('sexe','masculin','Homme')
        cont_r('age_diagnostic','Âge (médiane [range])'); cont_r('paquet_annee','Paquets-années (médiane [range])')
        cat_h('histologie_groupe'); [cat_r('histologie_groupe',v,v) for v in ['Adénocarcinome','Carcinome épidermoïde','Autre']]
        sep('Mutations NF'); [mut_r(m,m) for m in ['EGFR','ALK','ROS1','RET','NTRK','ERBB2','MET']]
        sep('Mutations fumeurs'); [mut_r(m,m) for m in ['KRAS','BRAF']]

        df_t = pd.DataFrame(rows)[['Caractéristique',lbl_tot,lbl_g1,lbl_g2,'p-value']]
        print(f"\n{'='*70}\n{titre}\n{'='*70}")
        try:
            from IPython.display import display; display(df_t)
        except: print(df_t.to_string(index=False))
        if legende: print(f"\nLégende : {legende}")
        return df_t

    t1 = tableau(df,'groupe_AB','Groupe A','Groupe B',
        f'Groupe A (n={(df["groupe_AB"]=="Groupe A").sum()})',
        f'Groupe B (n={(df["groupe_AB"]=="Groupe B").sum()})',
        'Tableau 1 — A vs B (Mutations NF vs Autres)')
    t2 = tableau(df[df['groupe_CD'].isin(['Groupe C','Groupe D'])],'groupe_CD','Groupe C','Groupe D',
        f'Groupe C (n={(df["groupe_CD"]=="Groupe C").sum()})',
        f'Groupe D (n={(df["groupe_CD"]=="Groupe D").sum()})',
        'Tableau 2 — C vs D (Non-fumeurs vs Fumeurs)')
    t3 = tableau(df,'groupe_AC_BD','Groupe A+C','Groupe B+D',
        f'Groupe A+C (n={(df["groupe_AC_BD"]=="Groupe A+C").sum()})',
        f'Groupe B+D (n={(df["groupe_AC_BD"]=="Groupe B+D").sum()})',
        'Tableau 3 — A+C vs B+D')
    return t1,t2,t3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RÉGRESSIONS LOGISTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _tableau_OR(modele, vars_modele):
    rows=[]
    for var in vars_modele:
        if var not in modele.params.index: continue
        p = modele.pvalues[var]
        p = float(p.iloc[0]) if isinstance(p,pd.Series) else float(p)
        rows.append({'Variable':var,'OR':round(float(np.exp(modele.params[var])),3),
            'IC 95% inf':round(float(np.exp(modele.conf_int().loc[var,0])),3),
            'IC 95% sup':round(float(np.exp(modele.conf_int().loc[var,1])),3),
            'p-value':'<0.001' if p<0.001 else f'{p:.3f}','p_num':p,'Sig':'✅' if p<0.05 else '—'})
    return pd.DataFrame(rows)

def _plot_roc_forest(modele, X, y, vars_modele, titre, couleur):
    y_pred=modele.predict(X); auc=roc_auc_score(y,y_pred); fpr,tpr,_=roc_curve(y,y_pred)
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,len(vars_modele)*0.38+1)))
    axes[0].plot(fpr,tpr,color=couleur,lw=2.5,label=f'AUC = {auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    axes[0].set_xlabel('1 - Spécificité'); axes[0].set_ylabel('Sensibilité')
    axes[0].set_title(f'Courbe ROC — {titre}',fontweight='bold'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    vars_ok=[v for v in vars_modele if v in modele.params.index]
    ORs=[float(np.exp(modele.params[v])) for v in vars_ok]
    IC_inf=[float(np.exp(modele.conf_int().loc[v,0])) for v in vars_ok]
    IC_sup=[float(np.exp(modele.conf_int().loc[v,1])) for v in vars_ok]
    pvals=[float(modele.pvalues[v]) if not isinstance(modele.pvalues[v],pd.Series) else float(modele.pvalues[v].iloc[0]) for v in vars_ok]
    axes[1].axvline(x=1,color='black',linestyle='--',lw=1.2)
    for i,(OR,lo,hi,c,p) in enumerate(zip(ORs,IC_inf,IC_sup,[couleur if p<0.05 else '#AAAAAA' for p in pvals],pvals)):
        axes[1].plot([lo,hi],[i,i],color=c,lw=2.2); axes[1].plot(OR,i,'o',color=c,markersize=8)
        if p<0.05:
            p_txt='<0.001' if p<0.001 else f'{p:.3f}'
            axes[1].text(max(IC_sup)*1.05,i,f'p={p_txt}',va='center',fontsize=8,color=couleur,fontweight='bold')
    axes[1].set_yticks(range(len(vars_ok))); axes[1].set_yticklabels(vars_ok,fontsize=9)
    axes[1].set_xscale('log'); axes[1].set_xlabel('Odds Ratio (IC 95%)')
    axes[1].set_title(f'Forest plot — {titre}',fontweight='bold'); axes[1].grid(True,axis='x',alpha=0.3)
    plt.tight_layout(); plt.show()
    return auc

def _run_regression(df, vars_modele, col_g, g1, g2, titre, couleur, inclure_paquet=True, cfg=None):
    if cfg:
        df = _filtrer(df, cfg)
    d = df[df[col_g].isin([g1,g2])].copy() if col_g else df.copy()
    if col_g: d['outcome']=(d[col_g]==g1).astype(int)
    vars_ok=[v for v in vars_modele if v in d.columns]
    d = d[vars_ok+['outcome']].dropna()
    if len(d)<30 or d['outcome'].sum()<5:
        print(f"⚠️  {titre} : effectif insuffisant (N={len(d)}, n+={d['outcome'].sum()})"); return None,None,None
    epv=d['outcome'].sum()/len(vars_ok)
    print(f"\n{'='*60}\n{titre}\n{'='*60}")
    print(f"N={len(d)} | n+={d['outcome'].sum()} ({d['outcome'].mean()*100:.1f}%) | EPV={epv:.1f} {'✅' if epv>=10 else '⚠️'}")
    scaler=StandardScaler(); d_std=d.copy(); d_std[vars_ok]=scaler.fit_transform(d[vars_ok])

    # ── Retirer les variables constantes (variance nulle après standardisation) ──
    vars_ok = [v for v in vars_ok if d_std[v].std() > 1e-8]
    if not vars_ok:
        print(f"⚠️  {titre} : toutes les variables sont constantes après filtrage."); return None,None,None

    # ── Retirer variables constantes ────────────────────────────────────────
    vars_ok=[v for v in vars_ok if d_std[v].std()>1e-8]
    if not vars_ok: print(f"⚠️  {titre} : toutes variables constantes"); return None,None,None

    # ── Déséquilibre des classes ──────────────────────────────────────────
    n_pos=int(d['outcome'].sum()); n_neg=len(d)-n_pos
    if n_neg/max(1,n_pos)>3:
        print(f"  ⚠️  Déséquilibre détecté (ratio={n_neg/n_pos:.1f}) — logistique standard")

    X=sm.add_constant(d_std[vars_ok]); y=d_std['outcome']
    try: modele=sm.Logit(y,X).fit(disp=False)
    except:
        try: modele=sm.Logit(y,X).fit(disp=False,method='bfgs',maxiter=200)
        except Exception as e2:
            print(f"⚠️  {titre} : échec ({e2})"); return None,None,None
    tab=_tableau_OR(modele,vars_ok)
    try:
        from IPython.display import display; display(tab.drop(columns=['p_num']))
    except: print(tab.drop(columns=['p_num']).to_string(index=False))
    auc=_plot_roc_forest(modele,X,y,vars_ok,titre,couleur)
    print(f"AUC={auc:.3f} | R²McFadden={1-(modele.llf/modele.llnull):.3f} | AIC={modele.aic:.1f}")
    sig=tab[tab['Sig']=='✅']
    if len(sig)>0:
        print(f"\n  Variables significatives :")
        [print(f"    ✅ {r['Variable']:<35} OR={r['OR']:.3f} | p={r['p-value']}") for _,r in sig.iterrows()]
    else: print("\n  — Aucune variable significative.")
    return modele,tab,auc


def analyses_principales(df, cfg):
    """Lance les 3 régressions principales + synthèse."""
    print("\n"+"█"*65+"\nANALYSES PRINCIPALES\n"+"█"*65)
    res={}
    for key,col_g,g1,g2,titre,paquet in [
        ('AB','groupe_AB','Groupe A','Groupe B','A vs B — Mutations NF vs Autres',True),
        ('CD','groupe_CD','Groupe C','Groupe D','C vs D — Non-fumeurs vs Fumeurs',False),
        ('ACBD','groupe_AC_BD','Groupe A+C','Groupe B+D','A+C vs B+D',False),
    ]:
        vars_mod = _vars_modele(cfg, inclure_paquet=paquet)
        modele,tab,auc = _run_regression(df,vars_mod,col_g,g1,g2,titre,COULEURS[key],cfg=cfg)
        res[key]={'modele':modele,'tableau':tab,'auc':auc,'groupe':key}

    # Forest synthèse
    configs_ok = [(k,COULEURS[k]) for k in ['AB','CD','ACBD'] if res.get(k,{}).get('modele') is not None]
    if configs_ok:
        fig,axes=plt.subplots(1,len(configs_ok),figsize=(8*len(configs_ok),10))
        fig.suptitle('Forest plots — OR ajustés (IC 95%)',fontsize=13,fontweight='bold')
        if len(configs_ok)==1: axes=[axes]
        titres={'AB':'A vs B — Mutations NF','CD':'C vs D — Non-fumeurs vs Fumeurs','ACBD':'A+C vs B+D'}
        for ax,(key,couleur) in zip(axes,configs_ok):
            r=res[key]; tab=r['tableau']; ORs=tab['OR'].values; IC_inf=tab['IC 95% inf'].values
            IC_sup=tab['IC 95% sup'].values; pvals=tab['p_num'].values; vars_ok=tab['Variable'].tolist()
            ax.axvline(x=1,color='black',linestyle='--',lw=1.2)
            for i,(OR,lo,hi,p) in enumerate(zip(ORs,IC_inf,IC_sup,pvals)):
                c=couleur if p<0.05 else '#CCCCCC'
                ax.plot([lo,hi],[i,i],color=c,lw=2.5); ax.plot(OR,i,'o',color=c,markersize=9)
                if p<0.05:
                    p_txt='<0.001' if p<0.001 else f'{p:.3f}'
                    ax.text(max(IC_sup)*1.08,i,f'p={p_txt}',va='center',fontsize=7.5,color=couleur,fontweight='bold')
            ax.set_yticks(range(len(vars_ok))); ax.set_yticklabels(vars_ok,fontsize=9)
            ax.set_xscale('log'); ax.grid(True,axis='x',alpha=0.3)
            ax.set_title(titres[key],fontsize=11,fontweight='bold')
            ax.set_xlabel(f'Odds Ratio (IC 95%) | AUC = {r["auc"]:.3f}',fontsize=10)
        plt.tight_layout(); plt.show()
    return res


def analyses_mutations(df, cfg, mutations=None):
    """Régressions par mutation individuelle."""
    if mutations is None: mutations=['EGFR','MET','ALK','ERBB2','ROS1']
    print("\n"+"█"*65+"\nANALYSES PAR MUTATION\n"+"█"*65)
    print("\n── Effectifs ──")
    vars_mod = _vars_modele(cfg)
    for mut in MUTATIONS_NF:
        if f'mutation_{mut}' in df.columns:
            n=porte_mutation(df,mut).sum(); epv=n/max(1,len(vars_mod))
            print(f"  {mut:<8}: N+={n:>4} | EPV≈{epv:.1f} {'✅' if epv>=10 else '⚠️' if epv>=5 else '❌'}")
    res={}
    for mut in mutations:
        df_m=df.copy(); df_m['outcome']=porte_mutation(df_m,mut).astype(int)
        n_pos=df_m['outcome'].sum()
        d=_filtrer(df_m,cfg)
        vars_ok=[v for v in vars_mod if v in d.columns]
        d=d[vars_ok+['outcome']].dropna()
        if len(d)<30 or d['outcome'].sum()<5:
            print(f"⚠️  {mut} : effectif insuffisant"); res[mut]=None; continue
        epv=d['outcome'].sum()/len(vars_ok)
        titre=f'Mutation {mut} (n+={n_pos}, EPV={epv:.1f})'
        print(f"\n{'='*60}\n{titre}\n{'='*60}")
        scaler=StandardScaler(); d_std=d.copy(); d_std[vars_ok]=scaler.fit_transform(d[vars_ok])
        vars_ok=[v for v in vars_ok if d_std[v].std()>1e-8]
        n_neg_m=len(d)-d['outcome'].sum()
        use_firth=(epv<10) or (n_neg_m/max(1,d['outcome'].sum())>5)
        if use_firth:
            print(f"  → Régression de Firth activée (EPV={epv:.1f})")
            _,tab,auc=_run_regression_firth(d,vars_ok,f'Mutation {mut} (Firth)',COULEURS.get(mut,'#2E86AB'))
            if tab is None: res[mut]=None; continue
            res[mut]={'modele':None,'tableau':tab,'auc':auc,'n_pos':n_pos,'epv':epv,'methode':'Firth'}
            continue
        X=sm.add_constant(d_std[vars_ok]); y=d_std['outcome']
        try: modele=sm.Logit(y,X).fit(disp=False)
        except:
            try: modele=sm.Logit(y,X).fit(disp=False,method='bfgs',maxiter=200)
            except Exception as e2: print(f"⚠️  {mut}: {e2}"); res[mut]=None; continue
        tab=_tableau_OR(modele,vars_ok)
        try:
            from IPython.display import display; display(tab.drop(columns=['p_num']))
        except: print(tab.drop(columns=['p_num']).to_string(index=False))
        auc=_plot_roc_forest(modele,X,y,vars_ok,titre,COULEURS.get(mut,'#2E86AB'))
        print(f"AUC={auc:.3f} | R²={1-(modele.llf/modele.llnull):.3f}")
        res[mut]={'modele':modele,'tableau':tab,'auc':auc,'n_pos':n_pos,'epv':epv}
    return res


def analyse_stratifiee_sexe(df, cfg, groupe='AB'):
    """Analyse séparément chez les femmes et les hommes."""
    print(f"\n"+"█"*65+f"\nSTRATIFICATION PAR SEXE — {groupe}\n"+"█"*65)
    col_map={'AB':('groupe_AB','Groupe A','Groupe B'),'CD':('groupe_CD','Groupe C','Groupe D'),
             'ACBD':('groupe_AC_BD','Groupe A+C','Groupe B+D')}
    col_g,g1,g2 = col_map[groupe]
    titres={'AB':'A vs B — Mutations NF','CD':'C vs D','ACBD':'A+C vs B+D'}
    paquet_map={'AB':True,'CD':False,'ACBD':False}
    res={}
    for sexe,couleur in [('feminin','#E76F51'),('masculin','#2E86AB')]:
        cfg_s={**cfg,'filtre_sexe':sexe,'inclure_sexe':False}
        vars_mod=_vars_modele(cfg_s,inclure_paquet=paquet_map[groupe])
        modele,tab,auc=_run_regression(df,vars_mod,col_g,g1,g2,
            f'{titres[groupe]} — {sexe.capitalize()}',couleur,cfg=cfg_s)
        res[sexe]={'modele':modele,'tableau':tab,'auc':auc}
    # Comparaison
    if res['feminin']['tableau'] is not None and res['masculin']['tableau'] is not None:
        t_f=res['feminin']['tableau'].set_index('Variable')
        t_h=res['masculin']['tableau'].set_index('Variable')
        print("\n── Comparaison OR Femmes vs Hommes ──")
        rows=[{'Variable':v,'OR_F':t_f.loc[v,'OR'],'p_F':t_f.loc[v,'p-value'],'sig_F':t_f.loc[v,'Sig'],
               'OR_H':t_h.loc[v,'OR'],'p_H':t_h.loc[v,'p-value'],'sig_H':t_h.loc[v,'Sig'],
               'Divergence':'⚠️' if t_f.loc[v,'Sig']!=t_h.loc[v,'Sig'] else '—'}
              for v in t_f.index.intersection(t_h.index)]
        df_comp=pd.DataFrame(rows)
        try:
            from IPython.display import display; display(df_comp)
        except: print(df_comp.to_string(index=False))
    return res


def analyse_sous_groupe(df, cfg, filtre_fumeur=None, filtre_histologie=None,
                         filtre_sexe=None, groupe='AB', titre_custom=None):
    """Analyse dans un sous-groupe spécifique."""
    col_map={'AB':('groupe_AB','Groupe A','Groupe B','#2E86AB'),
             'CD':('groupe_CD','Groupe C','Groupe D','#52B788'),
             'ACBD':('groupe_AC_BD','Groupe A+C','Groupe B+D','#7B2D8B')}
    col_g,g1,g2,couleur = col_map[groupe]
    paquet_map={'AB':True,'CD':False,'ACBD':False}
    cfg_s={**cfg,'filtre_fumeur':filtre_fumeur,'filtre_histologie':filtre_histologie,'filtre_sexe':filtre_sexe}
    titre = titre_custom or f'{groupe} — Sous-groupe'
    vars_mod=_vars_modele(cfg_s,inclure_paquet=paquet_map[groupe])
    modele,tab,auc=_run_regression(df,vars_mod,col_g,g1,g2,titre,couleur,cfg=cfg_s)
    return {'modele':modele,'tableau':tab,'auc':auc}


def dose_reponse_quintiles(df, cfg, polluant, groupe='AB', n_quintiles=5):
    """Dose-réponse par quintile d'exposition."""
    col_map={'AB':('groupe_AB','Groupe A','Groupe B',True,'#2E86AB'),
             'CD':('groupe_CD','Groupe C','Groupe D',False,'#52B788'),
             'ACBD':('groupe_AC_BD','Groupe A+C','Groupe B+D',False,'#7B2D8B')}
    col_g,g1,g2,paquet,couleur=col_map[groupe]
    d=_filtrer(df,cfg); d=d[d[col_g].isin([g1,g2])].copy(); d['outcome']=(d[col_g]==g1).astype(int)
    if polluant not in d.columns: print(f"⚠️  '{polluant}' absent."); return None
    d['quintile']=pd.qcut(d[polluant],q=n_quintiles,labels=[f'Q{i+1}' for i in range(n_quintiles)],duplicates='drop')
    d=d.dropna(subset=['quintile'])
    cfg_cov={**cfg,'cumul':[],'tendance':[],'pct_pm25':[],'pct_pm10':[],'pct_o3':[],'inclure_paquet':paquet}
    covariables=[v for v in _vars_modele(cfg_cov,inclure_paquet=paquet) if v in d.columns]
    dummy_df=pd.get_dummies(d[['quintile']],prefix='Q',drop_first=True); quintile_cols=list(dummy_df.columns)
    d=pd.concat([d,dummy_df],axis=1)
    df_m=d[quintile_cols+covariables+['outcome']].apply(pd.to_numeric,errors='coerce').dropna()
    scaler=StandardScaler(); df_std=df_m.copy(); df_std[covariables]=scaler.fit_transform(df_m[covariables])
    X=sm.add_constant(df_std[quintile_cols+covariables]).astype(float)
    y=df_std['outcome'].astype(float)
    mask=np.isfinite(X).all(axis=1)&np.isfinite(y); X,y=X[mask],y[mask]
    modele=sm.Logit(y,X).fit(disp=False); tab_q=_tableau_OR(modele,quintile_cols)
    print(f"\n── Dose-réponse {polluant} — {groupe} (Référence : Q1) ──")
    try:
        from IPython.display import display; display(tab_q.drop(columns=['p_num']))
    except: print(tab_q.drop(columns=['p_num']).to_string(index=False))
    # Test tendance
    d_t=d.copy(); d_t['qnum']=d['quintile'].cat.codes+1
    df_t=d_t[['qnum']+covariables+['outcome']].apply(pd.to_numeric,errors='coerce').dropna()
    scaler2=StandardScaler(); df_t_std=df_t.copy(); df_t_std[covariables]=scaler2.fit_transform(df_t[covariables])
    X_t=sm.add_constant(df_t_std[['qnum']+covariables]).astype(float); y_t=df_t_std['outcome'].astype(float)
    mask_t=np.isfinite(X_t).all(axis=1)&np.isfinite(y_t)
    m_t=sm.Logit(y_t[mask_t],X_t[mask_t]).fit(disp=False)
    p_t=float(m_t.pvalues['qnum']); or_t=float(np.exp(m_t.params['qnum']))
    print(f"\n  Test tendance : OR={or_t:.3f} | p={'<0.001' if p_t<0.001 else f'{p_t:.3f}'} {'✅' if p_t<0.05 else '—'}")
    # Visualisation
    labels_q=['Q1 (réf)']+[f'Q{i+2}' for i in range(len(quintile_cols))]
    ors_q=[1.0]+[float(np.exp(modele.params[c])) for c in quintile_cols]
    ic_inf_q=[1.0]+[float(np.exp(modele.conf_int().loc[c,0])) for c in quintile_cols]
    ic_sup_q=[1.0]+[float(np.exp(modele.conf_int().loc[c,1])) for c in quintile_cols]
    pvals_q=[1.0]+[float(modele.pvalues[c]) for c in quintile_cols]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.axhline(y=1,color='black',linestyle='--',lw=1.2)
    for i,(lab,OR,lo,hi,p) in enumerate(zip(labels_q,ors_q,ic_inf_q,ic_sup_q,pvals_q)):
        c=couleur if p<0.05 else '#AAAAAA'
        ax.errorbar(i,OR,yerr=[[OR-lo],[hi-OR]],fmt='o',color=c,markersize=9,capsize=5,lw=2)
    ax.set_xticks(range(len(labels_q))); ax.set_xticklabels(labels_q)
    ax.set_ylabel('Odds Ratio (IC 95%)'); ax.set_yscale('log')
    ax.set_title(f'Dose-réponse {polluant} — {groupe}',fontweight='bold'); ax.grid(True,axis='y',alpha=0.3)
    plt.tight_layout(); plt.show()
    return {'modele':modele,'tableau':tab_q,'p_trend':p_t,'or_trend':or_t}


def dose_reponse_spatiale(df, cfg, col_distance, mutation=None, groupe='AB', tranches_km=None):
    """OR par tranche de distance à un site ICPE."""
    if tranches_km is None: tranches_km=[0,1,3,5,10,999]
    col_map={'AB':('groupe_AB','Groupe A','Groupe B',True,'#2E86AB'),
             'CD':('groupe_CD','Groupe C','Groupe D',False,'#52B788'),
             'ACBD':('groupe_AC_BD','Groupe A+C','Groupe B+D',False,'#7B2D8B')}
    d=_filtrer(df,cfg)
    if mutation:
        d=d.copy(); d['outcome']=porte_mutation(d,mutation).astype(int)
        titre=f'Dose-réponse spatiale — {mutation} × {col_distance}'; couleur=COULEURS.get(mutation,'#2E86AB')
        paquet=True
    else:
        col_g,g1,g2,paquet,couleur=col_map[groupe]
        d=d[d[col_g].isin([g1,g2])].copy(); d['outcome']=(d[col_g]==g1).astype(int)
        titre=f'Dose-réponse spatiale — {groupe} × {col_distance}'
    if col_distance not in d.columns: print(f"⚠️  '{col_distance}' absent."); return None
    tranches_m=[t*1000 for t in tranches_km]
    labels=[f'{tranches_km[i]}–{tranches_km[i+1]} km' for i in range(len(tranches_km)-1)]
    d['tranche']=pd.cut(d[col_distance],bins=tranches_m,labels=labels); d=d.dropna(subset=['tranche','outcome'])
    cfg_cov={**cfg,'cumul':[],'tendance':[],'pct_pm25':[],'pct_pm10':[],'pct_o3':[],'inclure_paquet':paquet}
    covariables=[v for v in _vars_modele(cfg_cov,inclure_paquet=paquet) if v in d.columns]
    ref_label=labels[-1]; ref_df=d[d['tranche']==ref_label]
    results_d=[]
    for label in labels[:-1]:
        sub=pd.concat([ref_df,d[d['tranche']==label]])[covariables+['outcome']].apply(pd.to_numeric,errors='coerce').dropna()
        if len(sub)<10 or sub['outcome'].sum()<3: continue
        X_d=sm.add_constant(sub[covariables])
        try:
            m=sm.Logit(sub['outcome'],X_d).fit(disp=False)
            results_d.append({'Tranche':label,'N':len(d[d['tranche']==label]),
                'OR':round(float(np.exp(m.params['const'])),3),'IC inf':round(float(np.exp(m.conf_int().loc['const',0])),3),
                'IC sup':round(float(np.exp(m.conf_int().loc['const',1])),3),
                'p-value':'<0.001' if float(m.pvalues['const'])<0.001 else f"{float(m.pvalues['const']):.3f}",
                'Sig':'✅' if float(m.pvalues['const'])<0.05 else '—'})
        except Exception: continue
    if not results_d: print("⚠️  Effectifs insuffisants."); return None
    df_dist=pd.DataFrame(results_d)
    print(f"\n{titre} (référence : {ref_label})")
    try:
        from IPython.display import display; display(df_dist)
    except: print(df_dist.to_string(index=False))
    fig,ax=plt.subplots(figsize=(9,5))
    ax.axhline(y=1,color='black',linestyle='--',lw=1.2)
    for i,row in df_dist.iterrows():
        c=couleur if row['Sig']=='✅' else '#AAAAAA'
        ax.plot([i,i],[row['IC inf'],row['IC sup']],color=c,lw=2.5); ax.plot(i,row['OR'],'o',color=c,markersize=10)
    ax.set_xticks(range(len(df_dist))); ax.set_xticklabels(df_dist['Tranche'],rotation=20,ha='right')
    ax.set_ylabel(f'OR vs {ref_label} (IC 95%)'); ax.set_title(titre,fontweight='bold'); ax.grid(True,axis='y',alpha=0.3)
    plt.tight_layout(); plt.show()
    return df_dist


def analyse_pct_temps(df, cfg):
    """Teste chaque variable % du temps séparément dans un modèle ajusté."""
    GROUPES=[('groupe_AB','Groupe A','Groupe B',True,'A vs B'),
             ('groupe_CD','Groupe C','Groupe D',False,'C vs D'),
             ('groupe_AC_BD','Groupe A+C','Groupe B+D',False,'A+C vs B+D')]
    vars_pct=[c for c in df.columns if '_pct_sup' in c]
    if not vars_pct: print("⚠️ Aucune variable %."); return None
    resultats=[]
    for col_g,g1,g2,paquet,label_g in GROUPES:
        cfg_cov={**cfg,'cumul':[],'tendance':[],'pct_pm25':[],'pct_pm10':[],'pct_o3':[],'inclure_paquet':paquet}
        covariables=[v for v in _vars_modele(cfg_cov,inclure_paquet=paquet) if v in df.columns]
        df_sub=df[df[col_g].isin([g1,g2])].copy(); df_sub['outcome']=(df_sub[col_g]==g1).astype(int)
        for var_pct in vars_pct:
            vars_dispo=[v for v in [var_pct]+covariables if v in df_sub.columns]
            df_m=df_sub[['outcome']+vars_dispo].dropna()
            if len(df_m)<50 or df_m['outcome'].sum()<10: continue
            vars_cont=[v for v in vars_dispo if df_m[v].nunique()>5]
            scaler=StandardScaler(); df_std=df_m.copy(); df_std[vars_cont]=scaler.fit_transform(df_m[vars_cont])
            X=sm.add_constant(df_std[vars_dispo]); y=df_std['outcome']
            try:
                mod=sm.Logit(y,X).fit(disp=False); p=float(mod.pvalues[var_pct])
                resultats.append({'Groupe':label_g,'Variable':var_pct,'OR':round(float(np.exp(mod.params[var_pct])),3),
                    'IC inf':round(float(np.exp(mod.conf_int().loc[var_pct,0])),3),
                    'IC sup':round(float(np.exp(mod.conf_int().loc[var_pct,1])),3),
                    'p-value':'<0.001' if p<0.001 else f'{p:.3f}','AUC':round(roc_auc_score(y,mod.predict(X)),3),
                    'Sig':'✅' if p<0.05 else '—','p_num':p})
            except Exception: pass
    df_res=pd.DataFrame(resultats)
    try:
        from IPython.display import display; display(df_res.drop(columns=['p_num'],errors='ignore'))
    except: print(df_res.drop(columns=['p_num'],errors='ignore').to_string(index=False))
    sig=df_res[df_res['Sig']=='✅']
    if len(sig)>0:
        fig,ax=plt.subplots(figsize=(10,max(4,len(sig)*0.5)))
        ax.axvline(x=1,color='black',linestyle='--',lw=1.2)
        cg={'A vs B':COULEURS['AB'],'C vs D':COULEURS['CD'],'A+C vs B+D':COULEURS['ACBD']}
        for i,row in sig.reset_index().iterrows():
            c=cg.get(row['Groupe'],'#666666')
            ax.plot([row['IC inf'],row['IC sup']],[i,i],color=c,lw=2); ax.plot(row['OR'],i,'o',color=c,markersize=9)
            ax.text(row['IC sup']*1.05,i,f"{row['Variable']} ({row['Groupe']}) p={row['p-value']}",va='center',fontsize=8,color=c)
        ax.set_yticks(range(len(sig))); ax.set_yticklabels([f"{r['Variable']}\n{r['Groupe']}" for _,r in sig.iterrows()],fontsize=8)
        ax.set_xscale('log'); ax.set_title('Variables % du temps — OR significatifs',fontsize=12,fontweight='bold')
        plt.tight_layout(); plt.show()
    else: print("\n— Aucune variable % significative.")
    return df_res



def lasso_selection(df, cfg, groupe='AB', alpha_values=None):
    """
    Sélection de variables par régression pénalisée L1 (Lasso).
    Filtre automatiquement selon cfg['polluants_retenus'].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if alpha_values is None:
        alpha_values = np.logspace(-3, 1, 30)

    col_map={'AB':('groupe_AB','Groupe A','Groupe B'),
             'CD':('groupe_CD','Groupe C','Groupe D'),
             'ACBD':('groupe_AC_BD','Groupe A+C','Groupe B+D')}
    col_g,g1,g2=col_map[groupe]; couleur=COULEURS.get(groupe,'#2E86AB')

    # Variables d'exposition filtrées selon polluants_retenus
    polluants=cfg.get('polluants_retenus',['PM25','PM10','NO2','O3'])
    vars_exp=[]
    for v in cfg.get('cumul',[]):
        if v.split('_')[0] in polluants: vars_exp.append(v)
    for v in cfg.get('tendance',[]):
        if v.split('_')[0] in polluants: vars_exp.append(v)
    if 'PM25' in polluants: vars_exp+=[f'PM25_pct_sup{s}' for s in cfg.get('pct_pm25',[])]
    if 'PM10' in polluants: vars_exp+=[f'PM10_pct_sup{s}' for s in cfg.get('pct_pm10',[])]
    if 'NO2'  in polluants: vars_exp+=[f'NO2_pct_sup{s}'  for s in cfg.get('pct_no2',[])]
    if 'O3'   in polluants: vars_exp+=[f'O3_pct_sup{s}'   for s in cfg.get('pct_o3',[])]
    vars_exp+=cfg.get('icpe',[])

    d=_filtrer(df,cfg)
    d=d[d[col_g].isin([g1,g2])].copy(); d['outcome']=(d[col_g]==g1).astype(int)
    vars_ok=[v for v in vars_exp if v in d.columns]
    d=d[vars_ok+['outcome']].dropna()
    if len(d)<50: print("⚠️  Effectif insuffisant pour Lasso."); return None

    scaler=StandardScaler(); X=scaler.fit_transform(d[vars_ok]); y=d['outcome'].values
    n_pos=y.sum(); cw='balanced' if (len(y)-n_pos)/max(1,n_pos)>3 else None

    print(f"\n{'='*60}\nLASSO L1 — {groupe} | Polluants : {polluants}\n{'='*60}")
    print(f"N={len(d)} | n+={n_pos} | Variables : {len(vars_ok)}")

    resultats_lasso=[]; auc_cv=[]
    for alpha in alpha_values:
        C=1.0/(alpha*len(d))
        clf=LogisticRegression(penalty='l1',C=C,solver='liblinear',class_weight=cw,max_iter=500,random_state=42)
        clf.fit(X,y)
        auc_cv.append(cross_val_score(clf,X,y,cv=5,scoring='roc_auc').mean())
        for var,coef in zip(vars_ok,clf.coef_[0]):
            resultats_lasso.append({'alpha':alpha,'Variable':var,'Coef':coef,'Selectionne':coef!=0})

    df_lasso=pd.DataFrame(resultats_lasso)
    best_idx=np.argmax(auc_cv); best_alpha=alpha_values[best_idx]; best_auc=auc_cv[best_idx]
    vars_select=df_lasso[(df_lasso['alpha']==best_alpha)&(df_lasso['Selectionne'])]['Variable'].tolist()

    print(f"\nMeilleur alpha={best_alpha:.4f} | AUC CV={best_auc:.3f}")
    print(f"Variables sélectionnées ({len(vars_select)}) :")
    for v in vars_select:
        coef=df_lasso[(df_lasso['alpha']==best_alpha)&(df_lasso['Variable']==v)]['Coef'].values[0]
        print(f"  ✅ {v:<35} coef={coef:+.4f}")
    if not vars_select: print("  — Aucune variable sélectionnée")

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'Lasso L1 — {groupe} | Polluants : {polluants}',fontsize=13,fontweight='bold')
    ax=axes[0]
    for var in vars_ok:
        d_var=df_lasso[df_lasso['Variable']==var].sort_values('alpha')
        sel=var in vars_select
        ax.plot(np.log10(d_var['alpha']),d_var['Coef'],
                lw=2 if sel else 0.8,alpha=1 if sel else 0.25,
                label=var if sel else None,color=couleur if sel else '#CCCCCC')
    ax.axhline(y=0,color='black',linestyle='--',lw=1)
    ax.axvline(x=np.log10(best_alpha),color='red',linestyle='--',lw=1.5,label='λ optimal')
    ax.set_xlabel('log10(alpha)'); ax.set_ylabel('Coefficient Lasso')
    ax.set_title('Chemin de régularisation')
    if vars_select: ax.legend(fontsize=8,loc='upper right')
    ax.grid(True,alpha=0.3)
    axes[1].plot(np.log10(alpha_values),auc_cv,color=couleur,lw=2)
    axes[1].axvline(x=np.log10(best_alpha),color='red',linestyle='--',lw=1.5,
                    label=f'λ optimal (AUC={best_auc:.3f})')
    axes[1].set_xlabel('log10(alpha)'); axes[1].set_ylabel('AUC (CV 5 folds)')
    axes[1].set_title('AUC CV selon lambda'); axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()
    return {'df_lasso':df_lasso,'vars_selectionnees':vars_select,'best_alpha':best_alpha,'auc_cv':best_auc}


# ═══════════════════════════════════════════════════════════════════════════════
# NOUVELLES FONCTIONS — Ajoutées à lungcancair_analyses.py
# ═══════════════════════════════════════════════════════════════════════════════

import copy


def lasso_selection_exposition(df, cfg, groupe='AB', alpha_values=None):
    """
    Lasso L1 pour sélectionner les variables d'EXPOSITION uniquement.
    Les covariables cliniques (tabac, sexe, âge, EDI, trafic) sont
    FORCÉES dans le modèle — seules les variables d'exposition sont pénalisées.

    Logique :
      1. Modèle avec covariables cliniques forcées + exposition pénalisée L1
      2. Lambda optimal par validation croisée
      3. Variables sélectionnées = exposition avec coef non nul
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if alpha_values is None:
        alpha_values = np.logspace(-3, 1, 30)

    col_map = {'AB':   ('groupe_AB',    'Groupe A',   'Groupe B'),
               'CD':   ('groupe_CD',    'Groupe C',   'Groupe D'),
               'ACBD': ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D')}
    col_g, g1, g2 = col_map[groupe]
    couleur = COULEURS.get(groupe, '#2E86AB')
    paquet_in = {'AB': True, 'CD': False, 'ACBD': False}[groupe]

    # Variables d'exposition (pénalisées par Lasso)
    polluants = cfg.get('polluants_retenus', ['PM25','O3'])
    vars_expo = []
    for pol in polluants:
        cols_cum = [c for c in df.columns
                    if c.startswith(pol+'_cumul_') and 'pct' not in c and 'manq' not in c]
        if cols_cum: vars_expo.append(cols_cum[0])
        if pol+'_mm365_sen_pente' in df.columns: vars_expo.append(pol+'_mm365_sen_pente')
        key_map = {'PM25':'pct_pm25','PM10':'pct_pm10','O3':'pct_o3','NO2':'pct_no2'}
        for s in cfg.get(key_map.get(pol,''), []):
            col_pct = pol+'_pct_sup'+str(s)
            if col_pct in df.columns: vars_expo.append(col_pct)
    for col in cfg.get('icpe', []):
        if col in df.columns: vars_expo.append(col)
    # IPG — ajouté à l'exposition si inclure_ipg=True et IPG disponible
    if cfg.get('inclure_ipg', False) and 'IPG' in df.columns:
        vars_expo.append('IPG')
    vars_expo = list(dict.fromkeys(vars_expo))

    # Covariables cliniques (forcées — non pénalisées)
    vars_clin = []
    if cfg.get('inclure_age', True)    and 'age_diagnostic'  in df.columns: vars_clin.append('age_diagnostic')
    if paquet_in and cfg.get('inclure_paquet', True) and 'paquet_annee' in df.columns: vars_clin.append('paquet_annee')
    if cfg.get('inclure_sexe', True)   and 'sexe_bin'        in df.columns: vars_clin.append('sexe_bin')
    if cfg.get('inclure_edi', True)    and 'quintileEDI2021' in df.columns: vars_clin.append('quintileEDI2021')
    if cfg.get('inclure_trafic', True) and 'indice_trafic'   in df.columns: vars_clin.append('indice_trafic')

    d = _filtrer(df, cfg)
    d = d[d[col_g].isin([g1, g2])].copy()
    d['outcome'] = (d[col_g] == g1).astype(int)
    vars_ok_clin = [v for v in vars_clin if v in d.columns]
    vars_ok_expo = [v for v in vars_expo if v in d.columns]
    all_ok = vars_ok_clin + vars_ok_expo
    d = d[all_ok + ['outcome']].dropna()
    if len(d) < 50: print('Effectif insuffisant.'); return None

    n_pos = int(d['outcome'].sum())
    n_neg = len(d) - n_pos
    cw = 'balanced' if n_neg / max(1, n_pos) > 3 else None

    sep = '='*65
    print(f'\n{sep}')
    print(f'LASSO — EXPOSITION — {groupe}')
    print(f'{sep}')
    print(f'N={len(d)} | n+={n_pos} | Polluants : {polluants}')
    print(f'Covariables forcées ({len(vars_ok_clin)}) : {vars_ok_clin}')
    print(f'Variables exposition ({len(vars_ok_expo)}) : {vars_ok_expo}')

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(d[all_ok])
    X_clin = X_all[:, :len(vars_ok_clin)]
    X_expo = X_all[:, len(vars_ok_clin):]
    y      = d['outcome'].values

    resultats_lasso = []
    auc_cv = []

    for alpha in alpha_values:
        C_expo = 1.0 / (alpha * len(d))
        X_concat = np.column_stack([X_clin, X_expo]) if X_expo.shape[1] > 0 else X_clin
        clf = LogisticRegression(penalty='l1', C=C_expo, solver='liblinear',
                                  class_weight=cw, max_iter=500, random_state=42)
        try:
            clf.fit(X_concat, y)
            cv = cross_val_score(clf, X_concat, y, cv=5, scoring='roc_auc')
            auc_cv.append(cv.mean())
            coefs_expo = clf.coef_[0][len(vars_ok_clin):]
            for var, coef in zip(vars_ok_expo, coefs_expo):
                resultats_lasso.append({'alpha':alpha,'Variable':var,'Coef':coef,'Selectionne':coef!=0})
        except Exception:
            auc_cv.append(0.5)

    if not resultats_lasso: print('Aucun résultat.'); return None

    df_lasso  = pd.DataFrame(resultats_lasso)
    best_idx  = np.argmax(auc_cv)
    best_alpha= alpha_values[best_idx]
    best_auc  = auc_cv[best_idx]
    vars_select = df_lasso[(df_lasso['alpha']==best_alpha)&(df_lasso['Selectionne'])]['Variable'].tolist()

    print(f'\nMeilleur alpha={best_alpha:.4f} | AUC CV={best_auc:.3f}')
    print(f'Variables exposition sélectionnées ({len(vars_select)}) :')
    for v in vars_select:
        coef = df_lasso[(df_lasso['alpha']==best_alpha)&(df_lasso['Variable']==v)]['Coef'].values[0]
        print(f'  {v:<40} coef={coef:+.4f}')
    if not vars_select:
        print('  Aucune variable exposition sélectionnée')
        print('  (covariables cliniques suffisent)')

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Lasso exposition — {groupe}', fontsize=13, fontweight='bold')
    ax = axes[0]
    for var in vars_ok_expo:
        d_var = df_lasso[df_lasso['Variable']==var].sort_values('alpha')
        sel   = var in vars_select
        ax.plot(np.log10(d_var['alpha']), d_var['Coef'],
                lw=2 if sel else 0.8, alpha=1 if sel else 0.25,
                label=var if sel else None,
                color=couleur if sel else '#CCCCCC')
    ax.axhline(y=0,color='black',linestyle='--',lw=1)
    ax.axvline(x=np.log10(best_alpha),color='red',linestyle='--',lw=1.5,label='lambda opt')
    ax.set_xlabel('log10(alpha)'); ax.set_ylabel('Coef Lasso (exposition)')
    ax.set_title('Chemin régularisation — Exposition')
    if vars_select: ax.legend(fontsize=8)
    ax.grid(True,alpha=0.3)
    axes[1].plot(np.log10(alpha_values),auc_cv,color=couleur,lw=2)
    axes[1].axvline(x=np.log10(best_alpha),color='red',linestyle='--',lw=1.5,
                    label=f'lambda opt (AUC={best_auc:.3f})')
    axes[1].set_xlabel('log10(alpha)'); axes[1].set_ylabel('AUC CV')
    axes[1].set_title('AUC CV selon lambda'); axes[1].legend(fontsize=9)
    axes[1].grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()

    return {'df_lasso':df_lasso,'vars_selectionnees':vars_select,
            'vars_cliniques_forcees':vars_ok_clin,
            'best_alpha':best_alpha,'auc_cv':best_auc}

def analyser_collinearite_icpe(df_final, cfg, seuil_r=0.85, seuil_vif=5.0):
    """
    Analyse la collinéarité entre les variables ICPE et nettoie CFG['icpe'].

    Problème courant : nb_ICPE_total_3km ≈ nb_ICPE_NS_3km (corrélation > 0.99)
    → le modèle devient instable (IC aberrants [0.005 ; 258])

    Étape 1 — Matrice de corrélation : toutes les paires d'ICPE
    Étape 2 — VIF (Variance Inflation Factor) : mesure le "gonflement"
              de la variance dû à la multicolinéarité
              VIF > 5 = problématique | VIF > 10 = très problématique
    Étape 3 — Suppression automatique des redondants selon une règle de priorité :
              distance ICPE > count ICPE spécifique > count ICPE total
              (le total est toujours la somme des spécifiques → redondant)

    Met à jour CFG['icpe'] directement et retourne la liste nettoyée.

    Paramètres :
        seuil_r   : seuil de corrélation au-delà duquel une variable est redondante (défaut 0.85)
        seuil_vif : seuil VIF au-delà duquel une variable est problématique (défaut 5.0)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vars_icpe = cfg.get('icpe', [])
    vars_dispo = [v for v in vars_icpe if v in df_final.columns]
    if len(vars_dispo) < 2:
        print("⚠️  Moins de 2 variables ICPE disponibles — rien à analyser.")
        return vars_dispo

    df_sub = df_final[vars_dispo].dropna()

    print("█"*60)
    print("COLLINÉARITÉ ICPE — ANALYSE ET NETTOYAGE AUTOMATIQUE")
    print("█"*60)
    print(f"Variables initiales ({len(vars_dispo)}) : {vars_dispo}")
    print(f"Seuil corrélation : |r| > {seuil_r} → redondant")
    print(f"Seuil VIF         : VIF > {seuil_vif} → problématique\n")

    # ── Étape 1 : Matrice de corrélation ─────────────────────────────────
    corr = df_sub.corr().round(3)

    print("── Matrice de corrélation ──")
    try:
        from IPython.display import display
        # Colorer les cases selon intensité
        styled = corr.style.background_gradient(cmap='RdYlGn_r', vmin=-1, vmax=1).format("{:.2f}")
        display(styled)
    except Exception:
        print(corr.to_string())

    # Paires problématiques
    paires_redondantes = []
    for i in range(len(vars_dispo)):
        for j in range(i+1, len(vars_dispo)):
            r = abs(corr.iloc[i, j])
            if r >= seuil_r:
                paires_redondantes.append((vars_dispo[i], vars_dispo[j], round(r, 3)))

    if paires_redondantes:
        print(f"\n⚠️  Paires très corrélées (|r| ≥ {seuil_r}) :")
        for v1, v2, r in sorted(paires_redondantes, key=lambda x: -x[2]):
            print(f"   {v1:35s} ↔  {v2:35s}  r = {r:.3f}")
    else:
        print(f"\n✅ Aucune paire corrélée au-dessus de {seuil_r}")

    # ── Étape 2 : VIF ────────────────────────────────────────────────────
    print("\n── VIF (Variance Inflation Factor) ──")
    try:
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(df_sub)
        vif_vals = {
            vars_dispo[i]: round(variance_inflation_factor(X_scaled, i), 2)
            for i in range(len(vars_dispo))
        }
        for v, vif in sorted(vif_vals.items(), key=lambda x: -x[1]):
            flag = '❌ TRÈS FORT' if vif > 10 else ('⚠️  Fort' if vif > seuil_vif else '✅ OK')
            print(f"   {v:40s}  VIF = {vif:7.2f}  {flag}")
    except Exception as e:
        print(f"   ⚠️  VIF non calculable : {e}")
        vif_vals = {}

    # ── Étape 3 : Nettoyage automatique ──────────────────────────────────
    # Règle de priorité : plus spécifique > plus général
    # distance (SH/SB) > count spécifique (SH, SB, NS) > count total
    def _priorite(v):
        if v.startswith('dist_'):       return 0   # distances = plus informatives
        if 'SH' in v:                   return 1   # Seveso HH
        if 'SB' in v:                   return 2   # Seveso SB
        if 'NS' in v:                   return 3   # Non-Seveso
        if 'total' in v:                return 4   # total = somme des autres → redondant
        return 5

    a_garder = list(vars_dispo)     # part de la liste complète
    supprimees = []
    raisons    = {}

    # Passe 1 : supprimer par corrélation
    for v1, v2, r in sorted(paires_redondantes, key=lambda x: -x[2]):
        if v1 not in a_garder or v2 not in a_garder:
            continue
        # Garder celle avec la plus haute priorité (indice le plus bas)
        if _priorite(v1) <= _priorite(v2):
            victime, survivant = v2, v1
        else:
            victime, survivant = v1, v2
        a_garder.remove(victime)
        supprimees.append(victime)
        raisons[victime] = f"r={r:.3f} avec {survivant} → garder {survivant} (plus spécifique)"

    # Passe 2 : supprimer par VIF élevé (si encore dans la liste)
    if vif_vals:
        for v, vif in sorted(vif_vals.items(), key=lambda x: -x[1]):
            if vif > seuil_vif and v in a_garder and v not in supprimees:
                # Ne supprimer par VIF que si pas déjà nettoyé par corrélation
                # et seulement si c'est une variable "totale" ou redondante
                if _priorite(v) >= 4:
                    a_garder.remove(v)
                    supprimees.append(v)
                    raisons[v] = f"VIF={vif:.1f} > {seuil_vif} (variable agrégée redondante)"

    # ── Résumé ───────────────────────────────────────────────────────────
    print(f"\n── Décision de nettoyage ──")
    if supprimees:
        for v in supprimees:
            print(f"   ❌ SUPPRIMÉE : {v}")
            print(f"      Raison    : {raisons[v]}")
        print(f"\n   ✅ Variables conservées ({len(a_garder)}) : {a_garder}")
    else:
        print("   ✅ Aucune variable supprimée — pas de colinéarité critique.")

    # ── Visualisation : une seule heatmap de corrélation, triangulaire ───
    if len(vars_dispo) >= 2:
        fig, ax = plt.subplots(figsize=(max(6, len(vars_dispo)*1.2),
                                        max(5, len(vars_dispo)*1.0)))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        labels = [v.replace('nb_ICPE_','nb_').replace('dist_ICPE_','dist_')
                  for v in vars_dispo]
        sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    vmin=-1, vmax=1, linewidths=0.5, mask=mask,
                    xticklabels=labels, yticklabels=labels)
        # Surligner en rouge les cases problématiques
        for i in range(len(vars_dispo)):
            for j in range(i):
                if abs(corr.iloc[i, j]) >= seuil_r:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor='red', lw=2.5))
        ax.set_title('Corrélations ICPE (cases rouges = redondantes → supprimées)',
                     fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ── Mise à jour CFG ───────────────────────────────────────────────────
    cfg['icpe'] = a_garder
    print(f"\n✅ CFG[\"icpe\"] mis à jour → {a_garder}")
    print("   Les analyses suivantes (Partie 7, 7b, 8, 9…) utiliseront cette liste nettoyée.")
    return a_garder


def detecter_vars_icpe(df_final, cfg):
    """Détecte automatiquement les colonnes ICPE selon rayons_icpe_m et icpe_types."""
    rayons = cfg.get('rayons_icpe_m', [3000])
    types  = cfg.get('icpe_types', {})
    cols   = []
    for t in ['SH', 'SB']:
        col = f'dist_ICPE_{t}_m'
        if col in df_final.columns: cols.append(col)
    for rayon in rayons:
        lbl = f'{rayon//1000}km' if rayon % 1000 == 0 else f'{rayon}m'
        col_tot = f'nb_ICPE_total_{lbl}'
        if col_tot in df_final.columns: cols.append(col_tot)
        for type_label in types:
            col_t = f'nb_ICPE_{type_label}_{lbl}'
            if col_t in df_final.columns: cols.append(col_t)
    cols = list(dict.fromkeys(cols))
    print(f"Variables ICPE détectées : {cols}")
    return cols


def analyses_monovariees(df, cfg, groupes=None):
    """
    Teste chaque variable séparément (OR brut, AUC, p-value) pour chaque groupe.
    Variables continues  → Mann-Whitney + régression logistique univariée
    Variables catégorielles → Chi² / Fisher
    """
    from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
    from sklearn.preprocessing import StandardScaler as SS

    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B — Mutations NF',          '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D — Non-fumeurs vs Fumeurs', '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D',                     '#7B2D8B'),
        ]

    pol_ret = cfg.get('polluants_retenus', ['PM25', 'PM10', 'NO2', 'O3'])

    def _classer_vars(df, cfg):
        cont, cat = [], []
        for pol in pol_ret:
            cols_cum = [c for c in df.columns
                        if c.startswith(pol+'_cumul_') and 'pct' not in c and 'manq' not in c]
            if cols_cum: cont.append(cols_cum[0])
            col_pente = pol+'_mm365_sen_pente'
            if col_pente in df.columns: cont.append(col_pente)
            key_map = {'PM25':'pct_pm25','PM10':'pct_pm10','NO2':'pct_no2','O3':'pct_o3'}
            key = key_map.get(pol, 'pct_'+pol.lower())
            seuils_r = cfg.get(key, [])
            if seuils_r:
                col_pct = pol+'_pct_sup'+str(seuils_r[0])
                if col_pct in df.columns: cont.append(col_pct)
            col_mk = pol+'_mm365_mk_tendance'
            if col_mk in df.columns: cat.append(col_mk)
        for v in ['age_diagnostic','paquet_annee','quintileEDI2021','indice_trafic','dist_RN_m']:
            if v in df.columns: cont.append(v)
        if 'sexe_bin' in df.columns: cont.append('sexe_bin')
        if 'histologie_groupe' in df.columns: cat.append('histologie_groupe')
        for col in cfg.get('icpe', []):
            if col in df.columns: cont.append(col)
        # IPG — inclus si présent dans df et si inclure_ipg=True
        if cfg.get('inclure_ipg', False) and 'IPG' in df.columns:
            cont.append('IPG')
        # Radon — inclus si présent dans df
        if 'radon_score' in df.columns:
            cont.append('radon_score')
        cont = list(dict.fromkeys(cont))
        cat  = list(dict.fromkeys(cat))
        return cont, cat

    vars_cont, vars_cat = _classer_vars(df, cfg)
    sep = '='*65
    print(f"Variables testées : {len(vars_cont)} continues + {len(vars_cat)} catégorielles")
    print(f"  Exposition  : cumul + tendance + 1 pct par polluant ({pol_ret})")
    print(f"  Clinique    : age, tabac, sexe, EDI, trafic")
    print(f"  Industriel  : {cfg.get('icpe',[])}") 
    resultats_dict = {}

    for col_g, g1, g2, titre, couleur in groupes:
        print(f"\n{'='*65}\n{titre}\n{'='*65}")
        df_sub = df[df[col_g].isin([g1, g2])].copy()
        df_sub['outcome'] = (df_sub[col_g] == g1).astype(int)
        rows = []

        for var in vars_cont:
            if var not in df_sub.columns: continue
            s = df_sub[[var,'outcome']].dropna()
            if len(s) < 20: continue
            s1 = s.loc[s['outcome']==1, var].values
            s2 = s.loc[s['outcome']==0, var].values
            if len(s1) < 5 or len(s2) < 5: continue
            try:
                _, p = mannwhitneyu(s1, s2, alternative='two-sided')
                auc  = roc_auc_score(s['outcome'], s[var])
                X_u  = sm.add_constant(SS().fit_transform(s[[var]]))
                try:
                    mod  = sm.Logit(s['outcome'], X_u).fit(disp=False)
                    OR   = float(np.exp(mod.params[1]))
                    ic_lo= float(np.exp(mod.conf_int().iloc[1,0]))
                    ic_hi= float(np.exp(mod.conf_int().iloc[1,1]))
                except Exception:
                    OR, ic_lo, ic_hi = float('nan'), float('nan'), float('nan')
                rows.append({'Variable':var,'Type':'Continue','Test':'Mann-Whitney',
                    'Médiane g1':round(float(np.median(s1)),3),
                    'Médiane g2':round(float(np.median(s2)),3),
                    'OR_brut':round(OR,3) if not np.isnan(OR) else 'NA',
                    'IC_inf':round(ic_lo,3) if not np.isnan(ic_lo) else 'NA',
                    'IC_sup':round(ic_hi,3) if not np.isnan(ic_hi) else 'NA',
                    'AUC':round(auc,3),'p_num':p,
                    'p-value':'<0.001' if p<0.001 else f'{p:.3f}',
                    'Sig':'✅' if p<0.05 else '—'})
            except Exception: continue

        for var in vars_cat:
            if var not in df_sub.columns: continue
            s = df_sub[[var,'outcome']].dropna()
            if len(s) < 20: continue
            try:
                tab = pd.crosstab(s[var], s['outcome'])
                if tab.shape[0] < 2 or tab.shape[1] < 2: continue
                if tab.values.min() < 5 or tab.shape == (2,2):
                    _, p = fisher_exact(tab.values[:2,:2])
                else:
                    _, p, _, _ = chi2_contingency(tab)
                auc = roc_auc_score(s['outcome'], pd.Categorical(s[var]).codes)
                rows.append({'Variable':var,'Type':'Catégorielle','Test':'Chi²/Fisher',
                    'Médiane g1':'—','Médiane g2':'—','OR_brut':'—','IC_inf':'—','IC_sup':'—',
                    'AUC':round(auc,3),'p_num':p,
                    'p-value':'<0.001' if p<0.001 else f'{p:.3f}',
                    'Sig':'✅' if p<0.05 else '—'})
            except Exception: continue

        df_res = pd.DataFrame(rows).sort_values('p_num')
        try:
            from IPython.display import display; display(df_res.drop(columns=['p_num']))
        except Exception:
            print(df_res.drop(columns=['p_num']).to_string(index=False))
        resultats_dict[titre] = df_res

        # Forest plot + AUC barplot
        df_c = df_res[df_res['Type']=='Continue'].copy().reset_index(drop=True)
        if len(df_c) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df_c)*0.35+1)))
            fig.suptitle(f'Analyses monovariées — {titre}', fontsize=13, fontweight='bold')
            ax = axes[0]
            ax.axvline(x=1, color='black', linestyle='--', lw=1.2)
            for i, row in df_c.iterrows():
                try:
                    OR   = float(row['OR_brut']); lo = float(row['IC_inf']); hi = float(row['IC_sup'])
                    c_   = couleur if row['Sig']=='✅' else '#AAAAAA'
                    ax.plot([lo,hi],[i,i],color=c_,lw=2); ax.plot(OR,i,'o',color=c_,markersize=8)
                    if row['Sig']=='✅':
                        ax.text(hi*1.05,i,f"p={row['p-value']}",va='center',fontsize=7.5,
                                color=couleur,fontweight='bold')
                except Exception: continue
            ax.set_yticks(range(len(df_c))); ax.set_yticklabels(df_c['Variable'].tolist(),fontsize=8)
            ax.set_xscale('log'); ax.set_xlabel('OR brut (IC 95%)'); ax.set_title('Forest plot — OR bruts')
            ax.grid(True,axis='x',alpha=0.3)
            ax2 = axes[1]
            colors_ = [couleur if r=='✅' else '#DDDDDD' for r in df_c['Sig']]
            bars_ = ax2.barh(range(len(df_c)),df_c['AUC'].values,color=colors_,edgecolor='white')
            ax2.axvline(x=0.5,color='red',linestyle='--',lw=1.2,alpha=0.7)
            for i,(bar,auc_v) in enumerate(zip(bars_,df_c['AUC'].values)):
                ax2.text(bar.get_width()+0.002,i,f'{auc_v:.3f}',va='center',fontsize=7.5,fontweight='bold')
            ax2.set_yticks(range(len(df_c))); ax2.set_yticklabels(df_c['Variable'].tolist(),fontsize=8)
            ax2.set_xlabel('AUC univariée'); ax2.set_title('AUC univariée par variable')
            ax2.grid(True,axis='x',alpha=0.3)
            plt.tight_layout(); plt.show()

    return resultats_dict


def trouver_seuil_concentration_optimal(data, df_clinique, df_final, cfg, groupes=None):
    """
    Partie 11e — Trouver C* sur les concentrations journalières brutes.
    Utilise la médiane journalière par patient pour respecter l'indépendance.

    Retourne dict {polluant: {groupe_label: C*}}
    """
    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B',   '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D',   '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D','#7B2D8B'),
        ]

    pol_ret      = cfg.get('polluants_retenus', ['PM25','O3'])
    fenetre_ans  = cfg.get('fenetre_ans', 10)
    fenetre_mois = int(fenetre_ans * 12)

    print(f"\n{'='*65}")
    print("PARTIE 11e — SEUIL OPTIMAL C* SUR DONNÉES JOURNALIÈRES BRUTES")
    print(f"Fenêtre : {fenetre_ans} an(s) | Polluants : {pol_ret}")
    print(f"{'='*65}")

    resultats_c_star = {}

    for pol in pol_ret:
        if pol not in data.columns:
            print(f"⚠️  {pol} absent de data"); continue

        resultats_c_star[pol] = {}
        print(f"\n── Polluant : {pol} ──")

        for col_g, g1, g2, titre, couleur in groupes:
            print(f"\n  {titre}")

            # Construire dataset : médiane journalière par patient + label
            rows_med = []
            for pseudo in df_final['pseudo_provisoire'].unique():
                row_f = df_final[df_final['pseudo_provisoire']==pseudo]
                if len(row_f) == 0: continue
                date_diag  = pd.Timestamp(row_f['date_diagnostic'].iloc[0])
                date_debut = date_diag - pd.DateOffset(months=fenetre_mois)
                groupe_val = row_f[col_g].iloc[0]
                if groupe_val not in [g1, g2]: continue

                d_pat = data[(data['pseudo_provisoire']==pseudo) &
                              (data['date']>=date_debut) &
                              (data['date']< date_diag)][[pol]].dropna()
                if len(d_pat) < 30: continue

                # Médiane journalière = résumé robuste de l'exposition
                med_val = float(d_pat[pol].median())
                rows_med.append({'pseudo':pseudo,'mediane':med_val,
                                  'outcome':1 if groupe_val==g1 else 0})

            if len(rows_med) < 30:
                print(f"  ⚠️  Effectif insuffisant"); continue

            df_med = pd.DataFrame(rows_med)
            n1 = (df_med['outcome']==1).sum()
            n2 = (df_med['outcome']==0).sum()

            # Courbe ROC sur médiane journalière
            try:
                fpr, tpr, thresholds = roc_curve(df_med['outcome'], df_med['mediane'])
                auc = roc_auc_score(df_med['outcome'], df_med['mediane'])
                youden  = tpr - fpr
                best_i  = np.argmax(youden)
                c_star  = round(float(thresholds[best_i]), 1)
                sens    = round(float(tpr[best_i]), 3)
                spec    = round(float(1-fpr[best_i]), 3)
                _, p_mwu = mannwhitneyu(
                    df_med.loc[df_med['outcome']==1,'mediane'],
                    df_med.loc[df_med['outcome']==0,'mediane'],
                    alternative='two-sided')

                resultats_c_star[pol][titre] = {
                    'C_star': c_star, 'AUC': round(auc,3),
                    'Sens': sens, 'Spec': spec,
                    'Youden': round(float(youden[best_i]),3),
                    'p_MWU': '<0.001' if p_mwu<0.001 else f'{p_mwu:.3f}',
                    'Sig': '✅' if p_mwu<0.05 else '—',
                    'N_g1': int(n1), 'N_g2': int(n2),
                    'col_g': col_g, 'g1': g1, 'g2': g2,
                }

                print(f"  C* = {c_star} μg/m³ | AUC={auc:.3f} | "
                      f"Sens={sens:.3f} Spec={spec:.3f} | p={('<0.001' if p_mwu<0.001 else f'{p_mwu:.3f}')} "
                      f"{'✅' if p_mwu<0.05 else '—'}")

                # Visualisation
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                fig.suptitle(f'C* — {pol} | {titre}', fontsize=12, fontweight='bold')

                # Courbe ROC
                axes[0].plot(fpr, tpr, color=couleur, lw=2.5, label=f'AUC={auc:.3f}')
                axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
                lbl_opt = f'C*={c_star} | Sens={sens:.2f} Spec={spec:.2f}'
                axes[0].scatter(fpr[best_i], tpr[best_i], s=120, color='red',
                                zorder=5, label=lbl_opt)
                # Seuils arbitraires sur la ROC
                seuils_arb = cfg.get('seuils_pct',{}).get(pol,[])
                for s_arb in seuils_arb:
                    s_bin_arb = (df_med['mediane'] > s_arb).astype(int)
                    try:
                        fpr_arb = 1 - (s_bin_arb[df_med['outcome']==0]==0).mean()
                        tpr_arb = (s_bin_arb[df_med['outcome']==1]==1).mean()
                        axes[0].scatter(fpr_arb, tpr_arb, s=50, color='orange',
                                        marker='D', zorder=4, alpha=0.7)
                        axes[0].annotate(f'>{s_arb}',
                                         xy=(fpr_arb,tpr_arb),
                                         xytext=(fpr_arb+0.02,tpr_arb-0.03),
                                         fontsize=7, color='darkorange')
                    except Exception: pass

                axes[0].set_xlabel('1 - Spécificité'); axes[0].set_ylabel('Sensibilité')
                axes[0].legend(fontsize=8, loc='lower right'); axes[0].grid(True,alpha=0.3)

                # Violin plot médiane groupe 1 vs groupe 2
                g1_vals = df_med.loc[df_med['outcome']==1,'mediane'].values
                g2_vals = df_med.loc[df_med['outcome']==0,'mediane'].values
                parts = axes[1].violinplot([g1_vals, g2_vals], positions=[0,1],
                                            showmedians=True, showextrema=True)
                for pc in parts['bodies']:
                    pc.set_alpha(0.6)
                axes[1].axhline(y=c_star, color='red', linestyle='--', lw=2,
                                 label=f'C* = {c_star} μg/m³')
                for s_arb in seuils_arb[:3]:
                    axes[1].axhline(y=s_arb, color='orange', linestyle=':', lw=1,
                                     alpha=0.7, label=f'Arbitraire {s_arb}')
                axes[1].set_xticks([0,1]); axes[1].set_xticklabels([f'{g1}\nn={n1}',f'{g2}\nn={n2}'])
                axes[1].set_ylabel(f'{pol} médiane journalière (μg/m³)')
                axes[1].legend(fontsize=8); axes[1].grid(True,axis='y',alpha=0.3)
                plt.tight_layout(); plt.show()

            except Exception as e:
                print(f"  ❌ Erreur : {e}"); continue

    # Tableau synthèse
    rows_synth = []
    for pol, gdict in resultats_c_star.items():
        for titre, res in gdict.items():
            rows_synth.append({'Polluant':pol,'Groupe':titre,
                'C* (μg/m³)':res['C_star'],'AUC':res['AUC'],
                'Sensibilité':res['Sens'],'Spécificité':res['Spec'],
                'p MWU':res['p_MWU'],'Sig':res['Sig']})
    df_synth = pd.DataFrame(rows_synth)
    print(f"\n── Synthèse C* ──")
    try:
        from IPython.display import display; display(df_synth)
    except Exception:
        print(df_synth.to_string(index=False))

    return resultats_c_star


def calculer_pct_seuil_optimal(data, df_clinique, df_final, cfg,
                                 resultats_c_star, groupes=None):
    """
    Partie 11f — Calcule PM25_pct_supC* pour chaque patient,
    puis cherche le seuil de % optimal P* par Youden.

    Retourne df_final enrichi + dict {polluant: {groupe: P*}}
    """
    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B',    '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D',    '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D','#7B2D8B'),
        ]

    fenetre_ans  = cfg.get('fenetre_ans', 10)
    fenetre_mois = int(fenetre_ans * 12)

    print(f"\n{'='*65}")
    print("PARTIE 11f — CALCUL PM25_pct_supC* ET SEUIL % OPTIMAL P*")
    print(f"{'='*65}")

    df_out  = df_final.copy()
    res_p_star = {}

    for pol, gdict in resultats_c_star.items():
        if pol not in data.columns: continue
        res_p_star[pol] = {}

        # Choisir C* le plus discriminant (AUC max) parmi les groupes
        best_groupe = max(gdict, key=lambda k: gdict[k]['AUC'])
        c_star_best = gdict[best_groupe]['C_star']
        print(f"\n── {pol} : C* retenu = {c_star_best} μg/m³ (meilleur dans {best_groupe})")

        # Calculer pct_supC* pour chaque patient
        col_new  = f'{pol}_pct_sup{c_star_best}'
        pct_vals = {}

        for pseudo in df_final['pseudo_provisoire'].unique():
            row_f = df_final[df_final['pseudo_provisoire']==pseudo]
            if len(row_f) == 0: continue
            date_diag  = pd.Timestamp(row_f['date_diagnostic'].iloc[0])
            date_debut = date_diag - pd.DateOffset(months=fenetre_mois)
            d_pat = data[(data['pseudo_provisoire']==pseudo) &
                          (data['date']>=date_debut) &
                          (data['date']<date_diag)][[pol]].dropna()
            if len(d_pat) == 0:
                pct_vals[pseudo] = float('nan'); continue
            pct = round((d_pat[pol] > c_star_best).sum() / len(d_pat) * 100, 1)
            pct_vals[pseudo] = pct

        df_out[col_new] = df_out['pseudo_provisoire'].map(pct_vals)
        print(f"  ✅ Colonne créée : {col_new}")
        print(f"     Médiane cohorte : {df_out[col_new].median():.1f}%")

        # Trouver P* par Youden sur cette nouvelle variable
        for col_g, g1, g2, titre, couleur in groupes:
            df_sub = df_out[df_out[col_g].isin([g1,g2])].copy()
            df_sub['outcome'] = (df_sub[col_g]==g1).astype(int)
            s = df_sub[[col_new,'outcome']].dropna()
            if len(s) < 30: continue
            s1 = s.loc[s['outcome']==1,col_new].values
            s2 = s.loc[s['outcome']==0,col_new].values
            if len(s1) < 5 or len(s2) < 5: continue

            try:
                fpr,tpr,thresholds = roc_curve(s['outcome'],s[col_new])
                auc = roc_auc_score(s['outcome'],s[col_new])
                youden = tpr-fpr; best_i = np.argmax(youden)
                p_star  = round(float(thresholds[best_i]),1)
                sens    = round(float(tpr[best_i]),3)
                spec    = round(float(1-fpr[best_i]),3)
                _,p_mwu = mannwhitneyu(s1,s2,alternative='two-sided')

                res_p_star[pol][titre] = {
                    'col': col_new,'C_star':c_star_best,'P_star':p_star,
                    'AUC':round(auc,3),'Sens':sens,'Spec':spec,
                    'p_MWU':'<0.001' if p_mwu<0.001 else f'{p_mwu:.3f}',
                    'Sig':'✅' if p_mwu<0.05 else '—'}

                print(f"\n  {titre} : P* = {p_star}% | AUC={auc:.3f} | "
                      f"Sens={sens} Spec={spec} | p={res_p_star[pol][titre]['p_MWU']} "
                      f"{res_p_star[pol][titre]['Sig']}")

                # Visualisation
                fig, axes = plt.subplots(1,2,figsize=(12,4.5))
                fig.suptitle(f'{pol} — {titre}\nC*={c_star_best} μg/m³ | P*={p_star}%',
                             fontsize=12,fontweight='bold')

                # ROC sur pct_supC*
                axes[0].plot(fpr,tpr,color=couleur,lw=2.5,label=f'AUC={auc:.3f}')
                axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
                lbl_ = f'P*={p_star}% | Sens={sens:.2f} Spec={spec:.2f}'
                axes[0].scatter(fpr[best_i],tpr[best_i],s=120,color='red',
                                zorder=5,label=lbl_)
                axes[0].set_xlabel('1 - Spécificité'); axes[0].set_ylabel('Sensibilité')
                axes[0].set_title(f'ROC sur {col_new}',fontweight='bold')
                axes[0].legend(fontsize=8,loc='lower right'); axes[0].grid(True,alpha=0.3)

                # Boxplot comparaison arbitraire vs C*
                # Trouver la meilleure variable arbitraire pour comparaison
                arb_col = None
                for s_arb in cfg.get('seuils_pct',{}).get(pol,[]):
                    c_arb = f'{pol}_pct_sup{s_arb}'
                    if c_arb in df_sub.columns:
                        arb_col = c_arb; break

                data_box = {f'Auto\n{col_new}': df_sub[[col_new,'outcome']].dropna()}
                if arb_col:
                    data_box[f'Arbitraire\n{arb_col.split("_")[-1]}'] = df_sub[[arb_col,'outcome']].dropna()

                pos = 0
                for lbl_b,d_b in data_box.items():
                    g1v = d_b.loc[d_b['outcome']==1,d_b.columns[0]].values
                    g2v = d_b.loc[d_b['outcome']==0,d_b.columns[0]].values
                    axes[1].boxplot([g1v,g2v],positions=[pos,pos+0.35],
                                     widths=0.3,patch_artist=True,
                                     boxprops=dict(facecolor=couleur if pos==0 else 'orange',alpha=0.6))
                    axes[1].text(pos+0.175,-5,lbl_b,ha='center',fontsize=8)
                    pos += 1.2

                axes[1].set_ylabel('% du temps au-dessus du seuil')
                axes[1].set_title('Comparaison arbitraire vs C*',fontweight='bold')
                axes[1].grid(True,axis='y',alpha=0.3)
                plt.tight_layout(); plt.show()

            except Exception as e:
                print(f"  ❌ {e}"); continue

    return df_out, res_p_star


def trouver_seuils_youden_continus(df_final, cfg, groupes=None):
    """
    Partie 11g — Seuils Youden pour les variables continues déjà dans df_final :
    cumul et pente. Crée des variables binaires correspondantes.

    Retourne df_final enrichi + dict des seuils.
    """
    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B',    '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D',    '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D','#7B2D8B'),
        ]

    pol_ret = cfg.get('polluants_retenus', ['PM25','O3'])
    df_out  = df_final.copy()
    seuils_youden_continus = {}

    print(f"\n{'='*65}")
    print("PARTIE 11g — SEUILS YOUDEN CUMUL ET PENTE")
    print(f"{'='*65}")

    for type_var, suffix, label_type in [
        ('cumul',  '_cumul_',         'Cumul'),
        ('pente',  '_mm365_sen_pente', 'Pente Theil-Sen'),
    ]:
        print(f"\n── {label_type} ──")
        for pol in pol_ret:
            # Trouver la colonne
            if type_var == 'cumul':
                cols_v = [c for c in df_final.columns
                          if c.startswith(f'{pol}_cumul_') and 'pct' not in c and 'manq' not in c]
                if not cols_v: continue
                var = cols_v[0]
            else:
                var = f'{pol}_mm365_sen_pente'
                if var not in df_final.columns: continue

            seuils_youden_continus[var] = {}

            for col_g,g1,g2,titre,couleur in groupes:
                df_sub = df_out[df_out[col_g].isin([g1,g2])].copy()
                df_sub['outcome'] = (df_sub[col_g]==g1).astype(int)
                s = df_sub[[var,'outcome']].dropna()
                if len(s)<30 or s[var].nunique()<5: continue
                s1 = s.loc[s['outcome']==1,var].values
                s2 = s.loc[s['outcome']==0,var].values
                if len(s1)<5 or len(s2)<5: continue

                try:
                    fpr,tpr,thresholds = roc_curve(s['outcome'],s[var])
                    auc   = roc_auc_score(s['outcome'],s[var])
                    youd  = tpr-fpr; best_i = np.argmax(youd)
                    seuil = round(float(thresholds[best_i]),1)
                    sens  = round(float(tpr[best_i]),3)
                    spec  = round(float(1-fpr[best_i]),3)
                    _,p   = mannwhitneyu(s1,s2,alternative='two-sided')

                    # Variable binaire
                    col_bin = f'{var}_sup{seuil}'.replace('.','_').replace('-','neg')
                    df_out[col_bin] = (df_out[var] > seuil).astype(int)

                    seuils_youden_continus[var][titre] = {
                        'seuil':seuil,'col_bin':col_bin,'AUC_cont':round(auc,3),
                        'AUC_bin':round(roc_auc_score(
                            s['outcome'],
                            (s[var]>seuil).astype(int)),3),
                        'Sens':sens,'Spec':spec,
                        'p_MWU':'<0.001' if p<0.001 else f'{p:.3f}',
                        'Sig':'✅' if p<0.05 else '—'}

                    print(f"  {var:<35} {titre}")
                    print(f"    Seuil Youden : {seuil} | AUC cont={auc:.3f} | "
                          f"AUC bin={seuils_youden_continus[var][titre]['AUC_bin']:.3f} | "
                          f"p={seuils_youden_continus[var][titre]['p_MWU']} "
                          f"{seuils_youden_continus[var][titre]['Sig']}")
                    print(f"    Colonne créée : {col_bin}")

                    # Visualisation
                    fig,ax = plt.subplots(figsize=(7,4.5))
                    ax.plot(fpr,tpr,color=couleur,lw=2.5,label=f'AUC={auc:.3f}')
                    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
                    lbl_ = f'Seuil={seuil} | Sens={sens:.2f} Spec={spec:.2f}'
                    ax.scatter(fpr[best_i],tpr[best_i],s=120,color='red',zorder=5,label=lbl_)
                    ax.set_xlabel('1 - Spécificité'); ax.set_ylabel('Sensibilité')
                    ax.set_title(f'ROC {label_type} — {pol} | {titre}',fontweight='bold')
                    ax.legend(fontsize=9,loc='lower right'); ax.grid(True,alpha=0.3)
                    plt.tight_layout(); plt.show()

                except Exception as e:
                    print(f"  ❌ {var} {titre}: {e}"); continue

    return df_out, seuils_youden_continus


def synthese_approches(df_final, cfg,
                        res_seuils_bh,
                        resultats_c_star,
                        res_p_star,
                        seuils_youden_continus,
                        groupes=None):
    """
    Partie 11h — Synthèse comparative complète.
    Compare AUC univariées : approche manuelle (BH) vs approche automatique (Youden).
    """
    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B',    '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D',    '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D','#7B2D8B'),
        ]

    pol_ret = cfg.get('polluants_retenus', ['PM25','O3'])

    print(f"\n{'='*65}")
    print("PARTIE 11h — SYNTHÈSE COMPARATIVE : MANUELLE vs AUTOMATIQUE")
    print(f"{'='*65}")

    rows = []

    for col_g,g1,g2,titre,couleur in groupes:
        df_sub = df_final[df_final[col_g].isin([g1,g2])].copy()
        df_sub['outcome'] = (df_sub[col_g]==g1).astype(int)

        for pol in pol_ret:
            # ── Approche manuelle BH ──────────────────────────────────────────
            # Seuil arbitraire le plus significatif
            seuil_bh  = None; auc_bh_pct = None
            df_bh = res_seuils_bh.get('df_res', pd.DataFrame())
            if len(df_bh) > 0:
                df_bh_pol = df_bh[(df_bh['Polluant']==pol)&(df_bh['col_g']==col_g)]
                if len(df_bh_pol) > 0:
                    best_bh   = df_bh_pol.loc[df_bh_pol['p-value'].idxmin()]
                    seuil_bh  = int(best_bh['Seuil'])
                    col_bh    = f'{pol}_pct_sup{seuil_bh}'
                    if col_bh in df_sub.columns:
                        s_bh = df_sub[[col_bh,'outcome']].dropna()
                        if len(s_bh) > 10:
                            try: auc_bh_pct = round(roc_auc_score(s_bh['outcome'],s_bh[col_bh]),3)
                            except Exception: pass

            # Cumul continu
            cols_cum = [c for c in df_final.columns if c.startswith(f'{pol}_cumul_')
                        and 'pct' not in c and 'manq' not in c]
            auc_cumul_cont = None
            if cols_cum:
                col_cum = cols_cum[0]
                s_cum   = df_sub[[col_cum,'outcome']].dropna()
                if len(s_cum) > 10:
                    try: auc_cumul_cont = round(roc_auc_score(s_cum['outcome'],s_cum[col_cum]),3)
                    except Exception: pass

            # Pente continue
            col_pente = f'{pol}_mm365_sen_pente'
            auc_pente_cont = None
            if col_pente in df_sub.columns:
                s_p = df_sub[[col_pente,'outcome']].dropna()
                if len(s_p) > 10:
                    try: auc_pente_cont = round(roc_auc_score(s_p['outcome'],s_p[col_pente]),3)
                    except Exception: pass

            # ── Approche automatique Youden ───────────────────────────────────
            # C* → pct_supC* → P*
            c_star   = resultats_c_star.get(pol,{}).get(titre,{}).get('C_star')
            auc_c_star = resultats_c_star.get(pol,{}).get(titre,{}).get('AUC')
            p_star   = res_p_star.get(pol,{}).get(titre,{}).get('P_star')
            auc_pct_auto = res_p_star.get(pol,{}).get(titre,{}).get('AUC')

            # Cumul binaire Youden
            auc_cumul_bin = None
            if cols_cum:
                d_cum_y = seuils_youden_continus.get(cols_cum[0],{}).get(titre,{})
                auc_cumul_bin = d_cum_y.get('AUC_bin')

            # Pente binaire Youden
            auc_pente_bin = None
            d_pente_y = seuils_youden_continus.get(col_pente,{}).get(titre,{})
            auc_pente_bin = d_pente_y.get('AUC_bin')

            rows.append({
                'Groupe'          : titre,
                'Polluant'        : pol,
                # Concentration
                'C_arbitraire'    : f'>{seuil_bh} μg/m³' if seuil_bh else '—',
                'AUC_pct_arb'     : auc_bh_pct,
                'C_star'          : f'>{c_star} μg/m³' if c_star else '—',
                'AUC_C_star'      : auc_c_star,
                'Δ_AUC_conc'      : round(auc_c_star-auc_bh_pct,3)
                                      if (auc_c_star and auc_bh_pct) else '—',
                # % du temps
                'Seuil_%_arb'     : f'>{seuil_bh}→%' if seuil_bh else '—',
                'AUC_%_arb'       : auc_bh_pct,
                'P_star'          : f'>{p_star}%' if p_star else '—',
                'AUC_%_auto'      : auc_pct_auto,
                'Δ_AUC_pct'       : round(auc_pct_auto-auc_bh_pct,3)
                                      if (auc_pct_auto and auc_bh_pct) else '—',
                # Cumul
                'AUC_cumul_cont'  : auc_cumul_cont,
                'AUC_cumul_bin'   : auc_cumul_bin,
                'Δ_AUC_cumul'     : round(auc_cumul_bin-auc_cumul_cont,3)
                                      if (auc_cumul_bin and auc_cumul_cont) else '—',
                # Pente
                'AUC_pente_cont'  : auc_pente_cont,
                'AUC_pente_bin'   : auc_pente_bin,
                'Δ_AUC_pente'     : round(auc_pente_bin-auc_pente_cont,3)
                                      if (auc_pente_bin and auc_pente_cont) else '—',
            })

    df_comp = pd.DataFrame(rows)
    print("\n── Tableau synthèse ──")
    try:
        from IPython.display import display; display(df_comp)
    except Exception:
        print(df_comp.to_string(index=False))

    # Visualisation — barplot comparatif AUC
    auc_cols = [('AUC_pct_arb','AUC_%_auto','% du temps'),
                ('AUC_cumul_cont','AUC_cumul_bin','Cumul'),
                ('AUC_pente_cont','AUC_pente_bin','Pente')]

    for col_man, col_auto, lbl_type in auc_cols:
        df_v = df_comp[['Groupe','Polluant',col_man,col_auto]].dropna()
        if len(df_v) == 0: continue
        fig, ax = plt.subplots(figsize=(10, max(3, len(df_v)*0.5+1)))
        x = range(len(df_v))
        w = 0.35
        ax.bar([i-w/2 for i in x], pd.to_numeric(df_v[col_man],errors='coerce'),
               width=w, color='#AAAAAA', label='Manuelle (BH)', edgecolor='white')
        ax.bar([i+w/2 for i in x], pd.to_numeric(df_v[col_auto],errors='coerce'),
               width=w, color='#2E86AB', label='Automatique (Youden)', edgecolor='white')
        ax.axhline(y=0.5,color='red',linestyle='--',lw=1,alpha=0.6,label='AUC=0.5')
        ax.set_xticks(range(len(df_v)))
        ax.set_xticklabels([f"{r['Polluant']}\n{r['Groupe']}"
                             for _,r in df_v.iterrows()], fontsize=8, rotation=20)
        ax.set_ylabel('AUC univariée')
        ax.set_title(f'Comparaison AUC — {lbl_type}\nManuelle vs Automatique',
                     fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True,axis='y',alpha=0.3)
        plt.tight_layout(); plt.show()

    return df_comp


def regressions_comparatives(df_final, cfg,
                               res_p_star,
                               seuils_youden_continus,
                               groupes_reg=None):
    """
    Partie 11i — Deux séries de régressions côte à côte :
      Série A : variables manuelles BH (Partie 7)
      Série B : variables automatiques Youden

    Retourne dict avec résultats des deux séries + comparaison.
    """
    if groupes_reg is None:
        groupes_reg = [
            ('AB',   'groupe_AB',    'Groupe A',   'Groupe B',   True,  '#2E86AB'),
            ('CD',   'groupe_CD',    'Groupe C',   'Groupe D',   False, '#52B788'),
            ('ACBD', 'groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', False, '#7B2D8B'),
        ]

    pol_ret = cfg.get('polluants_retenus', ['PM25','O3'])

    # ── Série A : variables CFG actuelles (manuelles BH) ─────────────────────
    print(f"\n{'█'*65}")
    print("SÉRIE A — Variables manuelles BH (identique Partie 7)")
    print(f"{'█'*65}")
    res_A = analyses_principales(df_final, cfg)

    # ── Construire CFG Youden ─────────────────────────────────────────────────
    cfg_y = copy.deepcopy(cfg)

    # Remplacer les variables pct par les variables C* automatiques
    # + ajouter les binaires cumul et pente Youden
    vars_youden_extra = []

    for pol in pol_ret:
        # Variable pct_supC* (depuis res_p_star)
        for titre, res in res_p_star.get(pol,{}).items():
            col_pct_auto = res.get('col')
            if col_pct_auto and col_pct_auto in df_final.columns:
                if col_pct_auto not in vars_youden_extra:
                    vars_youden_extra.append(col_pct_auto)

        # Cumul binaire Youden
        cols_cum = [c for c in df_final.columns
                    if c.startswith(f'{pol}_cumul_') and 'pct' not in c and 'manq' not in c]
        if cols_cum:
            d_cum_y = seuils_youden_continus.get(cols_cum[0],{})
            for titre, res in d_cum_y.items():
                col_bin = res.get('col_bin')
                if col_bin and col_bin in df_final.columns:
                    if col_bin not in vars_youden_extra:
                        vars_youden_extra.append(col_bin)
                    break  # Un seul seuil par variable

        # Pente binaire Youden
        col_pente = f'{pol}_mm365_sen_pente'
        d_pente_y = seuils_youden_continus.get(col_pente,{})
        for titre, res in d_pente_y.items():
            col_bin = res.get('col_bin')
            if col_bin and col_bin in df_final.columns:
                if col_bin not in vars_youden_extra:
                    vars_youden_extra.append(col_bin)
                break

    # Retirer les variables pct arbitraires et remplacer par Youden
    cfg_y['pct_pm25'] = []
    cfg_y['pct_pm10'] = []
    cfg_y['pct_o3']   = []
    cfg_y['cumul']    = []   # cumul continu remplacé par binaire Youden
    # Garder tendance continue
    # Ajouter les variables Youden via icpe (hack : ajouter dans icpe)
    cfg_y['icpe'] = list(cfg.get('icpe',[])) + vars_youden_extra

    print(f"\n{'█'*65}")
    print("SÉRIE B — Variables automatiques Youden")
    print(f"{'█'*65}")
    print(f"Variables Youden ajoutées : {vars_youden_extra}")
    res_B = analyses_principales(df_final, cfg_y)

    # ── Comparaison AUC ───────────────────────────────────────────────────────
    titres = {'AB':'A vs B','CD':'C vs D','ACBD':'A+C vs B+D'}
    rows_comp = []
    for key in ['AB','CD','ACBD']:
        auc_a = res_A.get(key,{}).get('auc',float('nan'))
        auc_b = res_B.get(key,{}).get('auc',float('nan'))
        rows_comp.append({
            'Groupe'      : titres.get(key,key),
            'AUC Série A (BH)'    : round(auc_a,3) if not pd.isna(auc_a) else 'NA',
            'AUC Série B (Youden)': round(auc_b,3) if not pd.isna(auc_b) else 'NA',
            'Δ AUC (B-A)' : round(auc_b-auc_a,3)
                             if (not pd.isna(auc_a) and not pd.isna(auc_b)) else 'NA',
        })
    df_comp_final = pd.DataFrame(rows_comp)
    print("\n── Comparaison AUC finale ──")
    try:
        from IPython.display import display; display(df_comp_final)
    except Exception:
        print(df_comp_final.to_string(index=False))

    # Barplot AUC
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(3)
    aucs_a = [res_A.get(k,{}).get('auc',0) for k in ['AB','CD','ACBD']]
    aucs_b = [res_B.get(k,{}).get('auc',0) for k in ['AB','CD','ACBD']]
    ax.bar([i-0.2 for i in x], aucs_a, width=0.35,
           color='#AAAAAA', label='Série A — BH (manuelle)', edgecolor='white')
    ax.bar([i+0.2 for i in x], aucs_b, width=0.35,
           color='#2E86AB', label='Série B — Youden (auto)', edgecolor='white')
    ax.axhline(y=0.5, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A vs B','C vs D','A+C vs B+D'])
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, max(max(aucs_a), max(aucs_b)) * 1.1 + 0.05)
    ax.set_title('Comparaison AUC — BH vs Youden', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
    for i, (a, b) in enumerate(zip(aucs_a, aucs_b)):
        delta = round(b-a, 3)
        color_d = '#16A34A' if delta > 0 else '#DC2626' if delta < 0 else '#888888'
        ax.text(i, max(a,b)+0.01, f'Δ={delta:+.3f}',
                ha='center', fontsize=9, color=color_d, fontweight='bold')
    plt.tight_layout(); plt.show()

    return {'serie_A': res_A, 'serie_B': res_B,
            'comparaison': df_comp_final, 'cfg_youden': cfg_y}



def chercher_seuils_optimaux(df, cfg, groupes=None, comparer_partie4=True):
    """
    Partie 11c — Pour chaque variable continue déjà dans df_final
    (cumul, pente Sen, % du temps arbitraires), identifie le seuil optimal
    par l'indice de Youden (max sensibilité + spécificité - 1).

    Si comparer_partie4=True, compare avec les seuils BH retenus (Partie 4).

    Retourne dict {groupe_label: DataFrame seuils optimaux}.
    """
    if groupes is None:
        groupes = [
            ('groupe_AB',    'Groupe A',   'Groupe B',   'A vs B — Mutations NF',          '#2E86AB'),
            ('groupe_CD',    'Groupe C',   'Groupe D',   'C vs D — Non-fumeurs vs Fumeurs', '#52B788'),
            ('groupe_AC_BD', 'Groupe A+C', 'Groupe B+D', 'A+C vs B+D',                     '#7B2D8B'),
        ]

    pol_ret = cfg.get('polluants_retenus', ['PM25','PM10','NO2','O3'])

    def _vars_a_tester(df, cfg):
        """Collecte toutes les variables continues à tester."""
        vars_out = {'cumul': [], 'pente': [], 'pct': []}
        for col in df.columns:
            for pol in pol_ret:
                if (col.startswith(f'{pol}_cumul_') and 'pct' not in col
                        and 'manq' not in col):
                    vars_out['cumul'].append(col)
                elif col == f'{pol}_mm365_sen_pente':
                    vars_out['pente'].append(col)
                elif col.startswith(f'{pol}_pct_sup'):
                    vars_out['pct'].append(col)
        # Dédoublonner
        for k in vars_out:
            vars_out[k] = list(dict.fromkeys(vars_out[k]))
        return vars_out

    # Seuils BH pour comparaison
    seuils_bh_map = {}
    if comparer_partie4:
        for pol in pol_ret:
            for key in [f'pct_{pol.lower()}', f'pct_{pol}']:
                for s in cfg.get(key, []):
                    seuils_bh_map[f'{pol}_pct_sup{s}'] = s

    print(f"\n{'='*65}")
    print("PARTIE 11c — SEUILS OPTIMAUX (YOUDEN) SUR VARIABLES df_final")
    print(f"{'='*65}")
    print(f"Polluants retenus : {pol_ret}")

    resultats_dict = {}

    for col_g, g1, g2, titre, couleur in groupes:
        print(f"\n{'─'*55}\n  {titre}\n{'─'*55}")
        df_sub = df[df[col_g].isin([g1, g2])].copy()
        df_sub['outcome'] = (df_sub[col_g] == g1).astype(int)

        vars_par_type = _vars_a_tester(df, cfg)
        rows = []

        for type_var, vars_list in vars_par_type.items():
            type_label = {
                'cumul': 'Cumul exposition',
                'pente': 'Pente Theil-Sen',
                'pct'  : '% du temps (seuils arbitraires)'
            }[type_var]

            for var in vars_list:
                if var not in df_sub.columns: continue
                s = df_sub[[var, 'outcome']].dropna()
                if len(s) < 30 or s[var].nunique() < 3: continue
                s1 = s.loc[s['outcome']==1, var].values
                s2 = s.loc[s['outcome']==0, var].values
                if len(s1) < 5 or len(s2) < 5: continue

                try:
                    fpr, tpr, thresholds = roc_curve(s['outcome'], s[var])
                    auc  = roc_auc_score(s['outcome'], s[var])
                    youd = tpr - fpr
                    bi   = np.argmax(youd)
                    sopt = round(float(thresholds[bi]), 1)
                    sens = round(float(tpr[bi]), 3)
                    spec = round(float(1-fpr[bi]), 3)
                    _, p = mannwhitneyu(s1, s2, alternative='two-sided')

                    row = {
                        'Variable'   : var,
                        'Type'       : type_label,
                        'AUC'        : round(auc, 3),
                        'Seuil_opt'  : sopt,
                        'Sensibilité': sens,
                        'Spécificité': spec,
                        'Youden'     : round(float(youd[bi]), 3),
                        'p_MWU'      : '<0.001' if p<0.001 else f'{p:.3f}',
                        'p_num'      : float(p),
                        'Sig'        : '✅' if p<0.05 else '—',
                    }

                    # Comparaison avec seuil BH
                    if comparer_partie4 and var in seuils_bh_map:
                        seuil_bh = seuils_bh_map[var]
                        row['Seuil_BH']  = seuil_bh
                        row['Différent'] = '⚠️' if abs(sopt - seuil_bh) > 1 else '='
                    else:
                        row['Seuil_BH']  = '—'
                        row['Différent'] = '—'

                    rows.append(row)

                except Exception:
                    continue

        if not rows:
            print("  ⚠️  Aucune variable testable.")
            resultats_dict[titre] = pd.DataFrame()
            continue

        df_seuils = pd.DataFrame(rows).sort_values('AUC', ascending=False)
        resultats_dict[titre] = df_seuils

        try:
            from IPython.display import display
            display(df_seuils.drop(columns=['p_num']))
        except Exception:
            print(df_seuils.drop(columns=['p_num']).to_string(index=False))

        # ── Visualisation ROC par variable ────────────────────────────────────
        vars_plot = (df_seuils[df_seuils['Sig']=='✅']['Variable'].tolist()
                     or df_seuils.head(4)['Variable'].tolist())

        if vars_plot:
            ncols = min(3, len(vars_plot))
            nrows = int(np.ceil(len(vars_plot)/ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(6*ncols, 4.5*nrows))
            fig.suptitle(f'Seuils optimaux Youden — {titre}', fontsize=12, fontweight='bold')
            axes_flat = np.array(axes).flatten() if nrows*ncols > 1 else [axes]

            for ax, var in zip(axes_flat, vars_plot):
                s    = df_sub[[var,'outcome']].dropna()
                fpr, tpr, thresholds = roc_curve(s['outcome'], s[var])
                auc  = roc_auc_score(s['outcome'], s[var])
                youd = tpr - fpr; bi = np.argmax(youd)
                sopt = thresholds[bi]

                ax.plot(fpr, tpr, color=couleur, lw=2.5, label=f'AUC={auc:.3f}')
                ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
                lbl_ = f'Seuil={sopt:.1f} Sens={tpr[bi]:.2f} Spec={1-fpr[bi]:.2f}'
                ax.scatter(fpr[bi], tpr[bi], s=120, color='red', zorder=5, label=lbl_)

                # Point seuil BH si disponible
                if var in seuils_bh_map:
                    s_bh = seuils_bh_map[var]
                    try:
                        s_bin = (s[var] > s_bh).astype(int)
                        fpr_b = 1 - (s_bin[s['outcome']==0]==0).mean()
                        tpr_b = (s_bin[s['outcome']==1]==1).mean()
                        ax.scatter(fpr_b, tpr_b, s=80, color='orange',
                                   marker='D', zorder=4,
                                   label=f'Seuil BH={s_bh}')
                    except Exception: pass

                ax.set_xlabel('1 - Spécificité'); ax.set_ylabel('Sensibilité')
                ax.set_title(var, fontsize=9, fontweight='bold')
                ax.legend(fontsize=7, loc='lower right')
                ax.grid(True, alpha=0.3)

            for ax in axes_flat[len(vars_plot):]:
                ax.set_visible(False)
            plt.tight_layout(); plt.show()

        # ── Comparaison seuil optimal vs BH ──────────────────────────────────
        df_comp = df_seuils[df_seuils['Seuil_BH'] != '—'].copy()
        if len(df_comp) > 0:
            fig, ax = plt.subplots(figsize=(max(6, len(df_comp)*1.2), 4))
            x = range(len(df_comp))
            opts = df_comp['Seuil_opt'].values
            bhs  = pd.to_numeric(df_comp['Seuil_BH'], errors='coerce').values
            ax.scatter(x, opts, color=couleur, s=100, zorder=5,
                       label='Seuil optimal Youden', marker='o')
            ax.scatter(x, bhs, color='orange', s=80, zorder=5,
                       label='Seuil BH Partie 4', marker='D')
            for i,(o,b) in enumerate(zip(opts, bhs)):
                if not np.isnan(b):
                    ax.plot([i,i],[o,b], color='gray', lw=1.5, linestyle='--', alpha=0.5)
                    if abs(o-b) > 1:
                        ax.annotate('⚠️', xy=(i, max(o,b)+0.5), ha='center', fontsize=10)
            ax.set_xticks(range(len(df_comp)))
            ax.set_xticklabels(df_comp['Variable'].tolist(), rotation=25, ha='right', fontsize=8)
            ax.set_ylabel('Valeur du seuil')
            ax.set_title(f'Seuil Youden vs Seuil BH — {titre}', fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout(); plt.show()

    return resultats_dict


def synthese_comparaison(resultats_dict):
    """Tableau comparatif de plusieurs résultats."""
    rows=[]
    for label,res in resultats_dict.items():
        if res is None or not isinstance(res,dict) or res.get('tableau') is None: continue
        tab=res['tableau']; sig_vars=tab[tab['Sig']=='✅']['Variable'].tolist()
        rows.append({'Analyse':label,'AUC':round(res.get('auc',float('nan')),3),
                     'N sig.':len(sig_vars),'Var. sig.':' | '.join(sig_vars) if sig_vars else '—'})
    df_comp=pd.DataFrame(rows)
    print("\n── Tableau comparatif ──")
    try:
        from IPython.display import display; display(df_comp)
    except: print(df_comp.to_string(index=False))
    return df_comp
