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

def _calculer_exposition_cumulee(serie, nom):
    s = serie[~np.isnan(serie)]
    n_total,n_valides = len(serie),len(s)
    pct = (n_total-n_valides)/n_total*100 if n_total>0 else np.nan
    if n_valides == 0:
        return {f'{nom}_cumul_120m':np.nan,f'{nom}_cumul_pct_manq':np.nan}
    return {f'{nom}_cumul_120m':np.nansum(serie),f'{nom}_cumul_pct_manq':pct}

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
    cfg doit contenir 'seuils_pct' : {'PM25':[...],'PM10':[...],'O3':[...]}
    Retourne df_air.
    """
    seuils_pct = cfg.get('seuils_pct', {'PM25':[5,10,15,25,35],'PM10':[35,45,50,60,80,90],'O3':[100,120,180]})

    patients_communs = set(data['pseudo_provisoire'].unique()) & set(df_clinique['pseudo_provisoire'].unique())
    print(f"Patients communs : {len(patients_communs)}")

    df_poll = data[data['pseudo_provisoire'].isin(patients_communs)].reset_index(drop=True)
    df_clin = df_clinique[df_clinique['pseudo_provisoire'].isin(patients_communs)].reset_index(drop=True)
    groupes_poll = df_poll.groupby('pseudo_provisoire')

    resultats, exclus = [], []
    for pseudo in tqdm(df_clin['pseudo_provisoire'].unique(), desc='Patients'):
        date_diag  = pd.Timestamp(df_clin.loc[df_clin['pseudo_provisoire']==pseudo,'date_diagnostic'].iloc[0])
        date_debut = date_diag - relativedelta(months=FENETRE_MAX_MOIS)
        if pseudo not in groupes_poll.groups:
            exclus.append(pseudo); continue
        df_pat = df_poll.loc[groupes_poll.groups[pseudo]]
        df_pat = df_pat[(df_pat['date']>=date_debut)&(df_pat['date']<date_diag)].copy()
        if len(df_pat) == 0:
            exclus.append(pseudo); continue
        m = {'pseudo_provisoire':pseudo,'date_diagnostic':date_diag}
        for pol in POLLUANTS_DEFAUT:
            if pol not in df_pat.columns: continue
            serie = df_pat[pol].values
            m.update(_calculer_tendance_centrale(serie,pol))
            m.update(_calculer_variabilite(serie,pol))
            m.update(_calculer_exposition_cumulee(serie,pol))
            m.update(_calculer_tendance_mm_mk(df_pat,pol))
            if pol in seuils_pct:
                m.update(_calculer_pct_seuils(serie,pol,seuils_pct[pol]))
        resultats.append(m)

    df_air = _calculer_IPG(pd.DataFrame(resultats))
    if exclus: print(f"⚠️  {len(exclus)} patients exclus")
    print(f"✅ df_air : {len(df_air)} patients × {len(df_air.columns)} variables")
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


def construire_df_final(data, df_clinique, df_air):
    """Fusionne df_air avec df_clinique → df_final avec groupes."""
    df_final = df_clinique.merge(df_air, on=['pseudo_provisoire','date_diagnostic'], how='inner')

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
    """Construit la liste des variables sans doublons depuis CFG."""
    seen, vars_mod = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); vars_mod.append(v)

    for v in cfg.get('cumul',[]): add(v)
    for v in cfg.get('tendance',[]): add(v)
    for s in cfg.get('pct_pm25',[]): add(f'PM25_pct_sup{s}')
    for s in cfg.get('pct_pm10',[]): add(f'PM10_pct_sup{s}')
    for s in cfg.get('pct_o3',[]): add(f'O3_pct_sup{s}')
    if cfg.get('inclure_age',True): add('age_diagnostic')
    if inclure_paquet and cfg.get('inclure_paquet',True): add('paquet_annee')
    if cfg.get('inclure_sexe',True): add('sexe_bin')
    if cfg.get('inclure_edi',True): add('quintileEDI2021')
    if cfg.get('inclure_trafic',True): add('indice_trafic')
    for v in cfg.get('icpe',[]): add(v)
    return vars_mod


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATION VISUELLE
# ═══════════════════════════════════════════════════════════════════════════════

def explorer_distribution_polluants(df):
    """Boxplots des cumuls par groupe pour les 4 polluants."""
    pols = [(f'{p}_cumul_120m', f'{p} (μg/m³·j)') for p in ['PM25','PM10','NO2','O3']
            if f'{p}_cumul_120m' in df.columns]
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
    seuils_test = cfg.get('seuils_pct', {'PM25':[5,10,15,25,35],'PM10':[35,45,50,60,80,90],'O3':[100,120,180]})
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
    return df_res


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CORRÉLATION & FAMD
# ═══════════════════════════════════════════════════════════════════════════════

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

    X=sm.add_constant(d_std[vars_ok]); y=d_std['outcome']
    try:
        modele=sm.Logit(y,X).fit(disp=False)
    except Exception as e:
        # Tentative avec méthode alternative (bfgs plus robuste)
        try:
            modele=sm.Logit(y,X).fit(disp=False, method='bfgs', maxiter=200)
        except Exception as e2:
            print(f"⚠️  {titre} : échec de la régression ({e2})")
            print("   → Cause probable : quasi-séparation parfaite ou trop peu de cas positifs.")
            return None,None,None
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
        X=sm.add_constant(d_std[vars_ok]); y=d_std['outcome']
        modele=sm.Logit(y,X).fit(disp=False); tab=_tableau_OR(modele,vars_ok)
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
