"""
═══════════════════════════════════════════════════════════════════════════════
LUNG-CANC'AIR — Analyse spatiale de Moran
Tester l'association spatiale entre mutations NF et industries Seveso SH

Analyses réalisées :
  1. Statistiques descriptives spatiales
  2. Moran's I global univarié (clustering mutations NF)
  3. LISA local (identifier les clusters)
  4. Moran's I bivarié (mutations NF × proximité Seveso SH)
  5. Test de permutation (significativité)
  6. Cartes de résultats

Auteur : LUNG-CANC'AIR
═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import cKDTree
from scipy.stats import norm, mannwhitneyu
from pyproj import Transformer

import libpysal
from libpysal.weights import DistanceBand, w_subset
from esda.moran import Moran, Moran_Local, Moran_BV, Moran_Local_BV

plt.style.use('seaborn-v0_8-whitegrid')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Modifier ici
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {

    # ── Chemins des fichiers ──────────────────────────────────────────────────
    # Fichier patients géocodés (doit contenir x, y en WGS84 et les colonnes cliniques)
    'patients_csv'  : r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Data\patients_geocoded_clean_idf_2018_2023.csv",

    # Shapefile ICPE (avec colonne 'seveso' : NS / SB / SH)
    'icpe_shp'      : r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Data\industries_a_risques\icpe.geojson\icpe_idf.shp",

    # Dossier de sortie des figures
    'output_dir'    : r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Notebooks\output\resultats_moran",

    # ── Paramètres spatiaux ───────────────────────────────────────────────────
    # Rayon de voisinage en mètres (pour la matrice de poids)
    'rayon_m'       : 3000,

    # Nombre de permutations pour les tests de significativité
    'n_permutations': 999,

    # Seuil de significativité
    'alpha'         : 0.05,

    # ── Colonnes dans le fichier patients ─────────────────────────────────────
    'col_x'         : 'x',            # longitude WGS84
    'col_y'         : 'y',            # latitude WGS84
    'col_id'        : 'pseudo_provisoire',

    # Mutations NF (non-fumeurs) à considérer
    'mutations_nf'  : ['EGFR', 'ALK', 'ROS1', 'RET', 'NTRK', 'ERBB2', 'MET'],

    # Colonne tabac
    'col_tabac'     : 'paquet_annee',
}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT ET PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def charger_patients(config):
    """
    Charge le fichier patients, crée les variables de groupe,
    projette en Lambert 93 (EPSG:2154) pour les calculs de distance.
    """
    print("─" * 65)
    print("ÉTAPE 1 — Chargement des données patients")
    print("─" * 65)

    df = pd.read_csv(config['patients_csv'], dtype=str, keep_default_na=False)
    df[config['col_x']] = pd.to_numeric(df[config['col_x']], errors='coerce')
    df[config['col_y']] = pd.to_numeric(df[config['col_y']], errors='coerce')
    df[config['col_tabac']] = pd.to_numeric(df[config['col_tabac']], errors='coerce')
    df = df.dropna(subset=[config['col_x'], config['col_y']]).reset_index(drop=True)

    print(f"  Patients chargés : {len(df)}")

    # Détecter mutations NF
    def porte_mutation(df, mut):
        col = f'mutation_{mut}'
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip().str.lower().isin(
            ['1','true','oui','yes','positive','positif','pos','muté','mute','detected','present'])

    mask_nf = pd.Series(False, index=df.index)
    for mut in config['mutations_nf']:
        mask_nf |= porte_mutation(df, mut)

    df['mutation_NF']    = mask_nf.astype(int)
    df['non_fumeur']     = (df[config['col_tabac']] == 0).astype(int)
    df['groupe_A']       = mask_nf.astype(int)                          # mutations NF
    df['groupe_C']       = (df[config['col_tabac']] == 0).astype(int)   # non-fumeurs
    df['groupe_AC']      = (mask_nf | (df[config['col_tabac']] == 0)).astype(int)

    print(f"  Groupe A (mutations NF)  : {df['groupe_A'].sum()} ({df['groupe_A'].mean()*100:.1f}%)")
    print(f"  Groupe C (non-fumeurs)   : {df['groupe_C'].sum()} ({df['groupe_C'].mean()*100:.1f}%)")
    print(f"  Groupe A+C               : {df['groupe_AC'].sum()} ({df['groupe_AC'].mean()*100:.1f}%)")

    # Projeter en Lambert 93 pour calculs de distance
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_l93, y_l93 = transformer.transform(df[config['col_x']].values,
                                           df[config['col_y']].values)
    df['x_l93'] = x_l93
    df['y_l93'] = y_l93

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[config['col_x']], df[config['col_y']]),
        crs='EPSG:4326'
    )

    print(f"  Coordonnées projetées en Lambert 93 ✅")
    return df, gdf


def charger_icpe(config):
    """Charge le shapefile ICPE et filtre les Seveso SH."""
    print("\n  Chargement ICPE...")
    icpe = gpd.read_file(config['icpe_shp'])
    if icpe.crs.to_epsg() != 2154:
        icpe = icpe.to_crs('EPSG:2154')

    icpe_sh = icpe[icpe['seveso'] == 'SH'].reset_index(drop=True)
    icpe_sb = icpe[icpe['seveso'] == 'SB'].reset_index(drop=True)
    icpe_ns = icpe[icpe['seveso'] == 'NS'].reset_index(drop=True)

    print(f"  ICPE total : {len(icpe)} | SH : {len(icpe_sh)} | SB : {len(icpe_sb)} | NS : {len(icpe_ns)}")
    return icpe, icpe_sh, icpe_sb, icpe_ns


def calculer_expositions_icpe(df, icpe_sh, icpe_sb, rayon_m):
    """
    Calcule pour chaque patient :
      - distance au Seveso SH le plus proche
      - nb de Seveso SH dans le rayon
      - score d'exposition = 1/distance (inversée normalisée)
    """
    print(f"\n  Calcul expositions ICPE (rayon {rayon_m}m)...")

    coords_patients = np.column_stack([df['x_l93'].values, df['y_l93'].values])
    coords_sh       = np.column_stack([icpe_sh.geometry.x.values, icpe_sh.geometry.y.values])
    coords_sb       = np.column_stack([icpe_sb.geometry.x.values, icpe_sb.geometry.y.values])

    # Distance au SH le plus proche
    tree_sh = cKDTree(coords_sh)
    dist_sh, _ = tree_sh.query(coords_patients, k=1)
    df['dist_SH_m'] = dist_sh.round(1)

    # Nombre de SH dans le rayon
    idx_rayon = tree_sh.query_ball_point(coords_patients, r=rayon_m)
    variable_name = f'nb_SH_{rayon_m//1000}km'
    df[variable_name] = [len(idx) for idx in idx_rayon]

    # Distance au SB le plus proche
    if len(coords_sb) > 0:
        tree_sb = cKDTree(coords_sb)
        dist_sb, _ = tree_sb.query(coords_patients, k=1)
        df['dist_SB_m'] = dist_sb.round(1)
    else:
        df['dist_SB_m'] = float('nan')

    # Score d'exposition SH = 1/distance normalisée (0-1)
    # Plus le score est élevé, plus le patient est proche d'un SH
    df['score_expo_SH'] = 1 / (df['dist_SH_m'] + 1)  # +1 pour éviter division par 0
    df['score_expo_SH'] = (df['score_expo_SH'] - df['score_expo_SH'].min()) / \
                           (df['score_expo_SH'].max() - df['score_expo_SH'].min())

    print(f"  dist_SH_m    : médiane={df['dist_SH_m'].median():.0f}m | "
          f"min={df['dist_SH_m'].min():.0f}m | max={df['dist_SH_m'].max():.0f}m")
    print(f"  {variable_name} : médiane={df[variable_name].median():.0f} | "
          f"max={df[variable_name].max():.0f}")
    print(f"  score_expo_SH: médiane={df['score_expo_SH'].median():.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — MATRICE DE POIDS SPATIAUX
# ─────────────────────────────────────────────────────────────────────────────

def construire_matrice_poids(df, rayon_m, n_permutations=999):
    """
    Construit la matrice de poids spatiaux W par rayon fixe.
    Vérifie la connectivité et affiche les statistiques.
    """
    print("\n" + "─" * 65)
    print(f"ÉTAPE 2 — Matrice de poids spatiaux (rayon {rayon_m}m)")
    print("─" * 65)

    coords = list(zip(df['x_l93'].values, df['y_l93'].values))

    # Construction W avec rayon fixe
    W = DistanceBand(coords, threshold=rayon_m, binary=True, silence_warnings=True)
    W.transform = 'r'  # Row-standardisation

    # Statistiques de connectivité
    n_voisins = [len(W.neighbors[i]) for i in range(len(df))]
    n_isoles  = sum(1 for n in n_voisins if n == 0)

    print(f"  N patients            : {len(df)}")
    print(f"  Rayon                 : {rayon_m} m")
    print(f"  Voisins par patient   : médiane={np.median(n_voisins):.0f} | "
          f"min={min(n_voisins)} | max={max(n_voisins)}")
    print(f"  Patients isolés (0 voisin) : {n_isoles} ({n_isoles/len(df)*100:.1f}%)")

    if n_isoles > len(df) * 0.3:
        print(f"\n  ⚠️  Plus de 30% de patients isolés !")
        print(f"     Envisager un rayon plus grand ou k plus proches voisins.")

    # Histogramme distribution des voisins
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(n_voisins, bins=30, color='#2E86AB', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(n_voisins), color='red', linestyle='--', lw=2,
               label=f'Médiane = {np.median(n_voisins):.0f}')
    ax.set_xlabel(f'Nombre de voisins (rayon {rayon_m//1000}km)', fontsize=11)
    ax.set_ylabel('Nombre de patients', fontsize=11)
    ax.set_title('Distribution du nombre de voisins par patient', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/01_distribution_voisins.png", dpi=150, bbox_inches='tight')
    plt.show()

    return W, n_voisins


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — MORAN'S I GLOBAL UNIVARIÉ
# ─────────────────────────────────────────────────────────────────────────────

def moran_global_univarie(df, W, variable, label, couleur='#2E86AB', n_perm=999):
    """
    Calcule Moran's I global pour une variable.
    Retourne (I, p-value, z-score).
    """
    y = df[variable].values.astype(float)

    # Retirer les patients isolés pour le calcul
    idx_ok = [i for i in range(len(df)) if len(W.neighbors[i]) > 0]
    if len(idx_ok) < len(df):
        print(f"  ⚠️  {len(df)-len(idx_ok)} patients isolés exclus du calcul")

    moran = Moran(y, W, permutations=n_perm)

    print(f"\n  {label}")
    print(f"  Moran's I = {moran.I:.4f} | E[I] = {moran.EI:.4f}")
    print(f"  z-score   = {moran.z_norm:.4f}")
    print(f"  p-value   = {moran.p_norm:.4f} {'✅ sig.' if moran.p_norm < 0.05 else '— non sig.'}")
    print(f"  p-valeur (permutation) = {moran.p_sim:.4f} "
          f"{'✅ sig.' if moran.p_sim < 0.05 else '— non sig.'}")

    # Moran scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    y_std  = (y - y.mean()) / y.std()
    Wy_std = W.sparse.dot(y_std)

    ax.scatter(y_std, Wy_std, color=couleur, alpha=0.4, s=15, edgecolors='none')
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.axvline(0, color='gray', lw=0.8, linestyle='--')

    # Droite de régression (pente = Moran's I)
    z = np.polyfit(y_std, Wy_std, 1)
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color='red', lw=2,
            label=f"Pente = I = {moran.I:.4f}")

    # Quadrants
    ax.text(y_std.max()*0.7, Wy_std.max()*0.8, 'HH', fontsize=14, color='#C0392B',
            fontweight='bold', alpha=0.6)
    ax.text(y_std.min()*0.7, Wy_std.max()*0.8, 'LH', fontsize=14, color='#2980B9',
            fontweight='bold', alpha=0.6)
    ax.text(y_std.max()*0.7, Wy_std.min()*0.8, 'HL', fontsize=14, color='#E67E22',
            fontweight='bold', alpha=0.6)
    ax.text(y_std.min()*0.7, Wy_std.min()*0.8, 'LL', fontsize=14, color='#27AE60',
            fontweight='bold', alpha=0.6)

    ax.set_xlabel(f'{label} (standardisé)', fontsize=11)
    ax.set_ylabel(f'Lag spatial de {label}', fontsize=11)
    ax.set_title(f"Diagramme de Moran — {label}\n"
                 f"I = {moran.I:.4f} | p = {moran.p_sim:.4f} "
                 f"({'sig.' if moran.p_sim < 0.05 else 'non sig.'})",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/02_moran_scatter_{variable}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    # Distribution des permutations
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(moran.sim, bins=50, color='#95A5A6', edgecolor='white', alpha=0.8,
            label=f'Distribution sous H₀\n({n_perm} permutations)')
    ax.axvline(moran.I, color='red', lw=2.5, label=f'I observé = {moran.I:.4f}')
    ax.axvline(moran.EI, color='black', lw=1.5, linestyle='--',
               label=f'E[I] théorique = {moran.EI:.4f}')
    ax.set_xlabel("Moran's I", fontsize=11)
    ax.set_ylabel('Fréquence', fontsize=11)
    ax.set_title(f"Test de permutation — {label}\np = {moran.p_sim:.4f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/03_permutation_{variable}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return {'I': moran.I, 'EI': moran.EI, 'z': moran.z_norm,
            'p_norm': moran.p_norm, 'p_sim': moran.p_sim,
            'variable': variable, 'label': label}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — LISA (Indicateurs locaux)
# ─────────────────────────────────────────────────────────────────────────────

def lisa_local(df, W, variable, label, icpe_sh, n_perm=999, alpha=0.05):
    """
    Calcule les indicateurs LISA locaux et cartographie les clusters.
    Quadrants :
      1 = HH (haute valeur entourée de hautes valeurs) → cluster positif
      2 = LH (basse valeur entourée de hautes valeurs) → outlier spatial
      3 = LL (basse valeur entourée de basses valeurs) → cluster négatif
      4 = HL (haute valeur entourée de basses valeurs) → outlier spatial
    """
    print(f"\n  LISA — {label}")

    y = df[variable].values.astype(float)
    lisa = Moran_Local(y, W, permutations=n_perm, seed=42)

    # Clusters significatifs
    sig       = lisa.p_sim < alpha
    quadrant  = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL

    df = df.copy()
    df['lisa_q']   = quadrant
    df['lisa_sig'] = sig.astype(int)
    df['lisa_type'] = 'Non sig.'
    df.loc[sig & (quadrant == 1), 'lisa_type'] = 'HH'
    df.loc[sig & (quadrant == 2), 'lisa_type'] = 'LH'
    df.loc[sig & (quadrant == 3), 'lisa_type'] = 'LL'
    df.loc[sig & (quadrant == 4), 'lisa_type'] = 'HL'

    n_hh = (sig & (quadrant == 1)).sum()
    n_ll = (sig & (quadrant == 3)).sum()
    n_hl = (sig & (quadrant == 4)).sum()
    n_lh = (sig & (quadrant == 2)).sum()

    print(f"    HH (cluster positif)  : {n_hh} patients")
    print(f"    LL (cluster négatif)  : {n_ll} patients")
    print(f"    HL (outlier spatial)  : {n_hl} patients")
    print(f"    LH (outlier spatial)  : {n_lh} patients")
    print(f"    Non significatif      : {(~sig).sum()} patients")

    # Palette LISA standard
    palette = {
        'HH'     : '#C0392B',
        'LH'     : '#2980B9',
        'LL'     : '#27AE60',
        'HL'     : '#E67E22',
        'Non sig.': '#D5D8DC',
    }

    # Carte LISA
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for type_lisa, color in palette.items():
        sub = df[df['lisa_type'] == type_lisa]
        if len(sub) == 0:
            continue
        ax.scatter(sub['x_l93'].values / 1000,
                   sub['y_l93'].values / 1000,
                   c=color, s=15 if type_lisa == 'Non sig.' else 40,
                   alpha=0.5 if type_lisa == 'Non sig.' else 0.9,
                   label=f'{type_lisa} (n={len(sub)})',
                   zorder=2 if type_lisa == 'Non sig.' else 3)

    # Ajouter les Seveso SH
    ax.scatter(icpe_sh.geometry.x.values / 1000,
               icpe_sh.geometry.y.values / 1000,
               c='black', marker='*', s=120, zorder=5,
               label=f'Seveso SH (n={len(icpe_sh)})')

    ax.set_xlabel('Lambert 93 X (km)', fontsize=11)
    ax.set_ylabel('Lambert 93 Y (km)', fontsize=11)
    ax.set_title(f'Carte LISA — {label}\n'
                 f'HH={n_hh} | LL={n_ll} | HL={n_hl} | LH={n_lh} '
                 f'(p < {alpha}, {n_perm} permutations)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right',
              title='Clusters LISA', title_fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/04_lisa_{variable}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return df, lisa, {'HH': n_hh, 'LL': n_ll, 'HL': n_hl, 'LH': n_lh}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — MORAN's I BIVARIÉ
# ─────────────────────────────────────────────────────────────────────────────

def moran_bivarie(df, W, var_x, var_y, label_x, label_y, n_perm=999):
    """
    Moran's I bivarié : mesure si les zones avec beaucoup de var_x
    sont proches des zones avec beaucoup de var_y.

    I_biv = Σ_i(x_i * Σ_j(w_ij * y_j)) / n

    Interprétation :
      I_biv > 0 : co-clustering positif (les deux variables se retrouvent ensemble)
      I_biv < 0 : co-clustering négatif (les deux s'évitent)
      I_biv ≈ 0 : pas d'association spatiale
    """
    print(f"\n  Moran bivarié : {label_x} × {label_y}")

    x = df[var_x].values.astype(float)
    y = df[var_y].values.astype(float)

    moran_bv = Moran_BV(x, y, W, permutations=n_perm)

    print(f"  I bivarié = {moran_bv.I:.4f}")
    print(f"  z-score   = {moran_bv.z_sim:.4f}")
    print(f"  p-value (permutation) = {moran_bv.p_sim:.4f} "
          f"{'✅ sig.' if moran_bv.p_sim < 0.05 else '— non sig.'}")

    # Scatter plot bivarié
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Moran bivarié : {label_x} × {label_y}',
                 fontsize=13, fontweight='bold')

    # Moran scatter
    ax = axes[0]
    x_std  = (x - x.mean()) / (x.std() + 1e-10)
    Wy_std = W.sparse.dot((y - y.mean()) / (y.std() + 1e-10))

    ax.scatter(x_std, Wy_std, color='#7B2D8B', alpha=0.4, s=15, edgecolors='none')
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.axvline(0, color='gray', lw=0.8, linestyle='--')
    z_fit = np.polyfit(x_std, Wy_std, 1)
    x_line = np.linspace(x_std.min(), x_std.max(), 100)
    ax.plot(x_line, np.polyval(z_fit, x_line), color='red', lw=2,
            label=f'I_biv = {moran_bv.I:.4f}')
    ax.set_xlabel(f'{label_x} (standardisé)', fontsize=10)
    ax.set_ylabel(f'Lag spatial de {label_y}', fontsize=10)
    ax.set_title('Diagramme de Moran bivarié', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Distribution permutations
    ax2 = axes[1]
    ax2.hist(moran_bv.sim, bins=50, color='#95A5A6', edgecolor='white',
             alpha=0.8, label=f'H₀ ({n_perm} permutations)')
    ax2.axvline(moran_bv.I, color='red', lw=2.5,
                label=f'I observé = {moran_bv.I:.4f}')
    ax2.axvline(0, color='black', lw=1.5, linestyle='--', label='E[I] = 0')
    ax2.set_xlabel("Moran's I bivarié", fontsize=10)
    ax2.set_ylabel('Fréquence', fontsize=10)
    ax2.set_title(f'Test de permutation\np = {moran_bv.p_sim:.4f}',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/05_moran_bivarie_{var_x}_{var_y}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return {'I': moran_bv.I, 'z': moran_bv.z_sim,
            'p_sim': moran_bv.p_sim, 'var_x': var_x, 'var_y': var_y,
            'label_x': label_x, 'label_y': label_y}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — LISA BIVARIÉ
# ─────────────────────────────────────────────────────────────────────────────

def lisa_bivarie(df, W, var_x, var_y, label_x, label_y,
                 icpe_sh, n_perm=999, alpha=0.05):
    """
    LISA bivarié : identifie localement les zones de co-clustering
    entre var_x et var_y.

    Quadrant 1 (HH) : patient avec valeur élevée de var_x
                       entouré de patients avec valeur élevée de var_y
    → Zones où beaucoup de mutés NF coïncident avec forte proximité SH
    """
    print(f"\n  LISA bivarié : {label_x} × {label_y}")

    x = df[var_x].values.astype(float)
    y = df[var_y].values.astype(float)

    lisa_bv = Moran_Local_BV(x, y, W, permutations=n_perm, seed=42)

    sig      = lisa_bv.p_sim < alpha
    quadrant = lisa_bv.q

    df = df.copy()
    df['lisa_bv_q']    = quadrant
    df['lisa_bv_sig']  = sig.astype(int)
    df['lisa_bv_type'] = 'Non sig.'
    df.loc[sig & (quadrant == 1), 'lisa_bv_type'] = 'HH'
    df.loc[sig & (quadrant == 2), 'lisa_bv_type'] = 'LH'
    df.loc[sig & (quadrant == 3), 'lisa_bv_type'] = 'LL'
    df.loc[sig & (quadrant == 4), 'lisa_bv_type'] = 'HL'

    n_hh = (sig & (quadrant == 1)).sum()
    n_ll = (sig & (quadrant == 3)).sum()
    n_hl = (sig & (quadrant == 4)).sum()
    n_lh = (sig & (quadrant == 2)).sum()

    print(f"    HH (co-cluster positif)  : {n_hh} patients")
    print(f"      → zones où mutés NF ET forte exposition SH coexistent")
    print(f"    LL (co-cluster négatif)  : {n_ll} patients")
    print(f"    HL/LH (outliers)         : {n_hl + n_lh} patients")
    print(f"    Non significatif         : {(~sig).sum()} patients")

    palette = {'HH':'#C0392B','LH':'#2980B9','LL':'#27AE60',
               'HL':'#E67E22','Non sig.':'#D5D8DC'}

    fig, ax = plt.subplots(figsize=(12, 10))

    for type_lisa, color in palette.items():
        sub = df[df['lisa_bv_type'] == type_lisa]
        if len(sub) == 0:
            continue
        ax.scatter(sub['x_l93'].values / 1000,
                   sub['y_l93'].values / 1000,
                   c=color, s=15 if type_lisa == 'Non sig.' else 45,
                   alpha=0.5 if type_lisa == 'Non sig.' else 0.9,
                   label=f'{type_lisa} (n={len(sub)})',
                   zorder=2 if type_lisa == 'Non sig.' else 3)

    ax.scatter(icpe_sh.geometry.x.values / 1000,
               icpe_sh.geometry.y.values / 1000,
               c='black', marker='*', s=150, zorder=5,
               label=f'Seveso SH (n={len(icpe_sh)})')

    ax.set_xlabel('Lambert 93 X (km)', fontsize=11)
    ax.set_ylabel('Lambert 93 Y (km)', fontsize=11)
    ax.set_title(
        f'Carte LISA bivarié — {label_x} × {label_y}\n'
        f'HH={n_hh} | LL={n_ll} | HL={n_hl} | LH={n_lh} '
        f'(p < {alpha}, {n_perm} permutations)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right',
              title='Clusters LISA bivarié', title_fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/06_lisa_bivarie_{var_x}_{var_y}.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return df, lisa_bv, {'HH': n_hh, 'LL': n_ll, 'HL': n_hl, 'LH': n_lh}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — SYNTHÈSE COMPARATIVE
# ─────────────────────────────────────────────────────────────────────────────

def synthese_resultats(resultats_moran_global, resultats_bivarie):
    """Tableau synthèse de toutes les analyses de Moran."""

    print("\n" + "═" * 65)
    print("SYNTHÈSE DES ANALYSES DE MORAN")
    print("═" * 65)

    rows = []

    for res in resultats_moran_global:
        rows.append({
            'Analyse'   : f"Moran I global — {res['label']}",
            'Type'      : 'Univarié',
            'I'         : round(res['I'], 4),
            'z-score'   : round(res['z'], 4),
            'p-value'   : res['p_sim'],
            'p_fmt'     : '<0.001' if res['p_sim'] < 0.001 else f"{res['p_sim']:.3f}",
            'Sig'       : '✅' if res['p_sim'] < 0.05 else '—',
            'Interp.'   : ('Clustering positif' if res['I'] > 0
                           else 'Clustering négatif' if res['I'] < 0
                           else 'Aléatoire'),
        })

    for res in resultats_bivarie:
        rows.append({
            'Analyse'   : f"Moran I bivarié — {res['label_x']} × {res['label_y']}",
            'Type'      : 'Bivarié',
            'I'         : round(res['I'], 4),
            'z-score'   : round(res['z'], 4),
            'p-value'   : res['p_sim'],
            'p_fmt'     : '<0.001' if res['p_sim'] < 0.001 else f"{res['p_sim']:.3f}",
            'Sig'       : '✅' if res['p_sim'] < 0.05 else '—',
            'Interp.'   : ('Co-clustering positif' if res['I'] > 0
                           else 'Co-clustering négatif' if res['I'] < 0
                           else 'Aucune association spatiale'),
        })

    df_synth = pd.DataFrame(rows)
    print(df_synth[['Analyse','Type','I','z-score','p_fmt','Sig','Interp.']].to_string(index=False))

    # Barplot résumé
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.5 + 1)))
    colors = ['#2E86AB' if r['Type'] == 'Univarié' else '#7B2D8B' for _, r in df_synth.iterrows()]
    edge_colors = ['red' if r['Sig'] == '✅' else 'gray' for _, r in df_synth.iterrows()]
    bars = ax.barh(range(len(df_synth)), df_synth['I'].values,
                   color=colors, edgecolor=edge_colors, linewidth=2, alpha=0.8)
    ax.axvline(x=0, color='black', lw=1.2, linestyle='--')
    for i, (bar, row) in enumerate(zip(bars, df_synth.itertuples())):
        ax.text(bar.get_width() + 0.001 if bar.get_width() >= 0 else bar.get_width() - 0.001,
                i, f" p={row.p_fmt} {row.Sig}",
                va='center', fontsize=9,
                color='red' if row.Sig == '✅' else 'gray')
    ax.set_yticks(range(len(df_synth)))
    ax.set_yticklabels(df_synth['Analyse'].tolist(), fontsize=9)
    ax.set_xlabel("Moran's I", fontsize=11)
    ax.set_title("Synthèse — Indices de Moran\n"
                 "(rouge = significatif, bleu = univarié, violet = bivarié)",
                 fontsize=12, fontweight='bold')
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='Univarié'),
        mpatches.Patch(color='#7B2D8B', label='Bivarié'),
        Line2D([0], [0], color='red', lw=2, label='Significatif (p<0.05)'),
        Line2D([0], [0], color='gray', lw=2, label='Non significatif'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/07_synthese_moran.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return df_synth


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_analyse_moran(config=None):
    """
    Lance l'analyse spatiale de Moran complète.
    Peut être appelé depuis un notebook Jupyter ou en standalone.

    Paramètres
    ----------
    config : dict (optionnel)
        Si fourni, remplace CONFIG global.
        Utile pour appel depuis le notebook avec df_final déjà chargé.
    """
    import os
    if config is None:
        config = CONFIG

    # Créer dossier de sortie
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Figures sauvegardées dans : {config['output_dir']}\n")

    # ── Étape 1 : Données ─────────────────────────────────────────────────────
    df, gdf = charger_patients(config)
    icpe, icpe_sh, icpe_sb, icpe_ns = charger_icpe(config)
    df = calculer_expositions_icpe(df, icpe_sh, icpe_sb, config['rayon_m'])

    # ── Étape 2 : Matrice de poids ────────────────────────────────────────────
    W, n_voisins = construire_matrice_poids(df, config['rayon_m'],
                                             config['n_permutations'])

    # ── Étape 3 : Moran global univarié ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("ÉTAPE 3 — Moran's I global univarié")
    print("─" * 65)

    resultats_global = []

    # 3a. Mutations NF (Groupe A)
    res_A = moran_global_univarie(df, W, 'groupe_A',
                                   'Mutations NF (Groupe A)',
                                   '#2E86AB', config['n_permutations'])
    resultats_global.append(res_A)

    # 3b. Non-fumeurs (Groupe C)
    res_C = moran_global_univarie(df, W, 'groupe_C',
                                   'Non-fumeurs (Groupe C)',
                                   '#52B788', config['n_permutations'])
    resultats_global.append(res_C)

    # 3c. Score exposition SH
    res_SH = moran_global_univarie(df, W, 'score_expo_SH',
                                    'Score exposition Seveso SH',
                                    '#E76F51', config['n_permutations'])
    resultats_global.append(res_SH)

    # 3d. Nombre SH dans 3km
    res_nb = moran_global_univarie(df, W, 'nb_SH_3km',
                                    'Nb Seveso SH dans 3km',
                                    '#7B2D8B', config['n_permutations'])
    resultats_global.append(res_nb)

    # ── Étape 4 : LISA univarié ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("ÉTAPE 4 — LISA (Moran local)")
    print("─" * 65)

    df, lisa_A, clusters_A = lisa_local(df, W, 'groupe_A',
                                          'Mutations NF (Groupe A)',
                                          icpe_sh, config['n_permutations'],
                                          config['alpha'])

    df, lisa_C, clusters_C = lisa_local(df, W, 'groupe_C',
                                          'Non-fumeurs (Groupe C)',
                                          icpe_sh, config['n_permutations'],
                                          config['alpha'])

    # ── Étape 5 : Moran bivarié ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("ÉTAPE 5 — Moran's I bivarié")
    print("─" * 65)

    resultats_bivarie = []

    # 5a. Mutations NF × Score exposition SH
    res_bv1 = moran_bivarie(df, W,
                              'groupe_A', 'score_expo_SH',
                              'Mutations NF', 'Score expo SH',
                              config['n_permutations'])
    resultats_bivarie.append(res_bv1)

    # 5b. Mutations NF × Nb SH dans 3km
    res_bv2 = moran_bivarie(df, W,
                              'groupe_A', 'nb_SH_3km',
                              'Mutations NF', 'Nb SH 3km',
                              config['n_permutations'])
    resultats_bivarie.append(res_bv2)

    # 5c. Groupe A+C × Score exposition SH
    res_bv3 = moran_bivarie(df, W,
                              'groupe_AC', 'score_expo_SH',
                              'Mutations NF + Non-fumeurs (A+C)', 'Score expo SH',
                              config['n_permutations'])
    resultats_bivarie.append(res_bv3)

    # ── Étape 6 : LISA bivarié ────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("ÉTAPE 6 — LISA bivarié")
    print("─" * 65)

    df, lisa_bv1, clusters_bv1 = lisa_bivarie(
        df, W, 'groupe_A', 'score_expo_SH',
        'Mutations NF', 'Score expo SH',
        icpe_sh, config['n_permutations'], config['alpha'])

    df, lisa_bv2, clusters_bv2 = lisa_bivarie(
        df, W, 'groupe_AC', 'score_expo_SH',
        'Groupe A+C', 'Score expo SH',
        icpe_sh, config['n_permutations'], config['alpha'])

    # ── Étape 7 : Synthèse ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("ÉTAPE 7 — Synthèse")
    print("─" * 65)

    df_synthese = synthese_resultats(resultats_global, resultats_bivarie)

    # Sauvegarder les résultats
    df_synthese.to_csv(f"{config['output_dir']}/synthese_moran.csv",
                        index=False, encoding='utf-8-sig')
    df[['pseudo_provisoire', 'x_l93', 'y_l93',
        'groupe_A', 'groupe_C', 'groupe_AC',
        'dist_SH_m', 'nb_SH_3km', 'score_expo_SH',
        'lisa_q', 'lisa_sig', 'lisa_type',
        'lisa_bv_q', 'lisa_bv_sig', 'lisa_bv_type']].to_csv(
        f"{config['output_dir']}/patients_moran.csv",
        index=False, encoding='utf-8-sig')

    print(f"\n✅ Analyse terminée")
    print(f"   Figures    : {config['output_dir']}")
    print(f"   Résultats  : {config['output_dir']}/synthese_moran.csv")
    print(f"   Patients   : {config['output_dir']}/patients_moran.csv")

    return {
        'df'                  : df,
        'W'                   : W,
        'resultats_global'    : resultats_global,
        'resultats_bivarie'   : resultats_bivarie,
        'clusters_A'          : clusters_A,
        'clusters_bv1'        : clusters_bv1,
        'synthese'            : df_synthese,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    resultats = run_analyse_moran()
