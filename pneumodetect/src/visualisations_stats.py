# =============================================================================
# VISUALISATIONS STATISTIQUES AVANCÉES
# =============================================================================
# Ce module crée des visualisations graphiques des résultats statistiques:
# - Forest plots (OR, Cohen's d, comparaisons)
# - Heatmaps de corrélations et p-values
# - Courbes dose-réponse
# - Graphiques de tendance temporelle
# - Résumés visuels des tests
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from requests import patch
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, spearmanr, shapiro, norm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# =============================================================================
# CONFIGURATION
# =============================================================================

NOMS_GROUPES = {
    'NF': 'Non-fumeurs',
    'F-SNF': 'Fumeurs (mut. NF)',
    'F-SF': 'Fumeurs (mut. F)',
}

GROUPES_ANALYSE = ['NF', 'F-SNF', 'F-SF']

CONFIG_POLLUANTS = {
    'PM25': {'nom_complet': 'PM2.5', 'unite': 'μg/m³', 'color': '#e74c3c'},
    'PM10': {'nom_complet': 'PM10', 'unite': 'μg/m³', 'color': '#e67e22'},
    'NO2': {'nom_complet': 'NO2', 'unite': 'μg/m³', 'color': '#9b59b6'},
    'O3': {'nom_complet': 'O3', 'unite': 'μg/m³', 'color': '#3498db'},
}

PALETTE_GROUPES = {
    'NF': {'fill': '#27ae60', 'dark': '#1e8449'},
    'F-SNF': {'fill': '#3498db', 'dark': '#2171b5'},
    'F-SF': {'fill': '#e74c3c', 'dark': '#c0392b'},
}

PALETTE_QUARTILES = {
    1: {'fill': '#2ecc71', 'label': 'Q1 (faible)'},
    2: {'fill': '#f1c40f', 'label': 'Q2'},
    3: {'fill': '#e67e22', 'label': 'Q3'},
    4: {'fill': '#e74c3c', 'label': 'Q4 (élevé)'},
}

FENETRES_TEMPORELLES = {
    '0_6m': '0-6 mois',
    '6_12m': '6-12 mois', 
    '12_18m': '12-18 mois',
    '18_24m': '18-24 mois',
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def format_pvalue(p):
    """Formate une p-value pour affichage."""
    if pd.isna(p):
        return "N/A"
    elif p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f}"


def get_significance_color(p):
    """Retourne une couleur selon la significativité."""
    if pd.isna(p):
        return '#bdc3c7'
    elif p < 0.001:
        return '#c0392b'
    elif p < 0.01:
        return '#e74c3c'
    elif p < 0.05:
        return '#f39c12'
    else:
        return '#95a5a6'


def get_effect_size_color(d):
    """Retourne une couleur selon la taille d'effet (Cohen's d)."""
    if pd.isna(d):
        return '#bdc3c7'
    d_abs = abs(d)
    if d_abs >= 0.8:
        return '#c0392b'
    elif d_abs >= 0.5:
        return '#e74c3c'
    elif d_abs >= 0.2:
        return '#f39c12'
    else:
        return '#95a5a6'


def cohen_d(group1, group2):
    """Calcule le d de Cohen."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculer_or_quartile(df, col_quartile, col_outcome, reference=1):
    """Calcule les OR par quartile."""
    df_clean = df[[col_quartile, col_outcome]].dropna()
    resultats = {}
    
    ref_data = df_clean[df_clean[col_quartile] == reference]
    a_ref = ref_data[col_outcome].sum()
    b_ref = len(ref_data) - a_ref
    
    if a_ref == 0 or b_ref == 0:
        return None
    
    for q in [1, 2, 3, 4]:
        if q == reference:
            resultats[q] = {'OR': 1.0, 'IC_inf': 1.0, 'IC_sup': 1.0, 'p': 1.0}
            continue
        
        q_data = df_clean[df_clean[col_quartile] == q]
        a = q_data[col_outcome].sum()
        b = len(q_data) - a
        
        if a == 0 or b == 0:
            resultats[q] = {'OR': np.nan, 'IC_inf': np.nan, 'IC_sup': np.nan, 'p': np.nan}
            continue
        
        OR = (a * b_ref) / (b * a_ref)
        SE_log_OR = np.sqrt(1/a + 1/b + 1/a_ref + 1/b_ref)
        log_OR = np.log(OR)
        IC_inf = np.exp(log_OR - 1.96 * SE_log_OR)
        IC_sup = np.exp(log_OR + 1.96 * SE_log_OR)
        
        try:
            _, p_val, _, _ = chi2_contingency([[a, b], [a_ref, b_ref]])
        except:
            p_val = np.nan
        
        resultats[q] = {'OR': OR, 'IC_inf': IC_inf, 'IC_sup': IC_sup, 'p': p_val,
                       'n_cas': int(a), 'n_total': int(a+b)}
    
    return resultats


def test_tendance(df, col_quartile, col_outcome):
    """Test de tendance Cochran-Armitage."""
    df_clean = df[[col_quartile, col_outcome]].dropna()
    if len(df_clean) < 20:
        return {'Z': np.nan, 'p': np.nan}
    
    scores = df_clean[col_quartile].values
    outcomes = df_clean[col_outcome].values
    
    n = len(scores)
    n1 = outcomes.sum()
    n0 = n - n1
    
    if n0 == 0 or n1 == 0:
        return {'Z': np.nan, 'p': np.nan}
    
    T = np.sum(scores * outcomes)
    x_bar = np.mean(scores)
    E_T = n1 * x_bar
    var_scores = np.var(scores, ddof=0) * n
    Var_T = (n0 * n1 * var_scores) / (n * (n - 1))
    
    if Var_T <= 0:
        return {'Z': np.nan, 'p': np.nan}
    
    Z = (T - E_T) / np.sqrt(Var_T)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    return {'Z': Z, 'p': p_value}


# =============================================================================
# 1. FOREST PLOT - COMPARAISONS ENTRE GROUPES
# =============================================================================

def plot_forest_comparaisons(df, polluants=None, save_path=None):
    """
    Forest plot des comparaisons entre groupes pour tous les polluants.
    
    Affiche:
    - Différence de moyennes
    - Cohen's d avec IC
    - P-values Mann-Whitney
    """
    if polluants is None:
        polluants = ['PM25', 'PM10', 'NO2', 'O3']
    
    comparaisons = [('NF', 'F-SF'), ('NF', 'F-SNF'), ('F-SNF', 'F-SF')]
    
    fig, axes = plt.subplots(1, len(polluants), figsize=(5*len(polluants), 8), sharey=True)
    if len(polluants) == 1:
        axes = [axes]
    
    for idx, pol in enumerate(polluants):
        ax = axes[idx]
        config = CONFIG_POLLUANTS.get(pol, {})
        col = f'{pol}_moyenne'
        
        if col not in df.columns:
            ax.text(0.5, 0.5, f'{pol}\nnon disponible', ha='center', va='center')
            continue
        
        y_positions = []
        labels = []
        
        for i, (g1, g2) in enumerate(comparaisons):
            y = len(comparaisons) - i - 1
            y_positions.append(y)
            
            data1 = df[df['groupe'] == g1][col].dropna()
            data2 = df[df['groupe'] == g2][col].dropna()
            
            if len(data1) < 5 or len(data2) < 5:
                labels.append(f'{g1} vs {g2}')
                continue
            
            # Calculs
            d = cohen_d(data1.values, data2.values)
            diff_moy = data1.mean() - data2.mean()
            _, p = mannwhitneyu(data1.values, data2.values)
            
            # IC pour Cohen's d (approximation)
            se_d = np.sqrt((len(data1) + len(data2)) / (len(data1) * len(data2)) + d**2 / (2*(len(data1)+len(data2))))
            d_inf = d - 1.96 * se_d
            d_sup = d + 1.96 * se_d
            
            # Couleur selon significativité
            color = get_significance_color(p)
            
            # Point et barre d'erreur
            ax.errorbar(d, y, xerr=[[d - d_inf], [d_sup - d]], 
                       fmt='o', markersize=12, color=color,
                       ecolor=color, capsize=6, capthick=2, elinewidth=2)
            
            # Annotations
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.text(ax.get_xlim()[1] * 0.95 if ax.get_xlim()[1] > 0 else 2, y,
                   f'd={d:.2f} {sig}\nΔμ={diff_moy:+.1f}',
                   va='center', ha='right', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            labels.append(f'{g1} vs {g2}')
        
        # Ligne de référence (pas d'effet)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
        
        # Zones d'interprétation
        ax.axvspan(-0.2, 0.2, alpha=0.1, color='green', label='Négligeable')
        ax.axvspan(0.2, 0.5, alpha=0.1, color='yellow')
        ax.axvspan(-0.5, -0.2, alpha=0.1, color='yellow')
        ax.axvspan(0.5, 0.8, alpha=0.1, color='orange')
        ax.axvspan(-0.8, -0.5, alpha=0.1, color='orange')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Cohen's d", fontsize=12)
        ax.set_title(f'{config.get("nom_complet", pol)}', fontsize=13, fontweight='bold',
                    color=config.get('color', 'black'))
        ax.set_xlim(-1.5, 1.5)
        ax.grid(axis='x', alpha=0.3)
    
    # Légende commune
    legend_elements = [
        mpatches.Patch(facecolor='#c0392b', label='p < 0.001'),
        mpatches.Patch(facecolor='#e74c3c', label='p < 0.01'),
        mpatches.Patch(facecolor='#f39c12', label='p < 0.05'),
        mpatches.Patch(facecolor='#95a5a6', label='p ≥ 0.05'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
              title='Significativité', fontsize=10)
    
    fig.suptitle("Forest Plot: Tailles d'effet (Cohen's d) par polluant\n"
                 "(d > 0 = groupe 1 plus exposé que groupe 2)",
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# 2. HEATMAP DES P-VALUES ET EFFETS
# =============================================================================

def plot_heatmap_stats(df, polluants=None, save_path=None):
    """
    Heatmap montrant les p-values et tailles d'effet pour toutes les comparaisons.
    """
    if polluants is None:
        polluants = ['PM25', 'PM10', 'NO2', 'O3']
    
    comparaisons = [('NF', 'F-SF'), ('NF', 'F-SNF'), ('F-SNF', 'F-SF')]
    
    # Créer matrices
    matrix_p = np.zeros((len(comparaisons), len(polluants)))
    matrix_d = np.zeros((len(comparaisons), len(polluants)))
    matrix_diff = np.zeros((len(comparaisons), len(polluants)))
    
    for j, pol in enumerate(polluants):
        col = f'{pol}_moyenne'
        if col not in df.columns:
            matrix_p[:, j] = np.nan
            matrix_d[:, j] = np.nan
            continue
        
        for i, (g1, g2) in enumerate(comparaisons):
            data1 = df[df['groupe'] == g1][col].dropna()
            data2 = df[df['groupe'] == g2][col].dropna()
            
            if len(data1) < 5 or len(data2) < 5:
                matrix_p[i, j] = np.nan
                matrix_d[i, j] = np.nan
                continue
            
            _, p = mannwhitneyu(data1.values, data2.values)
            d = cohen_d(data1.values, data2.values)
            diff = data1.mean() - data2.mean()
            
            matrix_p[i, j] = p
            matrix_d[i, j] = d
            matrix_diff[i, j] = diff
    
    # Figure avec 2 heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap 1: P-values
    ax1 = axes[0]
    
    # Transformer p-values en -log10(p) pour meilleure visualisation
    matrix_logp = -np.log10(matrix_p + 1e-10)
    
    im1 = ax1.imshow(matrix_logp, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=4)
    
    # Annotations
    for i in range(len(comparaisons)):
        for j in range(len(polluants)):
            p = matrix_p[i, j]
            if not np.isnan(p):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                text_color = 'white' if matrix_logp[i, j] > 2 else 'black'
                ax1.text(j, i, f'{p:.3f}\n{sig}', ha='center', va='center', 
                        fontsize=10, color=text_color, fontweight='bold')
    
    ax1.set_xticks(range(len(polluants)))
    ax1.set_xticklabels([CONFIG_POLLUANTS.get(p, {}).get('nom_complet', p) for p in polluants], fontsize=11)
    ax1.set_yticks(range(len(comparaisons)))
    ax1.set_yticklabels([f'{g1} vs {g2}' for g1, g2 in comparaisons], fontsize=11)
    ax1.set_title('P-values (Mann-Whitney)\n(vert = non significatif, rouge = très significatif)', 
                 fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('-log(p)', fontsize=10)
    
    # Heatmap 2: Cohen's d
    ax2 = axes[1]
    
    # Colormap divergente centrée sur 0
    max_d = np.nanmax(np.abs(matrix_d))
    im2 = ax2.imshow(matrix_d, cmap='RdBu_r', aspect='auto', vmin=-max_d, vmax=max_d)
    
    # Annotations
    for i in range(len(comparaisons)):
        for j in range(len(polluants)):
            d = matrix_d[i, j]
            diff = matrix_diff[i, j]
            if not np.isnan(d):
                text_color = 'white' if abs(d) > max_d * 0.5 else 'black'
                ax2.text(j, i, f'd={d:.2f}\nΔ={diff:+.1f}', ha='center', va='center',
                        fontsize=9, color=text_color, fontweight='bold')
    
    ax2.set_xticks(range(len(polluants)))
    ax2.set_xticklabels([CONFIG_POLLUANTS.get(p, {}).get('nom_complet', p) for p in polluants], fontsize=11)
    ax2.set_yticks(range(len(comparaisons)))
    ax2.set_yticklabels([f'{g1} vs {g2}' for g1, g2 in comparaisons], fontsize=11)
    ax2.set_title("Tailles d'effet (Cohen's d)\n(rouge = G1 > G2, bleu = G1 < G2)",
                 fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Cohen's d", fontsize=10)
    
    fig.suptitle('Résumé des comparaisons statistiques entre groupes',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# 3. GRAPHIQUE DOSE-RÉPONSE VISUEL
# =============================================================================

def plot_dose_reponse_complet(df, polluant, col_outcome, outcome_label='Stade avancé',
                               seuils_quartiles=None, save_path=None):
    """
    Graphique dose-réponse complet avec:
    - Courbe des OR
    - Barres de prévalence par quartile
    - Test de tendance
    - Comparaison par groupe
    """
    col_quartile = f'{polluant}_quartile'
    col_moy = f'{polluant}_moyenne'
    config = CONFIG_POLLUANTS.get(polluant, {})
    
    if col_quartile not in df.columns or col_outcome not in df.columns:
        print(f"⚠️ Colonnes manquantes: {col_quartile} ou {col_outcome}")
        return None
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1])
    
    # ===== Panel A: Forest plot des OR (global) =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    or_global = calculer_or_quartile(df, col_quartile, col_outcome)
    tendance_global = test_tendance(df, col_quartile, col_outcome)
    
    if or_global:
        quartiles = [1, 2, 3, 4]
        y_pos = np.arange(len(quartiles))
        
        for i, q in enumerate(quartiles):
            r = or_global.get(q, {})
            or_val = r.get('OR', np.nan)
            ic_inf = r.get('IC_inf', np.nan)
            ic_sup = r.get('IC_sup', np.nan)
            p = r.get('p', np.nan)
            
            color = PALETTE_QUARTILES[q]['fill']
            
            if q == 1:
                ax1.scatter(1, y_pos[i], marker='s', s=200, color=color, 
                           edgecolor='black', linewidth=2, zorder=10)
                ax1.text(1.1, y_pos[i], 'Référence', va='center', fontsize=10, fontweight='bold')
            elif not np.isnan(or_val):
                ax1.errorbar(or_val, y_pos[i], 
                            xerr=[[or_val - ic_inf], [ic_sup - or_val]],
                            fmt='o', markersize=14, color=color,
                            ecolor='black', capsize=6, capthick=2, elinewidth=2, zorder=10)
                
                sig = '*' if p < 0.05 else ''
                n_info = f"n={r.get('n_cas', '?')}/{r.get('n_total', '?')}"
                ax1.text(ic_sup + 0.1, y_pos[i], 
                        f'OR={or_val:.2f} [{ic_inf:.2f}-{ic_sup:.2f}]{sig}\n{n_info}',
                        va='center', fontsize=9)
        
        ax1.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='Pas d\'effet')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([PALETTE_QUARTILES[q]['label'] for q in quartiles], fontsize=11)
        ax1.set_xlabel('Odds Ratio', fontsize=12)
        ax1.set_xlim(0, max(3, ax1.get_xlim()[1]))
        ax1.grid(axis='x', alpha=0.3)
        
        # Annotation tendance
        p_trend = tendance_global.get('p', np.nan)
        z_trend = tendance_global.get('Z', np.nan)
        sig_trend = '***' if p_trend < 0.001 else '**' if p_trend < 0.01 else '*' if p_trend < 0.05 else 'ns'
        
        trend_color = '#27ae60' if p_trend < 0.05 else '#95a5a6'
        ax1.text(0.95, 0.95, f'Test de tendance\nZ = {z_trend:.2f}\np = {p_trend:.4f} {sig_trend}',
                transform=ax1.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=trend_color, alpha=0.3),
                fontweight='bold')
    
    ax1.set_title(f'A. Odds Ratios par quartile (global)\n{config.get("nom_complet", polluant)} → {outcome_label}',
                 fontsize=12, fontweight='bold')
    
    # ===== Panel B: Prévalence par quartile =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    prevalences = []
    for q in [1, 2, 3, 4]:
        df_q = df[df[col_quartile] == q]
        n_total = len(df_q)
        n_cas = df_q[col_outcome].sum() if n_total > 0 else 0
        prev = n_cas / n_total * 100 if n_total > 0 else 0
        prevalences.append({'q': q, 'prev': prev, 'n_cas': n_cas, 'n_total': n_total})
    
    bars = ax2.bar([1, 2, 3, 4], [p['prev'] for p in prevalences],
                   color=[PALETTE_QUARTILES[q]['fill'] for q in [1, 2, 3, 4]],
                   edgecolor='black', linewidth=1.5)
    
    for bar, p in zip(bars, prevalences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{p["prev"]:.1f}%\n({p["n_cas"]}/{p["n_total"]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ligne de tendance
    x_trend = np.array([1, 2, 3, 4])
    y_trend = np.array([p['prev'] for p in prevalences])
    z = np.polyfit(x_trend, y_trend, 1)
    p_line = np.poly1d(z)
    ax2.plot(x_trend, p_line(x_trend), 'r--', linewidth=2, label=f'Tendance (pente={z[0]:.1f}%/Q)')
    
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels([PALETTE_QUARTILES[q]['label'] for q in [1, 2, 3, 4]], fontsize=10)
    ax2.set_ylabel(f'Prévalence {outcome_label} (%)', fontsize=11)
    ax2.set_xlabel(f'Quartile de {config.get("nom_complet", polluant)}', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.set_title(f'B. Prévalence par quartile', fontsize=12, fontweight='bold')
    
    # ===== Panel C: OR par groupe =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    y_offset = 0
    group_positions = {}
    
    for groupe in GROUPES_ANALYSE:
        df_g = df[df['groupe'] == groupe]
        if len(df_g) < 30:
            continue
        
        or_g = calculer_or_quartile(df_g, col_quartile, col_outcome)
        tendance_g = test_tendance(df_g, col_quartile, col_outcome)
        
        if or_g is None:
            continue
        
        group_positions[groupe] = y_offset
        
        # Nom du groupe
        ax3.text(-0.5, y_offset + 1.5, NOMS_GROUPES[groupe], fontsize=11, 
                fontweight='bold', color=PALETTE_GROUPES[groupe]['dark'])
        
        for i, q in enumerate([1, 2, 3, 4]):
            y = y_offset + i
            r = or_g.get(q, {})
            or_val = r.get('OR', np.nan)
            ic_inf = r.get('IC_inf', np.nan)
            ic_sup = r.get('IC_sup', np.nan)
            
            if q == 1:
                ax3.scatter(1, y, marker='s', s=100, color=PALETTE_QUARTILES[q]['fill'],
                           edgecolor='black', linewidth=1.5)
            elif not np.isnan(or_val):
                ax3.errorbar(or_val, y, xerr=[[or_val-ic_inf], [ic_sup-or_val]],
                            fmt='o', markersize=10, color=PALETTE_QUARTILES[q]['fill'],
                            ecolor=PALETTE_GROUPES[groupe]['dark'], capsize=4, capthick=1.5)
        
        # P-trend
        p_trend = tendance_g.get('p', np.nan)
        sig = '***' if p_trend < 0.001 else '**' if p_trend < 0.01 else '*' if p_trend < 0.05 else ''
        color_trend = '#27ae60' if p_trend < 0.05 else '#bdc3c7'
        ax3.text(ax3.get_xlim()[1] * 0.9 if ax3.get_xlim()[1] > 1 else 3.5, y_offset + 1.5,
                f'p-trend={p_trend:.3f}{sig}', fontsize=10, ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color_trend, alpha=0.3))
        
        y_offset += 5
    
    ax3.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Odds Ratio', fontsize=11)
    ax3.set_xlim(0, 5)
    ax3.set_yticks([])
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_title('C. Odds Ratios par groupe', fontsize=12, fontweight='bold')
    
    # Légende quartiles
    legend_elements = [mpatches.Patch(facecolor=PALETTE_QUARTILES[q]['fill'], 
                                      edgecolor='black', label=f'Q{q}') for q in [1, 2, 3, 4]]
    ax3.legend(handles=legend_elements, loc='lower right', title='Quartile')
    
    # ===== Panel D: Distribution quartiles par groupe =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(GROUPES_ANALYSE))
    width = 0.2
    
    for i, q in enumerate([1, 2, 3, 4]):
        pcts = []
        for groupe in GROUPES_ANALYSE:
            df_g = df[df['groupe'] == groupe]
            total = df_g[col_quartile].notna().sum()
            n_q = (df_g[col_quartile] == q).sum()
            pct = n_q / total * 100 if total > 0 else 0
            pcts.append(pct)
        
        ax4.bar(x + (i - 1.5) * width, pcts, width,
               color=PALETTE_QUARTILES[q]['fill'], edgecolor='black',
               label=f'Q{q}' if i == 0 else '')
    
    ax4.axhline(y=25, color='gray', linestyle=':', linewidth=1.5)
    ax4.text(2.3, 26, '25% (uniforme)', fontsize=9, color='gray')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([NOMS_GROUPES[g] for g in GROUPES_ANALYSE], fontsize=10)
    ax4.set_ylabel('% de patients', fontsize=11)
    ax4.set_title('D. Distribution des quartiles par groupe', fontsize=12, fontweight='bold')
    ax4.legend(title='Quartile', loc='upper right', ncol=4)
    
    # Chi-carré
    contingence = pd.crosstab(df['groupe'], df[col_quartile])
    if contingence.shape[0] >= 2 and contingence.shape[1] >= 2:
        chi2, p_chi2, _, _ = chi2_contingency(contingence)
        sig_chi = '***' if p_chi2 < 0.001 else '**' if p_chi2 < 0.01 else '*' if p_chi2 < 0.05 else ''
        ax4.text(0.02, 0.98, f'χ²={chi2:.1f}, p={p_chi2:.4f}{sig_chi}',
                transform=ax4.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    fig.suptitle(f'Analyse dose-réponse: {config.get("nom_complet", polluant)} → {outcome_label}',
                fontsize=15, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# 4. GRAPHIQUE TEMPOREL (FENÊTRES)
# =============================================================================

def plot_fenetres_temporelles_stats(df, polluant, save_path=None):
    """
    Visualisation des statistiques par fenêtre temporelle.
    """
    config = CONFIG_POLLUANTS.get(polluant, {})
    
    fenetres = list(FENETRES_TEMPORELLES.keys())
    labels = list(FENETRES_TEMPORELLES.values())
    
    # Collecter les données
    resultats = []
    for fenetre in fenetres:
        col = f'{polluant}_moy_{fenetre}'
        if col not in df.columns:
            continue
        
        row = {'fenetre': fenetre, 'label': FENETRES_TEMPORELLES[fenetre]}
        
        donnees = {}
        for groupe in GROUPES_ANALYSE:
            data = df[df['groupe'] == groupe][col].dropna()
            donnees[groupe] = data
            row[f'{groupe}_moy'] = data.mean() if len(data) > 0 else np.nan
            row[f'{groupe}_std'] = data.std() if len(data) > 0 else np.nan
        
        # Test Kruskal-Wallis
        groupes_data = [donnees[g].values for g in GROUPES_ANALYSE if len(donnees[g]) >= 5]
        if len(groupes_data) >= 2:
            H, p = kruskal(*groupes_data)
            row['KW_p'] = p
        else:
            row['KW_p'] = np.nan
        
        # Comparaison NF vs F-SF
        if len(donnees['NF']) >= 5 and len(donnees['F-SF']) >= 5:
            _, p_nf = mannwhitneyu(donnees['NF'].values, donnees['F-SF'].values)
            d_nf = cohen_d(donnees['NF'].values, donnees['F-SF'].values)
            row['NF_vs_FSF_p'] = p_nf
            row['NF_vs_FSF_d'] = d_nf
        
        resultats.append(row)
    
    if len(resultats) == 0:
        print("⚠️ Pas de données temporelles disponibles")
        return None
    
    df_res = pd.DataFrame(resultats)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ===== Panel A: Moyennes par groupe =====
    ax1 = axes[0, 0]
    
    x = np.arange(len(df_res))
    width = 0.25
    
    for i, groupe in enumerate(GROUPES_ANALYSE):
        moys = df_res[f'{groupe}_moy'].values
        stds = df_res[f'{groupe}_std'].values
        ax1.bar(x + (i-1)*width, moys, width, yerr=stds/2, 
               color=PALETTE_GROUPES[groupe]['fill'], edgecolor='black',
               label=NOMS_GROUPES[groupe], capsize=3)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_res['label'].values, rotation=15)
    ax1.set_ylabel(f'{config.get("nom_complet", polluant)} ({config.get("unite", "")})')
    ax1.legend()
    ax1.set_title('A. Exposition moyenne par fenêtre et groupe', fontweight='bold')
    
    # ===== Panel B: P-values par fenêtre =====
    ax2 = axes[0, 1]
    
    colors = [get_significance_color(p) for p in df_res['KW_p'].values]
    bars = ax2.bar(x, -np.log10(df_res['KW_p'].values + 1e-10), color=colors, edgecolor='black')
    
    # Lignes de significativité
    ax2.axhline(y=-np.log10(0.05), color='orange', linestyle='--', label='p=0.05')
    ax2.axhline(y=-np.log10(0.01), color='red', linestyle='--', label='p=0.01')
    ax2.axhline(y=-np.log10(0.001), color='darkred', linestyle='--', label='p=0.001')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_res['label'].values, rotation=15)
    ax2.set_ylabel('-log(p)')
    ax2.legend(loc='upper right')
    ax2.set_title('B. Significativité Kruskal-Wallis par fenêtre', fontweight='bold')
    
    # ===== Panel C: Cohen's d (NF vs F-SF) =====
    ax3 = axes[1, 0]
    
    if 'NF_vs_FSF_d' in df_res.columns:
        d_values = df_res['NF_vs_FSF_d'].values
        colors_d = [get_effect_size_color(d) for d in d_values]
        
        bars = ax3.barh(x, d_values, color=colors_d, edgecolor='black')
        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)
        ax3.axvline(x=0.2, color='green', linestyle=':', alpha=0.5)
        ax3.axvline(x=-0.2, color='green', linestyle=':', alpha=0.5)
        ax3.axvline(x=0.5, color='orange', linestyle=':', alpha=0.5)
        ax3.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.5)
        
        ax3.set_yticks(x)
        ax3.set_yticklabels(df_res['label'].values)
        ax3.set_xlabel("Cohen's d (NF vs F-SF)")
        ax3.set_title("C. Taille d'effet par fenêtre\n(d > 0 = NF plus exposés)", fontweight='bold')
        
        # Annotations
        for i, (bar, d) in enumerate(zip(bars, d_values)):
            if not np.isnan(d):
                text_x = d + 0.05 if d >= 0 else d - 0.05
                ha = 'left' if d >= 0 else 'right'
                ax3.text(text_x, i, f'{d:.2f}', va='center', ha=ha, fontsize=10)
    
    # ===== Panel D: Évolution temporelle =====
    ax4 = axes[1, 1]
    
    # Convertir fenêtres en position temporelle (mois avant diagnostic)
    x_temps = [3, 9, 15, 21][:len(df_res)]  # Milieu de chaque fenêtre
    
    for groupe in GROUPES_ANALYSE:
        moys = df_res[f'{groupe}_moy'].values
        ax4.plot(x_temps, moys, 'o-', markersize=10, linewidth=2,
                color=PALETTE_GROUPES[groupe]['fill'], label=NOMS_GROUPES[groupe])
    
    ax4.set_xlabel('Mois avant diagnostic')
    ax4.set_ylabel(f'{config.get("nom_complet", polluant)} ({config.get("unite", "")})')
    ax4.set_xticks(x_temps)
    ax4.set_xticklabels(['0-6m', '6-12m', '12-18m', '18-24m'][:len(df_res)])
    ax4.legend()
    ax4.invert_xaxis()  # Plus récent à gauche
    ax4.set_title('D. Évolution temporelle par groupe', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    fig.suptitle(f'Analyse temporelle: {config.get("nom_complet", polluant)}',
                fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# 5. RÉSUMÉ VISUEL GLOBAL (DASHBOARD)
# =============================================================================

def plot_dashboard_resume(df, polluant, col_outcome=None, outcome_label='Stade avancé',
                          seuils_quartiles=None, save_path=None):
    """
    Dashboard résumé avec tous les résultats clés sur une seule figure.
    """
    config = CONFIG_POLLUANTS.get(polluant, {})
    col_moy = f'{polluant}_moyenne'
    col_quartile = f'{polluant}_quartile'
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.8], 
                           width_ratios=[1, 1, 1, 1.2])
    
    # ===== Row 1: Distributions =====
    
    # Panel 1: Violin plot
    ax1 = fig.add_subplot(gs[0, 0])
    
    data_list = []
    for groupe in GROUPES_ANALYSE:
        data = df[df['groupe'] == groupe][col_moy].dropna().values
        data_list.append(data)
    
    parts = ax1.violinplot(data_list, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE_GROUPES[GROUPES_ANALYSE[i]]['fill'])
        pc.set_alpha(0.7)
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels([NOMS_GROUPES[g] for g in GROUPES_ANALYSE], fontsize=9)
    ax1.set_ylabel(config.get('unite', ''))
    ax1.set_title(f'Distribution {config.get("nom_complet", polluant)}', fontsize=11, fontweight='bold')
    
    # Panel 2: Boxplot par quartile
    ax2 = fig.add_subplot(gs[0, 1])
    
    if col_quartile in df.columns:
        data_q = [df[df[col_quartile] == q][col_moy].dropna().values for q in [1, 2, 3, 4]]
        bp = ax2.boxplot(data_q, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PALETTE_QUARTILES[i+1]['fill'])
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax2.set_ylabel(config.get('unite', ''))
        ax2.set_title('Distribution par quartile', fontsize=11, fontweight='bold')
    
    # Panel 3: Distribution quartiles par groupe
    ax3 = fig.add_subplot(gs[0, 2])
    
    if col_quartile in df.columns:
        x = np.arange(3)
        width = 0.2
        
        for i, q in enumerate([1, 2, 3, 4]):
            pcts = []
            for groupe in GROUPES_ANALYSE:
                df_g = df[df['groupe'] == groupe]
                total = df_g[col_quartile].notna().sum()
                n_q = (df_g[col_quartile] == q).sum()
                pcts.append(n_q / total * 100 if total > 0 else 0)
            ax3.bar(x + (i-1.5)*width, pcts, width, color=PALETTE_QUARTILES[q]['fill'], 
                   edgecolor='black', linewidth=0.5)
        
        ax3.axhline(25, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([NOMS_GROUPES[g].split()[0] for g in GROUPES_ANALYSE])
        ax3.set_ylabel('%')
        ax3.set_title('% par quartile/groupe', fontsize=11, fontweight='bold')
    
    # Panel 4: Statistiques clés (texte)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Calculer stats
    text_stats = f"STATISTIQUES CLÉS - {config.get('nom_complet', polluant)}\n"
    text_stats += "="*40 + "\n\n"
    
    for groupe in GROUPES_ANALYSE:
        data = df[df['groupe'] == groupe][col_moy].dropna()
        if len(data) > 0:
            text_stats += f"{NOMS_GROUPES[groupe]}:\n"
            text_stats += f"  n = {len(data)}\n"
            text_stats += f"  Moyenne = {data.mean():.1f} ± {data.std():.1f}\n"
            text_stats += f"  Médiane = {data.median():.1f}\n\n"
    
    # Kruskal-Wallis
    groupes_data = [df[df['groupe'] == g][col_moy].dropna().values for g in GROUPES_ANALYSE]
    groupes_data = [g for g in groupes_data if len(g) >= 5]
    if len(groupes_data) >= 2:
        H, p = kruskal(*groupes_data)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        text_stats += f"Kruskal-Wallis:\n"
        text_stats += f"  H = {H:.2f}, p = {p:.4f} {sig}\n"
    
    ax4.text(0.05, 0.95, text_stats, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ===== Row 2: Comparaisons et dose-réponse =====
    
    # Panel 5: Forest plot comparaisons
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    comparaisons = [('NF', 'F-SF'), ('NF', 'F-SNF'), ('F-SNF', 'F-SF')]
    y_pos = np.arange(len(comparaisons))
    
    for i, (g1, g2) in enumerate(comparaisons):
        data1 = df[df['groupe'] == g1][col_moy].dropna()
        data2 = df[df['groupe'] == g2][col_moy].dropna()
        
        if len(data1) >= 5 and len(data2) >= 5:
            d = cohen_d(data1.values, data2.values)
            _, p = mannwhitneyu(data1.values, data2.values)
            se_d = np.sqrt((len(data1)+len(data2))/(len(data1)*len(data2)) + d**2/(2*(len(data1)+len(data2))))
            
            color = get_significance_color(p)
            ax5.errorbar(d, y_pos[i], xerr=1.96*se_d, fmt='o', markersize=12,
                        color=color, ecolor=color, capsize=5, capthick=2)
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax5.text(d + 1.96*se_d + 0.1, y_pos[i], f'd={d:.2f}{sig}', va='center', fontsize=10)
    
    ax5.axvline(0, color='gray', linestyle='--')
    ax5.axvspan(-0.2, 0.2, alpha=0.1, color='green')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f'{g1} vs {g2}' for g1, g2 in comparaisons])
    ax5.set_xlabel("Cohen's d")
    ax5.set_title('Comparaisons entre groupes', fontsize=11, fontweight='bold')
    ax5.set_xlim(-1.5, 1.5)
    
    # Panel 6: Dose-réponse (si outcome disponible)
    ax6 = fig.add_subplot(gs[1, 2:4])
    
    if col_outcome and col_outcome in df.columns and col_quartile in df.columns:
        or_global = calculer_or_quartile(df, col_quartile, col_outcome)
        tendance = test_tendance(df, col_quartile, col_outcome)
        
        if or_global:
            # Barres de prévalence
            prev_data = []
            for q in [1, 2, 3, 4]:
                df_q = df[df[col_quartile] == q]
                n_total = len(df_q)
                n_cas = df_q[col_outcome].sum() if n_total > 0 else 0
                prev = n_cas / n_total * 100 if n_total > 0 else 0
                prev_data.append(prev)
            
            ax6_twin = ax6.twinx()
            bars = ax6.bar([1, 2, 3, 4], prev_data, alpha=0.5,
                          color=[PALETTE_QUARTILES[q]['fill'] for q in [1, 2, 3, 4]],
                          edgecolor='black')
            ax6.set_ylabel(f'Prévalence {outcome_label} (%)', color='gray')
            
            # Courbe OR
            ors = [or_global[q]['OR'] for q in [1, 2, 3, 4]]
            ic_inf = [or_global[q]['IC_inf'] for q in [1, 2, 3, 4]]
            ic_sup = [or_global[q]['IC_sup'] for q in [1, 2, 3, 4]]
            
            ax6_twin.errorbar([1, 2, 3, 4], ors, 
                             yerr=[np.array(ors)-np.array(ic_inf), np.array(ic_sup)-np.array(ors)],
                             fmt='o-', markersize=10, color='darkred', linewidth=2,
                             capsize=5, capthick=2, label='OR')
            ax6_twin.axhline(1, color='gray', linestyle='--')
            ax6_twin.set_ylabel('Odds Ratio', color='darkred')
            ax6_twin.tick_params(axis='y', labelcolor='darkred')
            
            ax6.set_xticks([1, 2, 3, 4])
            ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
            ax6.set_xlabel(f'Quartile de {config.get("nom_complet", polluant)}')
            
            # Annotation tendance
            p_trend = tendance['p']
            sig = '***' if p_trend < 0.001 else '**' if p_trend < 0.01 else '*' if p_trend < 0.05 else 'ns'
            ax6.set_title(f'Dose-réponse: {outcome_label}\n(p-trend = {p_trend:.4f} {sig})',
                         fontsize=11, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Outcome non\ndisponible', ha='center', va='center',
                fontsize=12, transform=ax6.transAxes)
        ax6.axis('off')
    
    # ===== Row 3: Résumé visuel =====
    
    # Panel 7: Heatmap résumé
    ax7 = fig.add_subplot(gs[2, :])
    
    # Créer données pour heatmap
    metriques = ['Moyenne', 'Médiane', 'Std', 'P95']
    matrix_data = np.zeros((len(GROUPES_ANALYSE), len(metriques)))
    
    for i, groupe in enumerate(GROUPES_ANALYSE):
        data = df[df['groupe'] == groupe][col_moy].dropna()
        if len(data) > 0:
            matrix_data[i, 0] = data.mean()
            matrix_data[i, 1] = data.median()
            matrix_data[i, 2] = data.std()
            matrix_data[i, 3] = data.quantile(0.95)
    
    im = ax7.imshow(matrix_data, cmap='YlOrRd', aspect='auto')
    
    ax7.set_xticks(range(len(metriques)))
    ax7.set_xticklabels(metriques)
    ax7.set_yticks(range(len(GROUPES_ANALYSE)))
    ax7.set_yticklabels([NOMS_GROUPES[g] for g in GROUPES_ANALYSE])
    
    # Annotations
    for i in range(len(GROUPES_ANALYSE)):
        for j in range(len(metriques)):
            text_color = 'white' if matrix_data[i, j] > np.median(matrix_data) else 'black'
            ax7.text(j, i, f'{matrix_data[i, j]:.1f}', ha='center', va='center',
                    fontsize=11, color=text_color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax7, orientation='vertical', shrink=0.8)
    cbar.set_label(config.get('unite', ''))
    
    ax7.set_title(f'Résumé des métriques par groupe', fontsize=11, fontweight='bold')
    
    fig.suptitle(f'DASHBOARD: {config.get("nom_complet", polluant)} - Analyse complète',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# 6. CORRÉLATION ENTRE POLLUANTS
# =============================================================================

def plot_correlation_polluants(df, polluants=None, save_path=None):
    """
    Matrice de corrélation entre les polluants avec visualisation.
    """
    if polluants is None:
        polluants = ['PM25', 'PM10', 'NO2', 'O3']
    
    cols = [f'{p}_moyenne' for p in polluants if f'{p}_moyenne' in df.columns]
    
    if len(cols) < 2:
        print("⚠️ Pas assez de polluants disponibles")
        return None
    
    df_corr = df[cols].dropna()
    
    # Calcul corrélations Spearman
    corr_matrix = np.zeros((len(cols), len(cols)))
    pval_matrix = np.zeros((len(cols), len(cols)))
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            r, p = spearmanr(df_corr[col1], df_corr[col2])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Heatmap corrélations
    ax1 = axes[0]
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    labels = [CONFIG_POLLUANTS.get(p, {}).get('nom_complet', p) for p in polluants if f'{p}_moyenne' in df.columns]
    
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            r = corr_matrix[i, j]
            p = pval_matrix[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            text_color = 'white' if abs(r) > 0.5 else 'black'
            ax1.text(j, i, f'{r:.2f}{sig}', ha='center', va='center',
                    fontsize=11, color=text_color, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Corrélation (Spearman)')
    ax1.set_title('Matrice de corrélation entre polluants', fontweight='bold')
    
    # Panel 2: Scatter plots pour paires importantes
    ax2 = axes[1]
    
    # PM2.5 vs PM10 (typiquement forte corrélation)
    if 'PM25_moyenne' in df.columns and 'PM10_moyenne' in df.columns:
        for groupe in GROUPES_ANALYSE:
            df_g = df[df['groupe'] == groupe]
            ax2.scatter(df_g['PM25_moyenne'], df_g['PM10_moyenne'],
                       c=PALETTE_GROUPES[groupe]['fill'], alpha=0.5, s=30,
                       label=NOMS_GROUPES[groupe])
        
        r, p = spearmanr(df['PM25_moyenne'].dropna(), df['PM10_moyenne'].dropna())
        ax2.set_xlabel('PM2.5 (μg/m³)')
        ax2.set_ylabel('PM10 (μg/m³)')
        ax2.set_title(f'PM2.5 vs PM10\nr={r:.2f}, p={p:.2e}', fontweight='bold')
        ax2.legend()
        
        # Ligne de tendance
        x = df['PM25_moyenne'].dropna()
        y = df['PM10_moyenne'].dropna()
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x_line, p_line(x_line), 'k--', linewidth=2)
    
    fig.suptitle('Corrélations entre polluants', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# FONCTION PRINCIPALE: GÉNÉRER TOUTES LES VISUALISATIONS
# =============================================================================

def generer_toutes_visualisations(df, polluant='PM25', col_outcome=None,
                                   outcome_label='Stade avancé', save_dir=None):
    """
    Génère toutes les visualisations statistiques.
    
    Parameters
    ----------
    df : DataFrame avec données et groupes
    polluant : str (PM25, PM10, NO2, O3)
    col_outcome : str, colonne outcome binaire (optionnel)
    outcome_label : str, label pour l'outcome
    save_dir : str, dossier de sauvegarde
    
    Returns
    -------
    dict avec toutes les figures
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    config = CONFIG_POLLUANTS.get(polluant, {})
    
    print(f"\n{'='*60}")
    print(f"GÉNÉRATION DES VISUALISATIONS STATISTIQUES")
    print(f"Polluant: {config.get('nom_complet', polluant)}")
    print(f"{'='*60}")
    
    figures = {}
    
    # 1. Forest plot comparaisons
    print("\n📊 1. Forest plot des comparaisons...")
    save_path = f"{save_dir}/{polluant}_forest_comparaisons.png" if save_dir else None
    figures['forest_comparaisons'] = plot_forest_comparaisons(df, [polluant], save_path)
    
    # 2. Heatmap stats
    print("\n📊 2. Heatmap des statistiques...")
    save_path = f"{save_dir}/heatmap_stats_all.png" if save_dir else None
    figures['heatmap_stats'] = plot_heatmap_stats(df, save_path=save_path)
    
    # 3. Dose-réponse (si outcome)
    if col_outcome and col_outcome in df.columns:
        print(f"\n📊 3. Graphique dose-réponse ({outcome_label})...")
        save_path = f"{save_dir}/{polluant}_dose_reponse.png" if save_dir else None
        figures['dose_reponse'] = plot_dose_reponse_complet(df, polluant, col_outcome, 
                                                            outcome_label, save_path=save_path)
    
    # 4. Analyse temporelle
    print("\n📊 4. Analyse temporelle...")
    save_path = f"{save_dir}/{polluant}_fenetres_temporelles.png" if save_dir else None
    figures['fenetres'] = plot_fenetres_temporelles_stats(df, polluant, save_path)
    
    # 5. Dashboard résumé
    print("\n📊 5. Dashboard résumé...")
    save_path = f"{save_dir}/{polluant}_dashboard.png" if save_dir else None
    figures['dashboard'] = plot_dashboard_resume(df, polluant, col_outcome, outcome_label,
                                                  save_path=save_path)
    
    # 6. Corrélations polluants
    print("\n📊 6. Corrélations entre polluants...")
    save_path = f"{save_dir}/correlations_polluants.png" if save_dir else None
    figures['correlations'] = plot_correlation_polluants(df, save_path=save_path)
    
    print(f"\n✅ Toutes les visualisations générées!")
    if save_dir:
        print(f"📁 Fichiers sauvegardés dans: {save_dir}/")
    
    return figures



# =============================================================================
# BIBLIOTHÈQUE D'ANALYSE COMPLÈTE — TOUTES ÉTAPES
# =============================================================================
# from analyse_lib import *
# aide()  → affiche le catalogue complet
# =============================================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# from scipy.stats import (spearmanr, kruskal, mannwhitneyu,
#                          chi2_contingency, shapiro, norm)
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

# plt.rcParams['figure.dpi'] = 120
# plt.rcParams['font.size'] = 11
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False

# GROUPES  = ['NF', 'F-SNF', 'F-SF']
# NOMS     = {'NF': 'Non-fumeurs', 'F-SNF': 'Fumeurs (mut. NF)', 'F-SF': 'Fumeurs (mut. F)'}
# COULEURS = {'NF': '#27ae60', 'F-SNF': '#3498db', 'F-SF': '#e74c3c'}

# PALETTE_QUARTILES = {
#     1: {'fill': '#2ecc71', 'edge': '#27ae60', 'label': 'Q1 (faible)'},
#     2: {'fill': '#f1c40f', 'edge': '#d4ac0d', 'label': 'Q2 (modéré-faible)'},
#     3: {'fill': '#e67e22', 'edge': '#ca6f1e', 'label': 'Q3 (modéré-élevé)'},
#     4: {'fill': '#e74c3c', 'edge': '#c0392b', 'label': 'Q4 (élevé)'},
# }




# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ██╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ███║
# ██████╔╝██║     ██║   ██║██║         ╚██║
# ██╔══██╗██║     ██║   ██║██║          ██║
# ██████╔╝███████╗╚██████╔╝╚██████╗     ██║
# ÉTAPE 1 — DESCRIPTION DE LA COHORTE
# =============================================================================

def analyse_correlation(df, col1, col2, groupe=None, plot=True, save_path=None):
    """
    Corrélation de Spearman entre deux variables + scatter plot.

    Exemples :
        analyse_correlation(df_final, 'PM25_moyenne', 'PM25_P95')
        analyse_correlation(df_final, 'PM25_moyenne', 'PM25_cv', groupe='NF')
        analyse_correlation(df_final, 'PM25_moyenne', 'PM25_pct_Q4', plot=False)
    """
    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    df_w = df_w[[col1, col2, 'groupe']].dropna(subset=[col1, col2])

    r, p = spearmanr(df_w[col1], df_w[col2])

    if abs(r) > 0.80:
        interpretation = "⛔ TRÈS FORTE — redondantes, ne pas mettre ensemble dans un modèle"
        couleur_box    = '#ffe0e0'
    elif abs(r) > 0.60:
        interpretation = "⚠️  FORTE — partiellement redondantes, tester ΔAIC"
        couleur_box    = '#fff3cd'
    elif abs(r) > 0.40:
        interpretation = "📊 MODÉRÉE — partiellement indépendantes"
        couleur_box    = '#e8f4f8'
    else:
        interpretation = "✅ FAIBLE — deux dimensions indépendantes"
        couleur_box    = '#e0f0e0'

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    titre_groupe = f" | Groupe : {NOMS_GROUPES.get(groupe, groupe)}" if groupe else " | Tous patients"

    print(f"\n{'─'*60}")
    print(f"CORRÉLATION : {col1}  ↔  {col2}{titre_groupe}")
    print(f"{'─'*60}")
    print(f"  r de Spearman = {r:.3f}")
    print(f"  p-value       = {p:.2e} {sig}")
    print(f"  n             = {len(df_w)}")
    print(f"  → {interpretation}")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        for g in GROUPES_ANALYSE:
            d = df_w[df_w['groupe'] == g]
            if len(d) > 0:
                ax.scatter(d[col1], d[col2], color=PALETTE_GROUPES[g],
                           alpha=0.4, s=20, edgecolors='none',
                           label=f"{NOMS_GROUPES[g]} (n={len(d)})")
                ax.scatter(d[col1].mean(), d[col2].mean(),
                           color=PALETTE_GROUPES[g], s=150, marker='*',
                           edgecolors='black', linewidths=1.2, zorder=10)
        z      = np.polyfit(df_w[col1], df_w[col2], 1)
        x_line = np.linspace(df_w[col1].min(), df_w[col1].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'k--', linewidth=1.5, alpha=0.6)
        ax.text(0.03, 0.97, f'r = {r:.3f}, p = {p:.2e} {sig}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax.text(0.03, 0.04, interpretation, transform=ax.transAxes, fontsize=9,
                va='bottom', bbox=dict(boxstyle='round', facecolor=couleur_box, alpha=0.85))
        ax.set_xlabel(col1, fontsize=11)
        ax.set_ylabel(col2, fontsize=11)
        ax.set_title(f'Corrélation : {col1}  ↔  {col2}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return {'r': r, 'p': p, 'sig': sig, 'n': len(df_w), 'interpretation': interpretation}


def analyse_correlation_multiple(df, cols, titre='Matrice de corrélation',
                                  seuil_alerte=0.80, save_path=None):
    """
    Matrice de corrélation Spearman sur une liste de colonnes + heatmap.

    Exemples :
        analyse_correlation_multiple(df_final,
            ['PM25_moyenne', 'PM25_cv', 'PM25_pct_Q4', 'PM25_nb_episodes'])
        analyse_correlation_multiple(df_final,
            ['PM25_moyenne', 'PM10_moyenne', 'NO2_moyenne', 'O3_moyenne'])
    """
    cols_dispo = [c for c in cols if c in df.columns]
    if len(cols_dispo) < 2:
        print("⚠️  Moins de 2 colonnes disponibles")
        return None

    df_c = df[cols_dispo].dropna()
    n    = len(cols_dispo)
    mat_r, mat_p = np.zeros((n, n)), np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                mat_r[i,j], mat_p[i,j] = 1.0, 0.0
            else:
                r, p = spearmanr(df_c.iloc[:, i], df_c.iloc[:, j])
                mat_r[i,j], mat_p[i,j] = r, p

    labels = [c.replace('_moyenne', '') for c in cols_dispo]

    fig, ax = plt.subplots(figsize=(max(7, n*1.5), max(6, n*1.4)))
    im = ax.imshow(mat_r, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='r de Spearman', shrink=0.8)

    for i in range(n):
        for j in range(n):
            r_val, p_val = mat_r[i,j], mat_p[i,j]
            sig   = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            text  = '1.00' if i == j else f'{r_val:.2f}{sig}'
            color = 'white' if abs(r_val) > 0.55 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(titre, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n⚠️  Paires avec |r| > {seuil_alerte} :")
    found = False
    for i in range(n):
        for j in range(i+1, n):
            if abs(mat_r[i,j]) > seuil_alerte:
                found = True
                print(f"  {cols_dispo[i]}  ↔  {cols_dispo[j]} : r = {mat_r[i,j]:.3f}"
                      f" → Ne pas mettre ensemble dans un modèle")
    if not found:
        print(f"  ✅ Aucune paire > {seuil_alerte}")

    return pd.DataFrame(mat_r, index=cols_dispo, columns=cols_dispo)


def comparer_groupes(df, col, groupes=None, plot=True, save_path=None):
    """
    Compare une variable continue entre groupes.
    KW global + Mann-Whitney 2 à 2 + Cohen d + violin plot.

    Exemples :
        comparer_groupes(df_final, 'PM25_moyenne')
        comparer_groupes(df_final, 'NO2_moyenne', groupes=['NF', 'F-SF'])
        comparer_groupes(df_final, 'PM25_cv', plot=False)
    """
    if groupes is None:
        groupes = GROUPES_ANALYSE

    df_w = df[df['groupe'].isin(groupes)].copy()
    print(f"\n{'─'*65}")
    print(f"COMPARAISON : {col}")
    print(f"{'─'*65}")
    print(f"  {'Groupe':<25} {'n':>5} {'Médiane':>10} {'[Q1-Q3]':>22} {'Moyenne':>10}")
    print(f"  {'─'*70}")

    donnees = {}
    for g in groupes:
        d = df_w[df_w['groupe'] == g][col].dropna()
        donnees[g] = d.values
        if len(d) > 0:
            print(f"  {NOMS_GROUPES.get(g,g):<25} {len(d):>5} "
                  f"{d.median():>10.2f} "
                  f"[{d.quantile(0.25):.2f}-{d.quantile(0.75):.2f}]"
                  f"{d.mean():>10.2f}")

    valides  = [v for v in donnees.values() if len(v) >= 3]
    resultats = {'col': col, 'comparaisons': {}}

    if len(valides) >= 2:
        H, p_kw = kruskal(*valides)
        n_total = sum(len(v) for v in valides)
        eps2    = H / (n_total - 1)
        sig_kw  = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'ns'
        print(f"\n  Kruskal-Wallis : H={H:.2f}, p={p_kw:.4f} {sig_kw}, ε²={eps2:.3f}")
        resultats['KW'] = {'H': H, 'p': p_kw, 'eps2': eps2}

    paires  = [(groupes[i], groupes[j])
               for i in range(len(groupes)) for j in range(i+1, len(groupes))]
    n_tests = len(paires)

    print(f"\n  Comparaisons 2 à 2 (Bonferroni n={n_tests}) :")
    print(f"  {'Paire':<25} {'p-brut':>10} {'p-corr':>10} {'Cohen d':>10} {'Effet':>12} {'Δmoy':>8}")
    print(f"  {'─'*75}")

    for g1, g2 in paires:
        if len(donnees.get(g1,[])) >= 3 and len(donnees.get(g2,[])) >= 3:
            U, p     = mannwhitneyu(donnees[g1], donnees[g2])
            p_corr   = min(p * n_tests, 1.0)
            n1, n2   = len(donnees[g1]), len(donnees[g2])
            var_pool = (((n1-1)*np.var(donnees[g1], ddof=1) +
                         (n2-1)*np.var(donnees[g2], ddof=1)) / (n1+n2-2))
            d        = ((np.mean(donnees[g1]) - np.mean(donnees[g2]))
                        / np.sqrt(var_pool)) if var_pool > 0 else 0
            effet    = ("fort" if abs(d)>0.8 else "moyen" if abs(d)>0.5
                        else "faible" if abs(d)>0.2 else "négligeable")
            sig      = '*' if p_corr < 0.05 else ''
            dmoy     = np.mean(donnees[g1]) - np.mean(donnees[g2])
            print(f"  {g1} vs {g2:<16} {p:>10.4f} {p_corr:>10.4f} "
                  f"{d:>10.2f} {effet:>12} {dmoy:>+8.2f} {sig}")
            resultats['comparaisons'][f'{g1}_vs_{g2}'] = {
                'p': p, 'p_corr': p_corr, 'd': d, 'effet': effet,
                'delta_moy': dmoy, 'sig': p_corr < 0.05}

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        data_list = [donnees.get(g, np.array([])) for g in groupes]
        pos_valides = [i for i, d in enumerate(data_list) if len(d) > 0]
        dat_valides = [d for d in data_list if len(d) > 0]

        parts = ax.violinplot(dat_valides, positions=pos_valides,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PALETTE_GROUPES.get(groupes[pos_valides[i]], {}).get('fill', '#aaa'))
            pc.set_alpha(0.4)

        bp = ax.boxplot(dat_valides, positions=pos_valides,
                        widths=0.15, patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', linewidth=2))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PALETTE_GROUPES.get(groupes[pos_valides[i]], {}).get('fill', '#aaa'))
            patch.set_alpha(0.85)

        for i, g in enumerate(groupes):
            d = donnees.get(g, np.array([]))
            if len(d) > 0:
                ax.scatter(i, np.mean(d), marker='D', color='white',
                           s=60, zorder=10, edgecolors='black', linewidths=1.5)
                ax.text(i, ax.get_ylim()[1],
                        f'n={len(d)}\nméd={np.median(d):.1f}',
                        ha='center', va='top', fontsize=9)

        if 'KW' in resultats:
            kw = resultats['KW']
            sig_t = ('***' if kw['p']<0.001 else '**' if kw['p']<0.01
                     else '*' if kw['p']<0.05 else 'ns')
            ax.set_title(f'{col}\nKW: H={kw["H"]:.1f}, p={kw["p"]:.2e} {sig_t}',
                         fontsize=12, fontweight='bold')

        ax.set_xticks(range(len(groupes)))
        ax.set_xticklabels([NOMS_GROUPES.get(g,g) for g in groupes], fontsize=10)
        ax.set_ylabel(col, fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return resultats


def test_normalite(df, col, groupes=None):
    """
    Test de Shapiro-Wilk par groupe.

    Exemple :
        test_normalite(df_final, 'PM25_moyenne')
    """
    if groupes is None:
        groupes = GROUPES_ANALYSE
    print(f"\n  Shapiro-Wilk — {col}")
    print(f"  {'Groupe':<25} {'W':>8} {'p':>10} {'Normal ?':>10}")
    print(f"  {'─'*55}")
    resultats = {}
    for g in groupes:
        d = df[df['groupe'] == g][col].dropna().values
        if len(d) >= 3:
            sample = d[:5000] if len(d) > 5000 else d
            W, p   = shapiro(sample)
            normal = p > 0.05
            print(f"  {NOMS_GROUPES.get(g,g):<25} {W:>8.4f} {p:>10.4f} {'✓ oui' if normal else '✗ non':>10}")
            resultats[g] = {'W': W, 'p': p, 'normal': normal}
    return resultats


def tableau_descriptif(df, cols, groupes=None, decimales=2):
    """
    Tableau médiane [IQR] + KW pour une liste de colonnes.

    Exemple :
        tableau_descriptif(df_final,
            ['PM25_moyenne', 'PM25_cv', 'PM25_pct_Q4', 'PM25_nb_episodes'])
    """
    if groupes is None:
        groupes =  GROUPES_ANALYSE
    df_w    = df[df['groupe'].isin(groupes)].copy()
    cols_ok = [c for c in cols if c in df_w.columns]

    print(f"\n{'─'*80}")
    header = f"  {'Variable':<30}"
    for g in groupes:
        n = (df_w['groupe'] == g).sum()
        header += f"  {NOMS_GROUPES.get(g,g)[:18]} (n={n})"
    header += "  p-KW"
    print(header)
    print(f"  {'─'*80}")

    rows = []
    for col in cols_ok:
        row_data = {'Variable': col}
        row_str  = f"  {col:<30}"
        g_vals   = []
        for g in groupes:
            d   = df_w[df_w['groupe'] == g][col].dropna()
            g_vals.append(d.values)
            val = f"{d.median():.{decimales}f} [{d.quantile(0.25):.{decimales}f}-{d.quantile(0.75):.{decimales}f}]"
            row_str  += f"  {val:<28}"
            row_data[g] = val
        valides = [v for v in g_vals if len(v) >= 3]
        if len(valides) >= 2:
            H, p = kruskal(*valides)
            sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            row_str  += f"  {p:.4f} {sig}"
            row_data['p_KW'] = p
            row_data['sig']  = sig
        print(row_str)
        rows.append(row_data)

    return pd.DataFrame(rows)


def heatmap_mutations_polluants(df, mutations=None, polluants=None, save_path=None):
    """
    Heatmap de corrélation entre polluants et présence de mutations, par groupe.

    Exemple :
        heatmap_mutations_polluants(df_final)
        heatmap_mutations_polluants(df_final, mutations=['EGFR','ALK','KRAS'],
                                    polluants=['PM25_moyenne','NO2_moyenne'])
    """
    if mutations is None:
        mutations = ['EGFR','ALK','ROS1','KRAS','TP53','ERBB2','RET','MET','NTRK']
    if polluants is None:
        polluants = ['PM25_moyenne','PM10_moyenne','NO2_moyenne','O3_moyenne']

    cols_mut = [f'mutation_{m}' for m in mutations if f'mutation_{m}' in df.columns]
    cols_pol = [c for c in polluants if c in df.columns]

    if not cols_mut or not cols_pol:
        print("⚠️  Colonnes mutations ou polluants non trouvées")
        return None

    df_w = df.copy()
    for col in cols_mut:
        df_w[col] = (df_w[col].str.strip().str.lower() == 'positive').astype(float)

    fig, axes = plt.subplots(1, len(GROUPES_ANALYSE), figsize=(5*len(GROUPES_ANALYSE), max(5, len(cols_mut)*0.6+2)),
                              sharey=True)
    if len(GROUPES_ANALYSE) == 1:
        axes = [axes]

    for ax, g in zip(axes, GROUPES_ANALYSE):
        df_g  = df_w[df_w['groupe'] == g][cols_mut + cols_pol].dropna()
        mat   = np.zeros((len(cols_mut), len(cols_pol)))

        for i, cm in enumerate(cols_mut):
            for j, cp in enumerate(cols_pol):
                r, _ = spearmanr(df_g[cm], df_g[cp])
                mat[i, j] = r

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        for i in range(len(cols_mut)):
            for j in range(len(cols_pol)):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if abs(mat[i,j]) > 0.25 else 'black')

        ax.set_xticks(range(len(cols_pol)))
        ax.set_xticklabels([c.replace('_moyenne','') for c in cols_pol],
                           rotation=30, ha='right', fontsize=9)
        ax.set_title(f'{NOMS_GROUPES.get(g,g)}\n(n={len(df_g)})', fontsize=11,
                     fontweight='bold', color=PALETTE_GROUPES.get(g, {}).get('text', 'black'))

        if ax == axes[0]:
            ax.set_yticks(range(len(cols_mut)))
            ax.set_yticklabels([c.replace('mutation_','') for c in cols_mut], fontsize=10)

    plt.colorbar(im, ax=axes[-1], label='r de Spearman', shrink=0.8)
    fig.suptitle('Corrélation Polluants × Mutations par groupe', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ██████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ╚════██╗
# ██████╔╝██║     ██║   ██║██║          █████╔╝
# ██╔══██╗██║     ██║   ██║██║         ██╔═══╝
# ██████╔╝███████╗╚██████╔╝╚██████╗    ███████╗
# ÉTAPE 2 — COMPARAISONS ENTRE GROUPES (H1, H2)
# =============================================================================

def ancova_groupes(df, col_outcome_continu, col_covariable, col_groupe='groupe',
                   groupes_cibles=None, save_path=None):
    """
    ANCOVA : compare col_outcome entre groupes en contrôlant col_covariable.
    Usage principal : comparer F-SNF vs F-SF en contrôlant paquets-années.

    Exemples :
        ancova_groupes(df_final, 'PM25_moyenne', 'paquets_annees',
                       groupes_cibles=['F-SNF','F-SF'])
        ancova_groupes(df_final, 'NO2_moyenne', 'paquets_annees',
                       groupes_cibles=['F-SNF','F-SF'])
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("⚠️  statsmodels requis")
        return None

    if groupes_cibles is None:
        groupes_cibles = ['F-SNF', 'F-SF']

    df_w = df[df[col_groupe].isin(groupes_cibles)].copy()
    df_w = df_w[[col_outcome_continu, col_covariable, col_groupe]].dropna()
    df_w['groupe_bin'] = (df_w[col_groupe] == groupes_cibles[0]).astype(int)

    formula = f"{col_outcome_continu} ~ groupe_bin + {col_covariable}"
    model   = smf.ols(formula, data=df_w).fit()

    coef_groupe = model.params['groupe_bin']
    p_groupe    = model.pvalues['groupe_bin']
    ci          = model.conf_int().loc['groupe_bin']
    sig         = '***' if p_groupe<0.001 else '**' if p_groupe<0.01 else '*' if p_groupe<0.05 else 'ns'

    print(f"\n{'─'*65}")
    print(f"ANCOVA : {col_outcome_continu} ~ Groupe + {col_covariable}")
    print(f"{'─'*65}")
    print(f"  Groupes comparés  : {groupes_cibles[0]} vs {groupes_cibles[1]}")
    print(f"  n                 : {len(df_w)}")
    print(f"  Δ ajusté          : {coef_groupe:+.3f} μg/m³")
    print(f"  IC 95%            : [{ci[0]:.3f} ; {ci[1]:.3f}]")
    print(f"  p (groupe)        : {p_groupe:.4f} {sig}")
    print(f"  R²                : {model.rsquared:.3f}")

    if p_groupe < 0.05:
        print(f"  ✅ Différence significative après ajustement sur {col_covariable}")
    else:
        print(f"  ❌ Pas de différence significative après ajustement")

    # Scatter avec droites ajustées
    if save_path or True:
        fig, ax = plt.subplots(figsize=(9, 6))
        for g in groupes_cibles:
            d = df_w[df_w[col_groupe] == g]
            ax.scatter(d[col_covariable], d[col_outcome_continu],
                       color=PALETTE_GROUPES.get(g, {}).get('fill', 'gray'), alpha=0.4, s=20, label=NOMS_GROUPES.get(g,g))
            x_range = np.linspace(d[col_covariable].min(), d[col_covariable].max(), 50)
            gb      = 1 if g == groupes_cibles[0] else 0
            y_pred  = model.params['Intercept'] + gb*coef_groupe + model.params[col_covariable]*x_range
            ax.plot(x_range, y_pred, color=PALETTE_GROUPES.get(g, {}).get('fill', 'gray'), linewidth=2)

        ax.set_xlabel(col_covariable, fontsize=11)
        ax.set_ylabel(col_outcome_continu, fontsize=11)
        ax.set_title(f'ANCOVA : {col_outcome_continu} ajusté sur {col_covariable}\n'
                     f'Δ = {coef_groupe:+.2f}, p = {p_groupe:.4f} {sig}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return {'delta': coef_groupe, 'p': p_groupe, 'IC': [ci[0], ci[1]], 'model': model}


def barplot_comparatif(df, cols, titre='Comparaison par groupe', save_path=None):
    """
    Barplot des moyennes par groupe pour plusieurs variables.

    Exemple :
        barplot_comparatif(df_final,
            ['PM25_moyenne','PM10_moyenne','NO2_moyenne','O3_moyenne'],
            titre='Expositions moyennes par groupe')
    """
    df_w    = df[df['groupe'].isin(GROUPES_ANALYSE)].copy()
    cols_ok = [c for c in cols if c in df_w.columns]
    n_cols  = len(cols_ok)

    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_ok):
        means = [df_w[df_w['groupe']==g][col].mean() for g in GROUPES_ANALYSE]
        sems  = [df_w[df_w['groupe']==g][col].sem()  for g in GROUPES_ANALYSE]
        bars  = ax.bar(range(len(GROUPES_ANALYSE)), means, yerr=sems,
                       color=[PALETTE_GROUPES[g]['fill'] for g in GROUPES_ANALYSE],
                       edgecolor='black', linewidth=1.2,
                       error_kw=dict(elinewidth=1.5, capsize=5))
        valides = [df_w[df_w['groupe']==g][col].dropna().values for g in GROUPES_ANALYSE]
        valides = [v for v in valides if len(v) >= 3]
        if len(valides) >= 2:
            H, p = kruskal(*valides)
            sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            ax.set_title(f'{col}\np = {p:.4f} {sig}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(GROUPES_ANALYSE)))
        ax.set_xticklabels([NOMS_GROUPES.get(g,g)[:10] for g in GROUPES_ANALYSE], rotation=15, fontsize=9)
        ax.set_ylabel('Moyenne ± SEM')

    fig.suptitle(titre, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ██████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ╚════██╗
# ██████╔╝██║     ██║   ██║██║          █████╔╝
# ██╔══██╗██║     ██║   ██║██║         ╚═══██╔╝
# ██████╔╝███████╗╚██████╔╝╚██████╗    ██████╔╝
# ÉTAPE 3 — RÉGRESSION LOGISTIQUE
# =============================================================================

def regression_logistique(df, col_outcome, variables, groupe=None,
                           label='Modèle', save_path=None):
    """
    Régression logistique avec tableau OR + IC 95% + p, courbe ROC, Hosmer-Lemeshow, VIF.

    Exemples :
        regression_logistique(df_final, 'stade_avance',
            ['PM25_moyenne', 'age_diagnostic', 'sexe'])

        regression_logistique(df_final, 'stade_avance',
            ['PM25_moyenne', 'age_diagnostic', 'sexe'], groupe='NF')
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.metrics import roc_curve, auc as sk_auc
    except ImportError:
        print("⚠️  statsmodels et scikit-learn requis")
        return None

    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    cols_need = list(set(variables + [col_outcome]))
    df_w      = df_w[[c for c in cols_need if c in df_w.columns]].dropna()

    formula = f"{col_outcome} ~ {' + '.join(variables)}"
    model   = smf.logit(formula, data=df_w).fit(disp=0)

    titre_g = f" | Groupe : {NOMS_GROUPES.get(groupe,groupe)}" if groupe else " | Tous patients"
    print(f"\n{'═'*65}")
    print(f"RÉGRESSION LOGISTIQUE — {label}{titre_g}")
    print(f"{'═'*65}")
    print(f"  Outcome : {col_outcome} | n = {int(model.nobs)} | AIC = {model.aic:.1f}")
    print(f"\n  {'Variable':<30} {'OR':>8} {'IC 95%':>22} {'p':>10}")
    print(f"  {'─'*72}")

    resultats_vars = {}
    for var in model.params.index:
        if var == 'Intercept':
            continue
        coef   = model.params[var]
        OR     = np.exp(coef)
        ci     = model.conf_int().loc[var]
        p_val  = model.pvalues[var]
        IC_inf = np.exp(ci[0])
        IC_sup = np.exp(ci[1])
        sig    = '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'
        ic_str = f"[{IC_inf:.2f} - {IC_sup:.2f}]"
        print(f"  {var:<30} {OR:>8.3f} {ic_str:>22} {p_val:>9.4f} {sig}")
        resultats_vars[var] = {'OR': OR, 'IC_inf': IC_inf, 'IC_sup': IC_sup,
                               'p': p_val, 'sig': sig}

    # VIF
    print(f"\n  VIF (multicolinéarité) :")
    vars_num = [v for v in variables if v in df_w.columns and df_w[v].dtype in [float, int, 'float64','int64']]
    if len(vars_num) >= 2:
        X_vif = df_w[vars_num].dropna()
        X_vif = (X_vif - X_vif.mean()) / X_vif.std()
        X_vif.insert(0, 'const', 1)
        for i, v in enumerate(vars_num):
            vif = variance_inflation_factor(X_vif.values, i+1)
            alerte = " ⚠️ > 5" if vif > 5 else ""
            print(f"    {v:<30} VIF = {vif:.2f}{alerte}")

    # AUC / ROC
    y_true = df_w[col_outcome].values
    y_pred = model.predict()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = sk_auc(fpr, tpr)
    print(f"\n  AUC = {roc_auc:.3f}  {'✅ bon' if roc_auc>0.7 else '⚠️ faible'}")

    # Hosmer-Lemeshow
    df_hl    = pd.DataFrame({'pred': y_pred, 'obs': y_true})
    df_hl['decile'] = pd.qcut(df_hl['pred'], q=10, labels=False, duplicates='drop')
    hl_stat  = 0
    for dec in df_hl['decile'].unique():
        grp   = df_hl[df_hl['decile']==dec]
        O_k   = grp['obs'].sum()
        E_k   = grp['pred'].sum()
        N_k   = len(grp)
        pi_k  = grp['pred'].mean()
        denom = N_k * pi_k * (1 - pi_k)
        if denom > 0:
            hl_stat += (O_k - E_k)**2 / denom
    p_hl = 1 - stats.chi2.cdf(hl_stat, df=8)
    print(f"  Hosmer-Lemeshow : HL={hl_stat:.2f}, p={p_hl:.4f}"
          f"  {'✅ bon ajustement' if p_hl>0.05 else '⚠️ mauvais ajustement'}")

    # Courbe ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE_GROUPES.get(groupe, {}).get('fill', '#3498db'), linewidth=2,
            label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_xlabel('Taux de faux positifs', fontsize=11)
    ax.set_ylabel('Taux de vrais positifs', fontsize=11)
    ax.set_title(f'Courbe ROC — {label}{titre_g}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {'model': model, 'AUC': roc_auc, 'HL_p': p_hl,
            'variables': resultats_vars, 'AIC': model.aic}


def comparer_deux_modeles(df, col_outcome, vars_modele1, vars_modele2,
                           groupe=None, label1='Modèle 1', label2='Modèle 2'):
    """
    Compare deux modèles logistiques : ΔAIC + LRT.

    Exemples :
        comparer_deux_modeles(df_final, 'stade_avance',
            vars_modele1=['age_diagnostic','sexe'],
            vars_modele2=['age_diagnostic','sexe','PM25_moyenne'],
            label1='Base', label2='Base + PM2.5')

        comparer_deux_modeles(df_final, 'stade_avance',
            vars_modele1=['age_diagnostic','sexe','PM25_moyenne'],
            vars_modele2=['age_diagnostic','sexe','PM25_moyenne','PM25_pct_Q4'],
            groupe='NF', label1='Moyenne seule', label2='Moyenne + pics')
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("⚠️  statsmodels requis")
        return None

    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    toutes_vars = list(set(vars_modele1 + vars_modele2 + [col_outcome]))
    df_w        = df_w[[c for c in toutes_vars if c in df_w.columns]].dropna()

    def _fit(variables):
        formula = f"{col_outcome} ~ {' + '.join(variables)}"
        try:
            return smf.logit(formula, data=df_w).fit(disp=0)
        except Exception as e:
            print(f"  ⚠️  Erreur : {e}")
            return None

    m1, m2 = _fit(vars_modele1), _fit(vars_modele2)
    if m1 is None or m2 is None:
        return None

    aic1, aic2 = m1.aic, m2.aic
    delta_aic  = aic1 - aic2
    lrt_stat   = 2 * (m2.llf - m1.llf)
    df_diff    = len(vars_modele2) - len(vars_modele1)
    p_lrt      = 1 - stats.chi2.cdf(lrt_stat, df_diff) if df_diff > 0 else np.nan
    sig_lrt    = '***' if p_lrt<0.001 else '**' if p_lrt<0.01 else '*' if p_lrt<0.05 else 'ns'

    titre_g = f" | {NOMS_GROUPES.get(groupe,groupe)}" if groupe else ""
    print(f"\n{'─'*65}")
    print(f"COMPARAISON DE MODÈLES{titre_g}")
    print(f"{'─'*65}")
    print(f"  {label1:<35} : AIC = {aic1:.1f}")
    print(f"  {label2:<35} : AIC = {aic2:.1f}")
    print(f"  ΔAIC = {delta_aic:.1f}  "
          f"{'→ {label2} meilleur ✅' if delta_aic > 2 else '→ Pas daméliorarion significative'}")
    print(f"  LRT  = {lrt_stat:.2f}, p = {p_lrt:.4f} {sig_lrt}")

    return {'AIC1': aic1, 'AIC2': aic2, 'delta_AIC': delta_aic,
            'LRT': lrt_stat, 'p_LRT': p_lrt, 'modele1': m1, 'modele2': m2}


def tableau_comparaison_modeles(df, col_outcome, liste_modeles, groupe=None):
    """
    Compare plusieurs modèles d'un coup dans un tableau synthétique.

    Exemple :
        tableau_comparaison_modeles(df_final, 'stade_avance', [
            {'label': 'Base',             'vars': ['age_diagnostic','sexe']},
            {'label': 'Base + PM2.5',     'vars': ['age_diagnostic','sexe','PM25_moyenne']},
            {'label': 'Base + PM2.5+pics','vars': ['age_diagnostic','sexe','PM25_moyenne','PM25_pct_Q4']},
            {'label': 'Base + IPG',       'vars': ['age_diagnostic','sexe','IPG']},
        ])
    """
    try:
        import statsmodels.formula.api as smf
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("⚠️  statsmodels et scikit-learn requis")
        return None

    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()

    resultats = []
    print(f"\n{'─'*80}")
    print(f"  {'Modèle':<35} {'AIC':>8} {'AUC':>8} {'LRT vs base':>14} {'p-LRT':>10}")
    print(f"  {'─'*80}")

    modele_base = None
    for spec in liste_modeles:
        label = spec['label']
        vars_ = spec['vars']
        cols_need = list(set(vars_ + [col_outcome]))
        df_fit = df_w[[c for c in cols_need if c in df_w.columns]].dropna()

        formula = f"{col_outcome} ~ {' + '.join(vars_)}"
        try:
            m     = smf.logit(formula, data=df_fit).fit(disp=0)
            y_hat = m.predict()
            auc   = roc_auc_score(df_fit[col_outcome], y_hat)

            lrt_str = '—'
            p_str   = '—'
            if modele_base is not None and len(vars_) > len(liste_modeles[0]['vars']):
                lrt  = 2 * (m.llf - modele_base.llf)
                df_d = len(vars_) - len(liste_modeles[0]['vars'])
                p    = 1 - stats.chi2.cdf(lrt, df_d)
                sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
                lrt_str = f"{lrt:.1f}"
                p_str   = f"{p:.4f} {sig}"

            print(f"  {label:<35} {m.aic:>8.1f} {auc:>8.3f} {lrt_str:>14} {p_str:>10}")
            resultats.append({'label': label, 'AIC': m.aic, 'AUC': auc, 'model': m})

            if modele_base is None:
                modele_base = m

        except Exception as e:
            print(f"  {label:<35} ⚠️ Erreur : {e}")

    return resultats


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ██╗  ██╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ██║  ██║
# ██████╔╝██║     ██║   ██║██║         ███████║
# ██╔══██╗██║     ██║   ██║██║         ╚════██║
# ██████╔╝███████╗╚██████╔╝╚██████╗        ██║
# ÉTAPES 4-5 — DOSE-RÉPONSE ET PICS
# =============================================================================

def calculer_or_quartiles(df, col_quartile, col_outcome, groupe=None, verbose=True):
    """
    OR par quartile avec IC 95% (Q1 = référence).

    Exemples :
        calculer_or_quartiles(df_final, 'PM25_quartile', 'stade_avance')
        calculer_or_quartiles(df_final, 'NO2_quartile', 'stade_avance', groupe='NF')
    """
    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    df_w = df_w[[col_quartile, col_outcome]].dropna()

    ref   = df_w[df_w[col_quartile] == 1]
    a_ref = ref[col_outcome].sum()
    b_ref = len(ref) - a_ref

    if a_ref == 0 or b_ref == 0:
        print("⚠️  Pas assez de cas dans Q1 (référence)")
        return None

    resultats = {1: {'OR':1.0,'IC_inf':1.0,'IC_sup':1.0,'p':1.0,'ref':True}}

    if verbose:
        titre_g = f" | {NOMS_GROUPES.get(groupe,groupe)}" if groupe else " | Tous patients"
        print(f"\n  OR — {col_quartile} → {col_outcome}{titre_g}")
        print(f"  {'Q':<12} {'n cas/tot':>12} {'OR':>8} {'IC 95%':>22} {'p':>10}")
        print(f"  {'─'*66}")
        print(f"  Q1 (réf.)    {'—':>12} {'1.00':>8} {'(référence)':>22}")

    for q in [2, 3, 4]:
        d   = df_w[df_w[col_quartile] == q]
        a   = d[col_outcome].sum()
        b   = len(d) - a
        if a == 0 or b == 0:
            resultats[q] = {'OR':np.nan,'IC_inf':np.nan,'IC_sup':np.nan,'p':np.nan}
            continue
        OR     = (a * b_ref) / (b * a_ref)
        SE     = np.sqrt(1/a + 1/b + 1/a_ref + 1/b_ref)
        log_OR = np.log(OR)
        IC_inf = np.exp(log_OR - 1.96*SE)
        IC_sup = np.exp(log_OR + 1.96*SE)
        try:
            _, p, _, _ = chi2_contingency([[a,b],[a_ref,b_ref]])
        except:
            p = np.nan
        resultats[q] = {'OR':OR,'IC_inf':IC_inf,'IC_sup':IC_sup,
                        'p':p,'n_cas':int(a),'n_total':int(a+b)}
        if verbose:
            sig    = '*' if p<0.05 else ''
            ic_str = f"[{IC_inf:.2f} - {IC_sup:.2f}]"
            n_str  = f"{int(a)}/{int(a+b)}"
            print(f"  Q{q}           {n_str:>12} {OR:>8.2f} {ic_str:>22} {p:>9.4f}{sig}")

    return resultats


def test_tendance(df, col_quartile, col_outcome, groupe=None):
    """
    Test de tendance de Cochran-Armitage : gradient Q1→Q4 significatif ?

    Exemples :
        test_tendance(df_final, 'PM25_quartile', 'stade_avance')
        test_tendance(df_final, 'NO2_quartile', 'stade_avance', groupe='NF')
    """
    df_w     = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    df_w     = df_w[[col_quartile, col_outcome]].dropna()
    scores   = df_w[col_quartile].values
    outcomes = df_w[col_outcome].values
    n, n1    = len(scores), outcomes.sum()
    n0       = n - n1

    if n0 == 0 or n1 == 0 or n < 20:
        return {'Z':np.nan,'p':np.nan,'interpretation':'Effectif insuffisant'}

    x_bar  = np.mean(scores)
    T      = np.sum(scores * outcomes)
    Var_T  = (n0 * n1 * np.var(scores, ddof=0) * n) / (n * (n - 1))

    if Var_T <= 0:
        return {'Z':np.nan,'p':np.nan,'interpretation':'Variance nulle'}

    Z      = (T - np.sum(outcomes)*x_bar) / np.sqrt(Var_T)
    p      = 2 * (1 - norm.cdf(abs(Z)))
    sig    = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    direct = "↑ risque croissant" if Z > 0 else "↓ risque décroissant"

    titre_g = f" | {NOMS_GROUPES.get(groupe,groupe)}" if groupe else ""
    print(f"\n  Tendance — {col_quartile} → {col_outcome}{titre_g}")
    print(f"  Z = {Z:.3f}, p = {p:.4f} {sig} | {direct}")

    return {'Z':Z,'p':p,'sig':sig,'direction':direct,'n':n,'n_cas':int(n1)}


def forest_plot(resultats_or_dict, titre='Dose-réponse par quartile',
                tendances_dict=None, save_path=None):
    """
    Forest plot des OR par quartile pour un ou plusieurs groupes.

    Exemples :
        # Global
        or_g = calculer_or_quartiles(df_final, 'PM25_quartile', 'stade_avance')
        forest_plot({'Tous': or_g}, 'PM2.5 → Stade avancé')

        # Par groupe
        ors = {g: calculer_or_quartiles(df_final,'PM25_quartile','stade_avance',
                                         groupe=g,verbose=False) for g in GROUPES}
        tds = {g: test_tendance(df_final,'PM25_quartile','stade_avance',groupe=g)
               for g in GROUPES}
        forest_plot(ors, 'PM2.5 → Stade avancé', tendances_dict=tds)
    """
    fig, ax = plt.subplots(figsize=(11, max(5, len(resultats_or_dict)*2.5)))
    y_current, y_ticks, y_labels = 0, [], []

    for label, or_dict in resultats_or_dict.items():
        if or_dict is None:
            continue
        couleur_label = PALETTE_GROUPES.get(label,'#333333')
        ax.text(0.02, y_current+1.8, label, fontsize=11, fontweight='bold',
                color=couleur_label, transform=ax.get_yaxis_transform())

        for q in [1,2,3,4]:
            y     = y_current + (4-q)
            r     = or_dict.get(q,{})
            color = PALETTE_QUARTILES[q]['fill']
            y_ticks.append(y)
            y_labels.append(f"  Q{q}")

            if r.get('ref',False) or q == 1:
                ax.scatter(1.0, y, marker='s', s=100,
                           color=color, edgecolor='black', linewidth=1.5, zorder=5)
                ax.text(1.05, y, '1.00 (réf.)', va='center', fontsize=9, color='gray')
            else:
                or_v  = r.get('OR',np.nan)
                ic_i  = r.get('IC_inf',np.nan)
                ic_s  = r.get('IC_sup',np.nan)
                p_v   = r.get('p',np.nan)
                if not np.isnan(or_v):
                    ax.errorbar(or_v, y, xerr=[[or_v-ic_i],[ic_s-or_v]],
                                fmt='o', markersize=9, color=color,
                                ecolor='black', capsize=4, capthick=1.5, zorder=5)
                    sig   = '*' if p_v<0.05 else ''
                    ax.text(ic_s+0.05, y,
                            f'{or_v:.2f} [{ic_i:.2f}-{ic_s:.2f}]{sig}',
                            va='center', fontsize=9)

        if tendances_dict and label in tendances_dict:
            t   = tendances_dict[label]
            p_t = t.get('p', np.nan)
            sig = ('***' if p_t<0.001 else '**' if p_t<0.01
                   else '*' if p_t<0.05 else 'ns')
            ax.text(0.02, y_current+0.2,
                    f'p-trend = {p_t:.4f} {sig}',
                    fontsize=9, color='gray', style='italic',
                    transform=ax.get_yaxis_transform())
        y_current += 6

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Odds Ratio [IC 95%]', fontsize=11)
    ax.set_title(titre, fontsize=13, fontweight='bold')
    ax.set_xlim(left=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_dose_reponse(df, col_quartile, col_outcome, groupe=None, save_path=None):
    """
    Graphique dose-réponse : OR [IC95%] par quartile avec ligne de tendance.

    Exemples :
        plot_dose_reponse(df_final, 'PM25_quartile', 'stade_avance')
        plot_dose_reponse(df_final, 'PM25_quartile', 'stade_avance', groupe='NF')
    """
    or_dict  = calculer_or_quartiles(df, col_quartile, col_outcome, groupe, verbose=False)
    tendance = test_tendance(df, col_quartile, col_outcome, groupe)

    if or_dict is None:
        return None

    fig, ax  = plt.subplots(figsize=(8, 6))
    qs       = [1,2,3,4]
    ors      = [or_dict[q].get('OR',np.nan)     for q in qs]
    ic_inf   = [or_dict[q].get('IC_inf',np.nan) for q in qs]
    ic_sup   = [or_dict[q].get('IC_sup',np.nan) for q in qs]
    colors   = [PALETTE_QUARTILES[q]['fill']     for q in qs]

    for i, q in enumerate(qs):
        if not np.isnan(ors[i]):
            ax.errorbar(q, ors[i],
                        yerr=[[ors[i]-ic_inf[i]],[ic_sup[i]-ors[i]]],
                        fmt='o', markersize=12, color=colors[i],
                        ecolor='black', capsize=5, capthick=2,
                        elinewidth=2, zorder=5)

    # Ligne de tendance
    ors_valides = [(q, o) for q, o in zip(qs, ors) if not np.isnan(o)]
    if len(ors_valides) >= 2:
        x_t, y_t = zip(*ors_valides)
        z        = np.polyfit(x_t, y_t, 1)
        x_line   = np.linspace(0.8, 4.2, 50)
        ax.plot(x_line, np.poly1d(z)(x_line), 'k--', linewidth=1.5, alpha=0.5)

    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.2)
    ax.set_xticks(qs)
    ax.set_xticklabels([PALETTE_QUARTILES[q]['label'] for q in qs], fontsize=10)
    ax.set_xlabel('Quartile d\'exposition', fontsize=12)
    ax.set_ylabel('Odds Ratio [IC 95%]', fontsize=12)

    p_t   = tendance.get('p', np.nan)
    sig_t = '***' if p_t<0.001 else '**' if p_t<0.01 else '*' if p_t<0.05 else 'ns'
    titre_g = f" | {NOMS_GROUPES.get(groupe,groupe)}" if groupe else ""
    ax.set_title(f'Dose-réponse : {col_quartile}{titre_g}\n'
                 f'p-trend = {p_t:.4f} {sig_t}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ██████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ██╔════╝
# ██████╔╝██║     ██║   ██║██║         ███████╗
# ██╔══██╗██║     ██║   ██║██║         ██╔═══██╗
# ██████╔╝███████╗╚██████╔╝╚██████╗    ╚██████╔╝
# ÉTAPE 6 — FENÊTRES TEMPORELLES
# =============================================================================

def exposure_lag_plot(df, polluant, col_outcome, groupe=None, save_path=None):
    """
    Exposure-lag plot : OR par fenêtre temporelle sur l'axe X.
    Identifie la période critique d'exposition (H5).

    Exemples :
        exposure_lag_plot(df_final, 'PM25', 'stade_avance')
        exposure_lag_plot(df_final, 'NO2', 'stade_avance', groupe='NF')
    """
    fenetres = {
        '0_6m':   '0-6 mois',
        '6_12m':  '6-12 mois',
        '12_18m': '12-18 mois',
        '18_24m': '18-24 mois',
    }

    df_w    = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    titre_g = f" | {NOMS_GROUPES.get(groupe,groupe)}" if groupe else " | Tous patients"

    resultats_fen = {}
    print(f"\n  Exposure-lag — {polluant} → {col_outcome}{titre_g}")
    print(f"  {'Fenêtre':<15} {'OR':>8} {'IC 95%':>22} {'p':>10}")
    print(f"  {'─'*58}")

    for fen_key, fen_label in fenetres.items():
        col_fen = f'{polluant}_moy_{fen_key}'
        if col_fen not in df_w.columns:
            continue

        df_f    = df_w[[col_fen, col_outcome]].dropna()
        if len(df_f) < 20:
            continue

        # Quartiliser la fenêtre
        try:
            df_f['q_fen'] = pd.qcut(df_f[col_fen], q=4, labels=[1,2,3,4])
            or_f = calculer_or_quartiles(df_f, 'q_fen', col_outcome, verbose=False)
            or_q4 = or_f.get(4, {})
            OR    = or_q4.get('OR', np.nan)
            IC_i  = or_q4.get('IC_inf', np.nan)
            IC_s  = or_q4.get('IC_sup', np.nan)
            p_v   = or_q4.get('p', np.nan)
            resultats_fen[fen_label] = {'OR': OR, 'IC_inf': IC_i, 'IC_sup': IC_s, 'p': p_v}
            sig   = '*' if p_v < 0.05 else ''
            ic_str = f"[{IC_i:.2f}-{IC_s:.2f}]" if not np.isnan(OR) else "N/A"
            print(f"  {fen_label:<15} {OR:>8.2f} {ic_str:>22} {p_v:>9.4f}{sig}")
        except Exception:
            continue

    if not resultats_fen:
        print("  ⚠️  Aucune fenêtre temporelle disponible")
        return None

    # Plot
    fig, ax  = plt.subplots(figsize=(10, 6))
    labels   = list(resultats_fen.keys())
    ors_list = [resultats_fen[l]['OR']     for l in labels]
    ic_i_l   = [resultats_fen[l]['IC_inf'] for l in labels]
    ic_s_l   = [resultats_fen[l]['IC_sup'] for l in labels]
    ps       = [resultats_fen[l]['p']      for l in labels]
    x        = range(len(labels))

    ax.errorbar(x, ors_list,
                yerr=[[o-i for o,i in zip(ors_list,ic_i_l)],
                      [s-o for o,s in zip(ors_list,ic_s_l)]],
                fmt='o-', markersize=10,
                color=PALETTE_GROUPES.get(groupe,'#3498db'),
                ecolor='black', capsize=5, capthick=2,
                elinewidth=2, linewidth=2)

    for xi, (o, p_v) in enumerate(zip(ors_list, ps)):
        if p_v < 0.05:
            ax.text(xi, o + 0.05, '*', ha='center', fontsize=14, color='red')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Fenêtre temporelle avant diagnostic →', fontsize=12)
    ax.set_ylabel('OR Q4 vs Q1 [IC 95%]', fontsize=12)
    ax.set_title(f'Exposure-lag : {polluant}{titre_g}\n(OR du Q4 vs Q1 par fenêtre)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return resultats_fen


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     ███████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ╚═══ ═██╗
# ██████╔╝██║     ██║   ██║██║             ███╔╝
# ██╔══██╗██║     ██║   ██║██║            ███╔╝
# ██████╔╝███████╗╚██████╔╝╚██████╗     ███████╗
# ÉTAPE 7 — BIOMARQUEURS / MUTATIONS
# =============================================================================

def regression_mutation(df, mutation, variables, groupe='NF', save_path=None):
    """
    Régression logistique : mutation ~ polluants + covariables.
    Usage principal : tester H6 (EGFR/ALK liés à PM2.5 chez NF).

    Exemples :
        regression_mutation(df_final, 'EGFR',
            ['PM25_moyenne', 'age_diagnostic', 'sexe'])
        regression_mutation(df_final, 'EGFR',
            ['PM25_moyenne', 'age_diagnostic', 'sexe'], groupe='NF')
        regression_mutation(df_final, 'KRAS',
            ['PM25_moyenne', 'age_diagnostic', 'sexe'])  # contrôle négatif
    """
    col_mut = f'mutation_{mutation}'
    if col_mut not in df.columns:
        print(f"⚠️  Colonne {col_mut} non trouvée")
        return None

    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    df_w[col_mut] = (df_w[col_mut].astype(str).str.strip().str.lower() == 'positive').astype(int)

    print(f"\n  Régression : mutation {mutation} ~ exposition"
          f" | {NOMS_GROUPES.get(groupe, 'Tous')}")

    return regression_logistique(df_w, col_mut, variables,
                                  label=f'Mutation {mutation}',
                                  save_path=save_path)


def tableau_or_mutations(df, mutations, col_polluant, groupe='NF', covariables=None):
    """
    Tableau synthétique des OR de chaque mutation pour un polluant donné.

    Exemple :
        tableau_or_mutations(df_final,
            mutations=['EGFR','ALK','ROS1','KRAS','TP53'],
            col_polluant='PM25_moyenne',
            groupe='NF',
            covariables=['age_diagnostic','sexe'])
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("⚠️  statsmodels requis")
        return None

    if covariables is None:
        covariables = []

    df_w = df[df['groupe'] == groupe].copy() if groupe else df.copy()
    vars_ = [col_polluant] + covariables

    print(f"\n{'─'*70}")
    print(f"OR {col_polluant} → Mutations | {NOMS_GROUPES.get(groupe,'Tous')}")
    print(f"{'─'*70}")
    print(f"  {'Mutation':<12} {'OR':>8} {'IC 95%':>22} {'p':>10} {'Interp.':>20}")
    print(f"  {'─'*70}")

    rows = []
    for mut in mutations:
        col_mut = f'mutation_{mut}'
        if col_mut not in df_w.columns:
            continue
        df_m = df_w.copy()
        df_m[col_mut] = (df_m[col_mut].astype(str).str.lower() == 'positive').astype(int)
        cols_need = [col_mut] + vars_
        df_m = df_m[[c for c in cols_need if c in df_m.columns]].dropna()

        if len(df_m) < 20 or df_m[col_mut].sum() < 5:
            print(f"  {mut:<12} (effectif insuffisant)")
            continue

        formula = f"{col_mut} ~ {' + '.join(vars_)}"
        try:
            model  = smf.logit(formula, data=df_m).fit(disp=0)
            OR     = np.exp(model.params[col_polluant])
            ci     = model.conf_int().loc[col_polluant]
            p_val  = model.pvalues[col_polluant]
            IC_inf = np.exp(ci[0])
            IC_sup = np.exp(ci[1])
            sig    = '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'
            interp = "↑ exposition→mutation" if OR > 1 else "↓ exposition→mutation"
            ic_str = f"[{IC_inf:.2f}-{IC_sup:.2f}]"
            print(f"  {mut:<12} {OR:>8.3f} {ic_str:>22} {p_val:>9.4f} {sig}  {interp}")
            rows.append({'mutation':mut,'OR':OR,'IC_inf':IC_inf,'IC_sup':IC_sup,
                         'p':p_val,'sig':sig})
        except Exception as e:
            print(f"  {mut:<12} ⚠️  {e}")

    return pd.DataFrame(rows)


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗     █████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ██╔══██╗
# ██████╔╝██║     ██║   ██║██║         ╚█████╔╝
# ██╔══██╗██║     ██║   ██║██║         ██╔══██╗
# ██████╔╝███████╗╚██████╔╝╚██████╗    ╚█████╔╝
# ÉTAPE 8 — CLUSTERING / ACP
# =============================================================================

def acp_groupes(df, cols, n_composantes=2, save_path=None):
    """
    ACP + scatter PC1/PC2 coloré par groupe.

    Exemple :
        acp_groupes(df_final,
            ['PM25_moyenne','NO2_moyenne','O3_moyenne','PM25_cv','PM25_pct_Q4','IPG'])
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        print("⚠️  scikit-learn requis")
        return None

    cols_ok = [c for c in cols if c in df.columns]
    df_w    = df[df['groupe'].isin(GROUPES_ANALYSE)][cols_ok + ['groupe']].dropna()

    X       = StandardScaler().fit_transform(df_w[cols_ok])
    pca     = PCA(n_components=min(n_composantes, len(cols_ok)))
    X_pca   = pca.fit_transform(X)

    var_expl = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    for g in GROUPES_ANALYSE:
        mask = df_w['groupe'].values == g
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=PALETTE_GROUPES[g]['fill'], alpha=0.4, s=20, label=NOMS_GROUPES[g])
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, color=PALETTE_GROUPES[g], s=200, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_xlabel(f'PC1 ({var_expl[0]:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_expl[1]:.1f}% variance)', fontsize=12)
    ax.set_title('ACP — Espace d\'exposition par groupe\n(★ = centroïde)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Variance expliquée : PC1={var_expl[0]:.1f}%, PC2={var_expl[1]:.1f}%")
    print(f"  Total PC1+PC2     : {sum(var_expl[:2]):.1f}%")

    return {'pca': pca, 'X_pca': X_pca, 'var_expl': var_expl, 'df_w': df_w}


def kmeans_clustering(df, cols, k_max=6, save_path=None):
    """
    K-means : détermine k optimal (coude + silhouette) puis clusterise.

    Exemple :
        kmeans_clustering(df_final,
            ['PM25_moyenne','NO2_moyenne','O3_moyenne','PM25_cv'])
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("⚠️  scikit-learn requis")
        return None

    cols_ok = [c for c in cols if c in df.columns]
    df_w    = df[df['groupe'].isin(GROUPES_ANALYSE)][cols_ok + ['groupe']].dropna()
    X       = StandardScaler().fit_transform(df_w[cols_ok])

    # Détermination k optimal
    inertias, silhouettes = [], []
    k_range = range(2, min(k_max+1, len(df_w)//10))

    for k in k_range:
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, 'bo-', markersize=8)
    axes[0].set_xlabel('Nombre de clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertie intra-cluster', fontsize=11)
    axes[0].set_title('Méthode du coude', fontsize=12, fontweight='bold')

    k_optimal = list(k_range)[np.argmax(silhouettes)]
    axes[1].plot(list(k_range), silhouettes, 'rs-', markersize=8)
    axes[1].axvline(k_optimal, color='green', linestyle='--', linewidth=2,
                    label=f'k optimal = {k_optimal}')
    axes[1].set_xlabel('Nombre de clusters (k)', fontsize=11)
    axes[1].set_ylabel('Score silhouette', fontsize=11)
    axes[1].set_title(f'Score silhouette (k optimal = {k_optimal})',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png','_selection_k.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  k optimal (silhouette max) = {k_optimal}")
    print(f"  Score silhouette           = {max(silhouettes):.3f}"
          f"  {'✅ > 0.5' if max(silhouettes) > 0.5 else '⚠️ < 0.5'}")

    # Clustering final
    km_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df_w     = df_w.copy()
    df_w['cluster'] = km_final.fit_predict(X)

    # Test Chi² cluster × groupe
    contingence = pd.crosstab(df_w['cluster'], df_w['groupe'])
    chi2, p, _, _ = chi2_contingency(contingence)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  Chi² cluster × groupe : χ²={chi2:.1f}, p={p:.4f} {sig}"
          f"  {'✅ clusters ≠ groupes' if p<0.05 else '❌ clusters = groupes'}")

    print(f"\n  Distribution cluster × groupe :")
    print(contingence.to_string())

    return {'clusters': df_w['cluster'], 'k_optimal': k_optimal,
            'silhouette': max(silhouettes), 'chi2_p': p}


# =============================================================================
# ██████╗ ██╗      ██████╗  ██████╗      █████╗
# ██╔══██╗██║     ██╔═══██╗██╔════╝    ██╔══██╗
# ██████╔╝██║     ██║   ██║██║          ╚═╗███║
# ██╔══██╗██║     ██║   ██║██║         ██╔╝███║
# ██████╔╝███████╗╚██████╔╝╚██████╗    ╚██████╔╝
# ÉTAPE 9 — SENSIBILITÉ
# =============================================================================

def tableau_sensibilite(df, col_outcome, variables_principales,
                         configs_sensibilite, groupe=None):
    """
    Tableau comparatif des OR principaux vs toutes les analyses de sensibilité.

    Exemple :
        tableau_sensibilite(df_final, 'stade_avance',
            variables_principales=['PM25_moyenne','age_diagnostic','sexe'],
            configs_sensibilite=[
                {'label': 'Fenêtre 12 mois', 'vars': ['PM25_moy_0_12m','age_diagnostic','sexe']},
                {'label': 'Fenêtre 6 mois',  'vars': ['PM25_moy_0_6m','age_diagnostic','sexe']},
                {'label': 'Seuils OMS',      'vars': ['PM25_nb_sup_5_regl','age_diagnostic','sexe']},
                {'label': 'Sans anc. fumeurs','vars': ['PM25_moyenne','age_diagnostic','sexe'],
                 'filtre': lambda df: df[df['paquets_annees']==0]},
            ])
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("⚠️  statsmodels requis")
        return None

    def _or_principal(df_, vars_, outcome, g):
        df_g = df_[df_['groupe']==g].copy() if g else df_.copy()
        cols = list(set(vars_ + [outcome]))
        df_g = df_g[[c for c in cols if c in df_g.columns]].dropna()
        formula = f"{outcome} ~ {' + '.join(vars_)}"
        try:
            m      = smf.logit(formula, data=df_g).fit(disp=0)
            var_pm = [v for v in vars_ if 'PM25' in v or 'NO2' in v or 'IPG' in v]
            if not var_pm:
                return np.nan, np.nan, np.nan, np.nan
            vp     = var_pm[0]
            OR     = np.exp(m.params[vp])
            ci     = m.conf_int().loc[vp]
            p_val  = m.pvalues[vp]
            return OR, np.exp(ci[0]), np.exp(ci[1]), p_val
        except:
            return np.nan, np.nan, np.nan, np.nan

    print(f"\n{'═'*80}")
    print(f"TABLEAU DE SENSIBILITÉ — {col_outcome}")
    print(f"{'═'*80}")
    print(f"  {'Analyse':<30} {'OR':>8} {'IC 95%':>22} {'p':>10} {'ΔOR vs principal':>18}")
    print(f"  {'─'*80}")

    OR_ref, ic_i_ref, ic_s_ref, p_ref = _or_principal(df, variables_principales,
                                                        col_outcome, groupe)
    sig_ref = '***' if p_ref<0.001 else '**' if p_ref<0.01 else '*' if p_ref<0.05 else 'ns'
    ic_str  = f"[{ic_i_ref:.2f}-{ic_s_ref:.2f}]"
    print(f"  {'Analyse principale':<30} {OR_ref:>8.3f} {ic_str:>22} {p_ref:>9.4f} {sig_ref}  {'(référence)':>18}")

    rows = [{'analyse': 'Principale', 'OR': OR_ref, 'IC_inf': ic_i_ref,
             'IC_sup': ic_s_ref, 'p': p_ref, 'delta_OR': 0}]

    for config in configs_sensibilite:
        label  = config['label']
        vars_s = config.get('vars', variables_principales)
        filtre = config.get('filtre', None)
        df_s   = filtre(df) if filtre else df

        OR_s, ic_i_s, ic_s_s, p_s = _or_principal(df_s, vars_s, col_outcome, groupe)
        delta  = abs(OR_s - OR_ref) / OR_ref * 100 if not np.isnan(OR_s) and OR_ref != 0 else np.nan
        sig_s  = '***' if p_s<0.001 else '**' if p_s<0.01 else '*' if p_s<0.05 else 'ns'
        robuste= "✅" if not np.isnan(delta) and delta < 15 else "⚠️"
        ic_str = f"[{ic_i_s:.2f}-{ic_s_s:.2f}]" if not np.isnan(OR_s) else "N/A"
        delta_str = f"{delta:.1f}% {robuste}" if not np.isnan(delta) else "N/A"

        print(f"  {label:<30} {OR_s:>8.3f} {ic_str:>22} {p_s:>9.4f} {sig_s}  {delta_str:>18}")
        rows.append({'analyse': label, 'OR': OR_s, 'IC_inf': ic_i_s,
                     'IC_sup': ic_s_s, 'p': p_s, 'delta_OR_pct': delta})

    print(f"\n  Règle : ΔOR < 15% → résultats robustes ✅")
    return pd.DataFrame(rows)


# =============================================================================
# CATALOGUE — aide()
# =============================================================================

def aide():
    """Affiche le catalogue complet de toutes les fonctions."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BIBLIOTHÈQUE D'ANALYSE COMPLÈTE — CATALOGUE DES FONCTIONS          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ── ÉTAPE 1 : DESCRIPTION ─────────────────────────────────────────────── ║
║  analyse_correlation(df, col1, col2)                                         ║
║    → Spearman 2 variables + scatter | option: groupe='NF'                    ║
║  analyse_correlation_multiple(df, [cols])                                    ║
║    → Matrice Spearman heatmap + alertes multicolinéarité                     ║
║  comparer_groupes(df, 'col')                                                 ║
║    → KW + MW 2à2 + Cohen d + violin | option: groupes=['NF','F-SF']          ║
║  test_normalite(df, 'col')                                                   ║
║    → Shapiro-Wilk par groupe                                                 ║
║  tableau_descriptif(df, [cols])                                              ║
║    → Médiane [IQR] + KW pour liste de colonnes                               ║
║  heatmap_mutations_polluants(df)                                             ║
║    → Heatmap corrélation mutations × polluants par groupe                    ║
║                                                                              ║
║  ── ÉTAPE 2 : COMPARAISONS H1/H2 ──────────────────────────────────────── ║
║  ancova_groupes(df, 'col_Y', 'covariable', groupes_cibles=['F-SNF','F-SF']) ║
║    → ANCOVA : compare groupes en contrôlant paquets-années                  ║
║  barplot_comparatif(df, [cols])                                              ║
║    → Barplot moyennes ± SEM par groupe                                       ║
║                                                                              ║
║  ── ÉTAPE 3 : RÉGRESSION LOGISTIQUE ───────────────────────────────────── ║
║  regression_logistique(df, 'outcome', [vars])                                ║
║    → OR + IC + p + VIF + AUC + Hosmer-Lemeshow | option: groupe='NF'        ║
║  comparer_deux_modeles(df, 'outcome', vars1, vars2)                          ║
║    → ΔAIC + LRT entre 2 modèles | option: groupe='NF'                       ║
║  tableau_comparaison_modeles(df, 'outcome', [specs])                         ║
║    → Tableau AIC + AUC + LRT pour N modèles simultanément                   ║
║                                                                              ║
║  ── ÉTAPES 4-5 : DOSE-RÉPONSE ET PICS ────────────────────────────────── ║
║  calculer_or_quartiles(df, 'col_quartile', 'outcome')                        ║
║    → OR + IC 95% pour Q2, Q3, Q4 vs Q1 | option: groupe='NF'                ║
║  test_tendance(df, 'col_quartile', 'outcome')                                ║
║    → Cochran-Armitage : gradient Q1→Q4 | option: groupe='NF'                ║
║  forest_plot(dict_ors, titre)                                                ║
║    → Forest plot OR par quartile (1 ou plusieurs groupes)                    ║
║  plot_dose_reponse(df, 'col_quartile', 'outcome')                            ║
║    → Graphique OR avec IC et ligne de tendance | option: groupe='NF'         ║
║                                                                              ║
║  ── ÉTAPE 6 : FENÊTRES TEMPORELLES ────────────────────────────────────── ║
║  exposure_lag_plot(df, 'PM25', 'outcome')                                    ║
║    → OR Q4vs Q1 par fenêtre temporelle | option: groupe='NF'                 ║
║                                                                              ║
║  ── ÉTAPE 7 : MUTATIONS / BIOMARQUEURS ────────────────────────────────── ║
║  regression_mutation(df, 'EGFR', [vars], groupe='NF')                        ║
║    → Régression logistique mutation ~ polluants                               ║
║  tableau_or_mutations(df, [mutations], 'PM25_moyenne', groupe='NF')          ║
║    → Tableau OR de chaque mutation pour un polluant                          ║
║                                                                              ║
║  ── ÉTAPE 8 : CLUSTERING / ACP ────────────────────────────────────────── ║
║  acp_groupes(df, [cols])                                                     ║
║    → ACP + scatter PC1/PC2 coloré par groupe                                 ║
║  kmeans_clustering(df, [cols])                                               ║
║    → K-means : sélection k + silhouette + Chi² clusters×groupes              ║
║                                                                              ║
║  ── ÉTAPE 9 : SENSIBILITÉ ─────────────────────────────────────────────── ║
║  tableau_sensibilite(df, 'outcome', vars_princ, [configs])                   ║
║    → ΔOR vs principale pour chaque analyse de sensibilité                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)