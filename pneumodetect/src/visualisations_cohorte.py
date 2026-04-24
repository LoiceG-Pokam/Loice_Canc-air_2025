#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANALYSE COHORTE CANCER DU POUMON - VERSION FINALE SIMPLIFIÉE
================================================================================
Question: La qualité de l'air cause-t-elle le cancer chez les non-fumeurs ?
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# COULEURS SIMPLES ET CLAIRES
# ============================================================================

COULEURS = {
    'homme': '#4A90E2',      # Bleu clair
    'femme': '#E85D75',      # Rose
    'non_dispo': '#95A5A6',  # Gris
    'fumeur': '#E74C3C',     # Rouge
    'non_fumeur': '#27AE60', # Vert
    'mutation_pos': '#E74C3C',
    'mutation_neg': '#27AE60',
    'gris': '#95A5A6'
}

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================

def preparer_donnees(df):
    """Prépare TOUTES les variables nécessaires"""
    print("\n" + "="*70)
    print("🔧 PRÉPARATION DES DONNÉES")
    print("="*70)
    
    df = df.copy()
    
    # 1. SEXE - AVEC NONE
    df['sexe_clean'] = df['sexe'].apply(
        lambda x: 'Homme' if str(x).lower() in ['masculin', 'male', 'm', 'h', 'homme']
        else 'Femme' if str(x).lower() in ['feminin', 'female', 'f', 'femme']
        else 'Non disponible'
    )
    nb_h = (df['sexe_clean']=='Homme').sum()
    nb_f = (df['sexe_clean']=='Femme').sum()
    nb_nd = (df['sexe_clean']=='Non disponible').sum()
    print(f"✓ Sexe: {nb_h} hommes, {nb_f} femmes, {nb_nd} non disponibles")
    
    # 2. ÂGE - CATÉGORIES
    df['age_categorie'] = pd.cut(
        df['age_diagnostic'],
        bins=[0, 40, 50, 60, 70, 80, 120],
        labels=['<40', '40-50', '50-60', '60-70', '70-80', '>80']
    )
    print(f"✓ Âge: {df['age_diagnostic'].notna().sum()} valeurs disponibles")
    
    # 3. TABAGISME
    df['fumeur'] = df['statut_tabagique'].apply(
        lambda x: 'Non-fumeur' if str(x).lower() == 'non fumeur' else 'Fumeur/Ex-fumeur'
    )
    nb_nf = (df['fumeur'] == 'Non-fumeur').sum()
    print(f"✓ Tabac: {nb_nf} NON-FUMEURS ({nb_nf/len(df)*100:.1f}%) 🔑")
    
    # 4. STADES SIMPLIFIÉS
    def simplifier_stade(s):
        if pd.isna(s):
            return 'Non disponible'
        s = str(s).upper().replace('-','').replace(' ','')
        
        if 'OLD' in s or 'NON DISPONIBLE' in s:
            return 'Non disponible'
        elif s.startswith('IV'):
            return 'IV'
        elif s.startswith('III'):
            return 'III'
        elif s.startswith('II'):
            return 'II'
        elif s.startswith('I'):
            return 'I'
        else:
            return 'Non disponible'
    
    df['stade'] = df['stade'].apply(simplifier_stade)
    print(f"✓ Stades simplifiés: I, II, III, IV, Inconnu")
    
    # 5. MUTATIONS - SIMPLIFIÉ
    mutations = ['mutation_EGFR', 'mutation_KRAS', 'mutation_ALK', 'mutation_BRAF']
    
    for mut in mutations:
        if mut in df.columns:
            df[f'{mut}_status'] = df[mut].apply(
                lambda x: 'Positive' if str(x).lower() == 'positive' else
                         'Négative' if str(x).lower() == 'negative' else
                         'Non fait'
            )
    
    # Compter mutations positives
    df['nb_mut_positives'] = 0
    for mut in mutations:
        if f'{mut}_status' in df.columns:
            df['nb_mut_positives'] += (df[f'{mut}_status'] == 'Positive').astype(int)
    
    nb_mutes = (df['nb_mut_positives'] > 0).sum()
    print(f"✓ Mutations: {nb_mutes} patients avec mutation(s) positive(s)")
    
    print("="*70)
    print(f"✅ {len(df)} patients prêts pour analyse\n")
    
    return df


# ============================================================================
# STATISTIQUES DESCRIPTIVES
# ============================================================================

def stats_descriptives(df):
    """Affiche les statistiques principales"""
    print("\n" + "="*70)
    print("📊 STATISTIQUES DESCRIPTIVES")
    print("="*70 + "\n")
    
    print(f"EFFECTIF: {len(df)} patients\n")
    
    # Sexe
    print("SEXE:")
    for sexe in ['Homme', 'Femme', 'Non disponible']:
        n = (df['sexe_clean'] == sexe).sum()
        print(f"  {sexe}: {n} ({n/len(df)*100:.1f}%)")
    
    # Âge
    print(f"\nÂGE:")
    print(f"  Moyenne: {df['age_diagnostic'].mean():.1f} ans")
    print(f"  Médiane: {df['age_diagnostic'].median():.1f} ans")
    print(f"  Min-Max: {df['age_diagnostic'].min():.0f}-{df['age_diagnostic'].max():.0f} ans")
    
    # Tabagisme - IMPORTANT
    print(f"\n🔑 TABAGISME:")
    for statut in df['fumeur'].unique():
        n = (df['fumeur'] == statut).sum()
        print(f"  {statut}: {n} ({n/len(df)*100:.1f}%)")
    
    # Stades
    print(f"\nSTADES:")
    for stade in ['I', 'II', 'III', 'IV', 'Inconnu']:
        n = (df['stade'] == stade).sum()
        if n > 0:
            print(f"  {stade}: {n} ({n/len(df)*100:.1f}%)")
    
    # Mutations
    print(f"\nMUTATIONS:")
    mutes = (df['nb_mut_positives'] > 0).sum()
    print(f"  Patients avec mutation(s): {mutes} ({mutes/len(df)*100:.1f}%)")
    print(f"  Patients sans mutation: {len(df)-mutes} ({(len(df)-mutes)/len(df)*100:.1f}%)")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# GRAPHIQUES
# ============================================================================

def graph_1_sexe(df):
    """Graphique 1: Répartition H/F avec Non disponible"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordre spécifique
    ordre = ['Homme', 'Femme', 'Non disponible']
    counts = df['sexe_clean'].value_counts().reindex(ordre, fill_value=0)
    colors = [COULEURS['homme'], COULEURS['femme'], COULEURS['non_dispo']]
    
    bars = ax.bar(counts.index, counts.values, color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.8)
    
    # Valeurs
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_title('Répartition par Sexe', fontsize=15, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1_sexe.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 1 sauvegardé: 1_sexe.png")


def graph_2_pyramide_ages(df):
    """Graphique 2: Pyramide des âges - VERSION QUI MARCHE"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # IMPORTANT: Filtrer les NaN ET les 'Non disponible'
    df_valid = df[(df['age_categorie'].notna()) & 
                  (df['sexe_clean'].isin(['Homme', 'Femme']))].copy()
    
    # Compter
    hommes = df_valid[df_valid['sexe_clean']=='Homme']['age_categorie'].value_counts()
    femmes = df_valid[df_valid['sexe_clean']=='Femme']['age_categorie'].value_counts()
    
    # Réindexer
    categories = ['<40', '40-50', '50-60', '60-70', '70-80', '>80']
    hommes = hommes.reindex(categories, fill_value=0)
    femmes = femmes.reindex(categories, fill_value=0)
    
    print(f"  Hommes: {hommes.sum()}, Femmes: {femmes.sum()}")
    
    y = np.arange(len(categories))
    
    # Barres
    ax.barh(y, -hommes.values, color=COULEURS['homme'], alpha=0.8, 
            label='Hommes', edgecolor='white', linewidth=1.5)
    ax.barh(y, femmes.values, color=COULEURS['femme'], alpha=0.8,
            label='Femmes', edgecolor='white', linewidth=1.5)
    
    # Valeurs
    max_val = max(hommes.max(), femmes.max())
    for i, (h, f) in enumerate(zip(hommes.values, femmes.values)):
        if h > 0:
            ax.text(-h-max_val*0.02, i, str(int(h)), va='center', ha='right', fontsize=11)
        if f > 0:
            ax.text(f+max_val*0.02, i, str(int(f)), va='center', ha='left', fontsize=11)
    
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_ylabel('Âge au diagnostic', fontsize=13, fontweight='bold')
    ax.set_title('Pyramide des Âges (Hommes et Femmes)', fontsize=15, fontweight='bold', pad=20)
    
    # Axe symétrique
    ax.set_xlim(-max_val*1.15, max_val*1.15)
    ax.axvline(0, color='black', linewidth=1.5)
    
    # Labels positifs
    labels = [abs(int(x)) for x in ax.get_xticks()]
    ax.set_xticklabels(labels)
    
    ax.legend(loc='lower right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2_pyramide_ages.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 2 sauvegardé: 2_pyramide_ages.png")


def graph_3_fumeurs(df):
    """Graphique 3: Fumeurs vs Non-fumeurs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df['fumeur'].value_counts()
    colors = [COULEURS['non_fumeur'] if 'Non' in x else COULEURS['fumeur'] for x in counts.index]
    
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor='white', linewidth=2, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_title('Statut Tabagique', 
                fontsize=15, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3_fumeurs.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 3 sauvegardé: 3_fumeurs_KEY.png")


def graph_4_stades(df):
    """Graphique 4: Répartition par stade"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ordre = ['I', 'II', 'III', 'IV', 'Inconnu']
    counts = df['stade'].value_counts().reindex(ordre, fill_value=0)
    
    # Couleurs dégradées
    colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C', '#95A5A6']
    
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor='white', linewidth=2, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_title('Répartition par Stade', fontsize=15, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_stades.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 4 sauvegardé: 4_stades.png")


def graph_5_mutations(df):
    """Graphique 5: Mutations - VERSION CLAIRE"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compter mutations POSITIVES uniquement
    mutations = ['EGFR', 'KRAS', 'ALK', 'BRAF']
    data = []
    
    for mut in mutations:
        col = f'mutation_{mut}_status'
        if col in df.columns:
            n_pos = (df[col] == 'Positive').sum()
            pct = n_pos / len(df) * 100
            data.append({'Mutation': mut, 'N': n_pos, 'Pct': pct})
    
    if not data:
        print("⚠️ Pas de données de mutations")
        return
    
    mut_df = pd.DataFrame(data).sort_values('Pct', ascending=False)
    
    bars = ax.bar(mut_df['Mutation'], mut_df['Pct'],
                  color='#E74C3C', edgecolor='white', linewidth=2, alpha=0.8)
    
    for bar, row in zip(bars, mut_df.itertuples()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={row.N})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('% de patients', fontsize=13, fontweight='bold')
    ax.set_title('Prévalence des Mutations POSITIVES', 
                fontsize=15, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_mutations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 5 sauvegardé: 5_mutations.png")


def graph_6_geographie(df):
    """Graphique 6: Répartition géographique"""
    if 'NOM_REG' not in df.columns:
        print("⚠️ Pas de données géographiques")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top10 = df['NOM_REG'].value_counts().head(10)
    
    bars = ax.barh(range(len(top10)), top10.values,
                   color='#3498DB', edgecolor='white', linewidth=1.5, alpha=0.8)
    
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index, fontsize=11)
    ax.set_xlabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_title('🔑 Répartition Géographique (Top 10 régions)',
                fontsize=15, fontweight='bold', pad=20)
    
    for i, (bar, val) in enumerate(zip(bars, top10.values)):
        ax.text(val + max(top10)*0.01, i, f'{val} ({val/len(df)*100:.1f}%)',
                va='center', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('6_geographie_KEY.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 6 sauvegardé: 6_geographie_KEY.png")


def graph_7_non_fumeurs(df):
    """Graphique 7: Profil des non-fumeurs"""
    df_nf = df[df['fumeur'] == 'Non-fumeur'].copy()
    
    if len(df_nf) == 0:
        print("⚠️ Aucun non-fumeur")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'🔑 PROFIL DES NON-FUMEURS (n={len(df_nf)} / {len(df)})',
                fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Sexe
    ax1 = axes[0,0]
    ordre_sexe = ['Homme', 'Femme', 'Non disponible']
    sexe = df_nf['sexe_clean'].value_counts().reindex(ordre_sexe, fill_value=0)
    colors_sexe = [COULEURS['homme'], COULEURS['femme'], COULEURS['non_dispo']]
    bars1 = ax1.bar(range(len(sexe)), sexe.values, color=colors_sexe, alpha=0.8)
    ax1.set_xticks(range(len(sexe)))
    ax1.set_xticklabels(sexe.index, rotation=15, ha='right')
    ax1.set_title('Sexe', fontweight='bold')
    ax1.set_ylabel('Nombre')
    for i, v in enumerate(sexe.values):
        if v > 0:
            ax1.text(i, v, f'{v}\n({v/len(df_nf)*100:.0f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # 2. Âge
    ax2 = axes[0,1]
    ages = df_nf['age_diagnostic'].dropna()
    ax2.hist(ages, bins=15, color=COULEURS['non_fumeur'], alpha=0.7, edgecolor='white')
    ax2.axvline(ages.median(), color='red', linestyle='--', linewidth=2,
               label=f'Médiane: {ages.median():.0f} ans')
    ax2.set_title('Âge au diagnostic', fontweight='bold')
    ax2.set_xlabel('Âge')
    ax2.set_ylabel('Nombre')
    ax2.legend()
    
    # 3. Stades
    ax3 = axes[1,0]
    stades = df_nf['stade'].value_counts().reindex(['I','II','III','IV','Inconnu'], fill_value=0)
    colors_s = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C', '#95A5A6']
    ax3.bar(stades.index, stades.values, color=colors_s, alpha=0.8)
    ax3.set_title('Stades', fontweight='bold')
    ax3.set_ylabel('Nombre')
    for i, v in enumerate(stades.values):
        if v > 0:
            ax3.text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Mutations
    ax4 = axes[1,1]
    mut_counts = df_nf['nb_mut_positives'].value_counts().sort_index()
    ax4.bar(mut_counts.index, mut_counts.values, 
            color=COULEURS['mutation_pos'], alpha=0.8)
    ax4.set_title('Nombre de mutations positives', fontweight='bold')
    ax4.set_xlabel('Nombre de mutations')
    ax4.set_ylabel('Nombre de patients')
    for i, v in enumerate(mut_counts.values):
        ax4.text(mut_counts.index[i], v, f'{v}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('7_profil_non_fumeurs_KEY.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Graphique 7 sauvegardé: 7_profil_non_fumeurs_KEY.png")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def analyse_complete(df):
    """Lance l'analyse complète"""
    print("\n" + "="*70)
    print("🚀 ANALYSE COHORTE CANCER DU POUMON")
    print("="*70)
    print("\n🎯 Question: Qualité de l'air → Cancer chez non-fumeurs ?\n")
    
    # # 1. Charger
    # print("📂 Chargement des données...")
    # try:
    #     df = pd.read_csv(fichier)
    #     print(f"   ✓ {len(df)} patients chargés\n")
    # except:
    #     print(f"   ❌ Fichier '{fichier}' introuvable")
    #     return None
    
    print("📂 Données déjà chargées\n")
    # 2. Préparer
    df = preparer_donnees(df)
    
    # 3. Stats
    stats_descriptives(df)
    
    # 4. Graphiques
    print("📊 Génération des graphiques...\n")
    graph_1_sexe(df)
    graph_2_pyramide_ages(df)
    graph_3_fumeurs(df)
    graph_4_stades(df)
    graph_5_mutations(df)
    graph_6_geographie(df)
    graph_7_non_fumeurs(df)
    
    # Résumé
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE")
    print("="*70)
    print("\n📊 7 graphiques sauvegardés (PNG haute résolution)")
    print("\n🎯 PROCHAINES ÉTAPES:")
    print("   1. ✅ Non-fumeurs identifiés et caractérisés")
    print("   2. ✅ Répartition géographique connue")
    print("   3. ⏳ Obtenir données qualité de l'air par région")
    print("   4. ⏳ Croiser : Cancer non-fumeurs × Pollution")
    print("="*70 + "\n")
    
    return df


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    df = analyse_complete('patients.csv')