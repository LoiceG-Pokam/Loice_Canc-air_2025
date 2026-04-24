"""
═══════════════════════════════════════════════════════════════════════════════
LUNG-CANC'AIR — Moteur d'analyse modulaire
═══════════════════════════════════════════════════════════════════════════════

Usage dans le notebook :
    from lungcancair_engine import AnalyseConfig, run_analyse

Exemple minimal :
    cfg = AnalyseConfig()
    results = run_analyse(df_final, cfg)

Exemple stratifié par sexe :
    cfg = AnalyseConfig(filtre_sexe="feminin")
    results = run_analyse(df_final, cfg)

Exemple EGFR seul :
    cfg = AnalyseConfig(outcome_mutation="EGFR")
    results = run_analyse(df_final, cfg)

Exemple dose-réponse O3 :
    from lungcancair_engine import run_dose_reponse
    run_dose_reponse(df_final, polluant="O3_cumul_120m", groupe="groupe_CD")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyseConfig:
    """
    Configuration complète d'une analyse. Tous les paramètres ont des valeurs
    par défaut — changer uniquement ce dont vous avez besoin.

    ── Filtres sur la population ──────────────────────────────────────────────
    filtre_sexe         : None | "feminin" | "masculin"
    filtre_fumeur       : None | True (fumeurs) | False (non-fumeurs)
    filtre_histologie   : None | "adenocarcinome" | "epidermoide" | autre
    filtre_mutation     : None | "EGFR" | "ALK" | ... (garder uniquement les porteurs)
    filtre_custom       : dict {colonne: valeur} pour tout autre filtre

    ── Outcome ───────────────────────────────────────────────────────────────
    groupe              : "AB" | "CD" | "ACBD"
    outcome_mutation    : None | "EGFR" | "ALK" | ... (régression mutation vs tous)

    ── Variables d'exposition ────────────────────────────────────────────────
    polluants_cumul     : liste des variables de cumul à inclure
    polluants_tendance  : liste des variables de tendance Sen à inclure
    pct_pm25            : seuils PM2.5 à inclure (ex: [10, 15])
    pct_pm10            : seuils PM10 à inclure (ex: [35, 45])
    pct_o3              : seuils O3 à inclure (ex: [100, 120])

    ── Variables contextuelles ───────────────────────────────────────────────
    vars_icpe           : liste des variables ICPE à inclure
    inclure_trafic      : inclure indice_trafic
    inclure_edi         : inclure quintileEDI2021

    ── Variables cliniques ───────────────────────────────────────────────────
    inclure_paquet_annee: inclure paquet_annee (False si C vs D)
    inclure_age         : inclure age_diagnostic
    inclure_sexe        : inclure sexe_bin

    ── Interactions ──────────────────────────────────────────────────────────
    interaction_sexe_var: variable à croiser avec le sexe (ex: "dist_ICPE_SH_m")

    ── Fenêtre temporelle ────────────────────────────────────────────────────
    fenetre_ans         : 10 (défaut) | 5 | 15 — filtre sur date_diagnostic

    ── Affichage ─────────────────────────────────────────────────────────────
    titre_custom        : titre affiché sur les graphiques (None = auto)
    couleur             : couleur hex pour les graphiques
    verbose             : True = afficher les résultats dans la console
    """

    # ── Filtres population ────────────────────────────────────────────────────
    filtre_sexe        : Optional[str]  = None
    filtre_fumeur      : Optional[bool] = None
    filtre_histologie  : Optional[str]  = None
    filtre_mutation    : Optional[str]  = None
    filtre_custom      : Dict           = field(default_factory=dict)

    # ── Outcome ───────────────────────────────────────────────────────────────
    groupe             : str            = "AB"
    outcome_mutation   : Optional[str]  = None

    # ── Variables exposition ──────────────────────────────────────────────────
    polluants_cumul    : List[str]      = field(default_factory=lambda: [
        "PM25_cumul_120m", "PM10_cumul_120m", "O3_cumul_120m"
    ])
    polluants_tendance : List[str]      = field(default_factory=lambda: [
        "PM25_mm365_sen_pente", "PM10_mm365_sen_pente", "O3_mm365_sen_pente"
    ])
    pct_pm25           : List[int]      = field(default_factory=lambda: [10])
    pct_pm10           : List[int]      = field(default_factory=lambda: [35])
    pct_o3             : List[int]      = field(default_factory=lambda: [100, 120])

    # ── Variables contextuelles ───────────────────────────────────────────────
    vars_icpe          : List[str]      = field(default_factory=lambda: [
        "nb_ICPE_total_3km", "dist_ICPE_SH_m"
    ])
    inclure_trafic     : bool           = True
    inclure_edi        : bool           = True

    # ── Variables cliniques ───────────────────────────────────────────────────
    inclure_paquet_annee: bool          = True
    inclure_age        : bool           = True
    inclure_sexe       : bool           = True

    # ── Interactions ──────────────────────────────────────────────────────────
    interaction_sexe_var: Optional[str] = None

    # ── Fenêtre temporelle ────────────────────────────────────────────────────
    fenetre_ans        : int            = 10

    # ── Affichage ─────────────────────────────────────────────────────────────
    titre_custom       : Optional[str]  = None
    couleur            : str            = "2E86AB"
    verbose            : bool           = True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════════════════

COULEURS_GROUPES = {
    "AB":   "2E86AB",
    "CD":   "52B788",
    "ACBD": "7B2D8B",
}

NOMS_GROUPES = {
    "AB":   ("Groupe A", "Groupe B",   "groupe_AB",    "Mutations NF vs Autres"),
    "CD":   ("Groupe C", "Groupe D",   "groupe_CD",    "Non-fumeurs vs Fumeurs"),
    "ACBD": ("Groupe A+C","Groupe B+D","groupe_AC_BD", "Profil Environnemental vs Autres"),
}

def _titre(cfg: AnalyseConfig) -> str:
    if cfg.titre_custom:
        return cfg.titre_custom
    parts = []
    if cfg.filtre_sexe:
        parts.append(cfg.filtre_sexe.capitalize())
    if cfg.filtre_fumeur is True:
        parts.append("Fumeurs")
    elif cfg.filtre_fumeur is False:
        parts.append("Non-fumeurs")
    if cfg.filtre_histologie:
        parts.append(cfg.filtre_histologie)
    if cfg.outcome_mutation:
        return f"Mutation {cfg.outcome_mutation} — {' | '.join(parts) if parts else 'Cohorte entière'}"
    g = NOMS_GROUPES.get(cfg.groupe, ("?", "?", "?", "?"))
    base = g[3]
    if parts:
        return f"{base} — {' | '.join(parts)}"
    return base

def _couleur(cfg: AnalyseConfig) -> str:
    if cfg.couleur != "2E86AB":
        return cfg.couleur
    return COULEURS_GROUPES.get(cfg.groupe, "2E86AB")

def _appliquer_filtres(df: pd.DataFrame, cfg: AnalyseConfig) -> pd.DataFrame:
    """Applique tous les filtres de population définis dans la config."""
    d = df.copy()

    if cfg.filtre_sexe:
        col = "sexe" if "sexe" in d.columns else None
        if col:
            d = d[d[col].str.strip().str.lower() == cfg.filtre_sexe.lower()]

    if cfg.filtre_fumeur is not None:
        if "paquet_annee" in d.columns:
            if cfg.filtre_fumeur:
                d = d[d["paquet_annee"] > 0]
            else:
                d = d[d["paquet_annee"] == 0]

    if cfg.filtre_histologie and "histologie_groupe" in d.columns:
        d = d[d["histologie_groupe"].str.lower() == cfg.filtre_histologie.lower()]

    if cfg.filtre_mutation:
        col_mut = f"mutation_{cfg.filtre_mutation}"
        if col_mut in d.columns:
            d = d[d[col_mut].notna()]

    for col, val in cfg.filtre_custom.items():
        if col in d.columns:
            d = d[d[col] == val]

    return d.reset_index(drop=True)

def _construire_vars(df: pd.DataFrame, cfg: AnalyseConfig) -> Tuple[List[str], List[str]]:
    """
    Retourne (vars_modele, vars_continues) en ne gardant que les colonnes
    réellement présentes dans df.
    """
    cols = set(df.columns)

    # ── Exposition ────────────────────────────────────────────────────────────
    vars_exp = []
    for v in cfg.polluants_cumul:
        if v in cols: vars_exp.append(v)
    for v in cfg.polluants_tendance:
        if v in cols: vars_exp.append(v)
    for s in cfg.pct_pm25:
        v = f"PM25_pct_sup{s}"
        if v in cols: vars_exp.append(v)
    for s in cfg.pct_pm10:
        v = f"PM10_pct_sup{s}"
        if v in cols: vars_exp.append(v)
    for s in cfg.pct_o3:
        v = f"O3_pct_sup{s}"
        if v in cols: vars_exp.append(v)

    # ── Cliniques ─────────────────────────────────────────────────────────────
    vars_clin = []
    if cfg.inclure_age and "age_diagnostic" in cols:
        vars_clin.append("age_diagnostic")
    if cfg.inclure_paquet_annee and "paquet_annee" in cols:
        vars_clin.append("paquet_annee")
    if cfg.inclure_sexe and "sexe_bin" in cols:
        vars_clin.append("sexe_bin")

    # ── Contextuelles ─────────────────────────────────────────────────────────
    vars_ctx = []
    if cfg.inclure_edi and "quintileEDI2021" in cols:
        vars_ctx.append("quintileEDI2021")
    if cfg.inclure_trafic and "indice_trafic" in cols:
        vars_ctx.append("indice_trafic")
    for v in cfg.vars_icpe:
        if v in cols: vars_ctx.append(v)

    vars_modele = vars_exp + vars_clin + vars_ctx

    # ── Terme d'interaction sexe × variable ──────────────────────────────────
    if cfg.interaction_sexe_var and cfg.interaction_sexe_var in cols and "sexe_bin" in cols:
        nom_inter = f"sexe_x_{cfg.interaction_sexe_var}"
        # sera créé dans _preparer_df
        vars_modele.append(nom_inter)

    # Variables continues = toutes sauf sexe_bin (0/1 mais OK à standardiser)
    vars_cont = [v for v in vars_modele if v in cols or v.startswith("sexe_x_")]

    return vars_modele, vars_cont

def _preparer_df(df: pd.DataFrame, cfg: AnalyseConfig,
                 vars_modele: List[str], vars_cont: List[str],
                 col_g: str, g1: str, g2: str,
                 outcome_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prépare X et y pour la régression."""
    d = df.copy()

    # Terme d'interaction
    if cfg.interaction_sexe_var and cfg.interaction_sexe_var in d.columns and "sexe_bin" in d.columns:
        nom_inter = f"sexe_x_{cfg.interaction_sexe_var}"
        d[nom_inter] = d["sexe_bin"] * d[cfg.interaction_sexe_var]

    # Colonnes disponibles
    cols_dispo = [v for v in vars_modele if v in d.columns]

    if outcome_col:
        d["outcome"] = d[outcome_col].astype(int)
    elif col_g in d.columns:
        d = d[d[col_g].isin([g1, g2])].copy()
        d["outcome"] = (d[col_g] == g1).astype(int)

    d = d[cols_dispo + ["outcome"]].dropna()

    scaler = StandardScaler()
    cont_dispo = [v for v in vars_cont if v in d.columns]
    d_std = d.copy()
    d_std[cont_dispo] = scaler.fit_transform(d[cont_dispo])

    X = sm.add_constant(d_std[cols_dispo])
    y = d_std["outcome"]

    return X, y, d

def _tableau_OR(modele, vars_modele: List[str]) -> pd.DataFrame:
    """Construit le tableau OR/IC95%/p-value."""
    rows = []
    for var in vars_modele:
        if var not in modele.params.index:
            continue
        p = modele.pvalues[var]
        if isinstance(p, pd.Series):
            p = p.iloc[0]
        rows.append({
            "Variable"    : var,
            "OR"          : round(float(np.exp(modele.params[var])), 3),
            "IC 95% inf"  : round(float(np.exp(modele.conf_int().loc[var, 0])), 3),
            "IC 95% sup"  : round(float(np.exp(modele.conf_int().loc[var, 1])), 3),
            "p-value"     : "<0.001" if p < 0.001 else f"{p:.3f}",
            "p_num"       : p,
            "Sig"         : "✅" if p < 0.05 else "—",
        })
    return pd.DataFrame(rows)

def _plot_roc_forest(modele, X, y, vars_modele: List[str],
                     titre: str, couleur: str) -> float:
    """Affiche ROC + Forest plot côte à côte."""
    y_pred = modele.predict(X)
    auc    = roc_auc_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_pred)

    couleur_hex = f"#{couleur}"

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(vars_modele) * 0.38 + 1)))

    # ROC
    axes[0].plot(fpr, tpr, color=couleur_hex, lw=2.5, label=f"AUC = {auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    axes[0].set_xlabel("1 - Spécificité"); axes[0].set_ylabel("Sensibilité")
    axes[0].set_title(f"Courbe ROC — {titre}", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Forest
    vars_ok    = [v for v in vars_modele if v in modele.params.index]
    ORs        = [float(np.exp(modele.params[v])) for v in vars_ok]
    IC_inf     = [float(np.exp(modele.conf_int().loc[v, 0])) for v in vars_ok]
    IC_sup     = [float(np.exp(modele.conf_int().loc[v, 1])) for v in vars_ok]
    pvalues    = [modele.pvalues[v] if not isinstance(modele.pvalues[v], pd.Series)
                  else modele.pvalues[v].iloc[0] for v in vars_ok]
    couleurs_p = [couleur_hex if p < 0.05 else "#AAAAAA" for p in pvalues]

    axes[1].axvline(x=1, color="black", linestyle="--", lw=1.2)
    for i, (OR, lo, hi, c, p) in enumerate(zip(ORs, IC_inf, IC_sup, couleurs_p, pvalues)):
        axes[1].plot([lo, hi], [i, i], color=c, lw=2.2)
        axes[1].plot(OR, i, "o", color=c, markersize=8)
        if p < 0.05:
            p_txt = "<0.001" if p < 0.001 else f"{p:.3f}"
            axes[1].text(max(IC_sup)*1.05, i, f"p={p_txt}",
                         va="center", fontsize=8, color=couleur_hex, fontweight="bold")

    axes[1].set_yticks(range(len(vars_ok)))
    axes[1].set_yticklabels(vars_ok, fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Odds Ratio (IC 95%)")
    axes[1].set_title(f"Forest plot — {titre}", fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return auc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analyse(df: pd.DataFrame, cfg: AnalyseConfig) -> Dict:
    """
    Lance une régression logistique complète selon la configuration.

    Retourne un dictionnaire avec :
        - "modele"    : objet statsmodels
        - "tableau"   : DataFrame OR/IC95%/p
        - "auc"       : float
        - "n"         : effectif
        - "n_outcome" : n outcome=1
        - "cfg"       : config utilisée
    """
    titre   = _titre(cfg)
    couleur = _couleur(cfg)

    # ── 1. Filtres ────────────────────────────────────────────────────────────
    df_f = _appliquer_filtres(df, cfg)

    if cfg.verbose:
        print(f"\n{'='*65}")
        print(f"ANALYSE : {titre}")
        print(f"{'='*65}")
        print(f"  Cohorte après filtres : {len(df_f)} patients")

    # ── 2. Outcome ────────────────────────────────────────────────────────────
    if cfg.outcome_mutation:
        # Mutation individuelle vs tous les autres
        mut_col = f"mutation_{cfg.outcome_mutation}"
        if mut_col not in df_f.columns:
            raise ValueError(f"Colonne '{mut_col}' absente de df_final.")
        df_f["outcome"] = df_f[mut_col].notna().astype(int)
        col_g, g1, g2 = None, None, None
        n_pos = df_f["outcome"].sum()
        epv   = n_pos / max(1, len([v for v in cfg.polluants_cumul + cfg.polluants_tendance if v in df_f.columns]))
        if cfg.verbose:
            print(f"  Outcome : Mutation {cfg.outcome_mutation} | N+ = {n_pos} | EPV ≈ {epv:.1f}")
            if epv < 5:
                print(f"  ⚠️  EPV < 5 — résultats à interpréter avec prudence")
    else:
        g_info = NOMS_GROUPES.get(cfg.groupe)
        if not g_info:
            raise ValueError(f"groupe doit être 'AB', 'CD' ou 'ACBD'. Reçu : '{cfg.groupe}'")
        g1, g2, col_g, _ = g_info

        # Forcer paquet_annee=False pour C vs D
        if cfg.groupe == "CD":
            cfg.inclure_paquet_annee = False

    # ── 3. Variables ──────────────────────────────────────────────────────────
    vars_modele, vars_cont = _construire_vars(df_f, cfg)

    if cfg.verbose:
        print(f"  Variables modèle ({len(vars_modele)}) : {vars_modele}")

    # ── 4. Préparation X, y ───────────────────────────────────────────────────
    if cfg.outcome_mutation:
        X, y, df_prep = _preparer_df(df_f, cfg, vars_modele, vars_cont,
                                     col_g=None, g1=None, g2=None,
                                     outcome_col="outcome")
    else:
        X, y, df_prep = _preparer_df(df_f, cfg, vars_modele, vars_cont,
                                     col_g=col_g, g1=g1, g2=g2)

    if cfg.verbose:
        print(f"  N final = {len(df_prep)} | Outcome=1 : {int(y.sum())} ({y.mean()*100:.1f}%)")

    # ── 5. Régression ─────────────────────────────────────────────────────────
    modele = sm.Logit(y, X).fit(disp=False)

    # ── 6. Résultats ──────────────────────────────────────────────────────────
    vars_ok = [v for v in vars_modele if v in modele.params.index]
    tableau = _tableau_OR(modele, vars_ok)
    auc     = _plot_roc_forest(modele, X, y, vars_ok, titre, couleur)

    if cfg.verbose:
        r2 = 1 - (modele.llf / modele.llnull)
        print(f"\n  AUC = {auc:.3f} | R²McFadden = {r2:.3f} | AIC = {modele.aic:.1f}")
        print(f"\n  Tableau OR :")
        print(tableau.drop(columns=["p_num"]).to_string(index=False))

        sig = tableau[tableau["Sig"] == "✅"]
        if len(sig) > 0:
            print(f"\n  Variables significatives (p < 0.05) :")
            for _, row in sig.iterrows():
                print(f"    ✅ {row['Variable']:<35} OR={row['OR']:.3f} | p={row['p-value']}")
        else:
            print(f"\n  Aucune variable significative après ajustement.")

    return {
        "modele" : modele,
        "tableau": tableau,
        "auc"    : auc,
        "n"      : len(df_prep),
        "n_outcome": int(y.sum()),
        "cfg"    : cfg,
        "titre"  : titre,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANALYSE STRATIFIÉE PAR SEXE
# ═══════════════════════════════════════════════════════════════════════════════

def run_stratifie_sexe(df: pd.DataFrame, cfg: AnalyseConfig) -> Dict:
    """
    Relance l'analyse séparément chez les femmes et chez les hommes.
    Retourne les deux jeux de résultats + comparaison des OR.
    """
    print(f"\n{'='*65}")
    print("ANALYSE STRATIFIÉE PAR SEXE")
    print(f"{'='*65}")

    results = {}
    for sexe in ["feminin", "masculin"]:
        cfg_s = AnalyseConfig(**{
            **cfg.__dict__,
            "filtre_sexe": sexe,
            "inclure_sexe": False,  # sexe fixé → pas confondant
            "titre_custom": f"{_titre(cfg)} — {sexe.capitalize()}",
            "couleur": "E76F51" if sexe == "feminin" else "2E86AB",
        })
        try:
            results[sexe] = run_analyse(df, cfg_s)
        except Exception as e:
            print(f"  ⚠️ Échec pour {sexe} : {e}")
            results[sexe] = None

    # ── Comparaison des OR ────────────────────────────────────────────────────
    if results["feminin"] and results["masculin"]:
        print(f"\n── Comparaison OR Femmes vs Hommes ──")
        t_f = results["feminin"]["tableau"].set_index("Variable")
        t_m = results["masculin"]["tableau"].set_index("Variable")
        vars_communes = t_f.index.intersection(t_m.index)

        rows = []
        for var in vars_communes:
            rows.append({
                "Variable" : var,
                "OR_femmes": t_f.loc[var, "OR"],
                "p_femmes" : t_f.loc[var, "p-value"],
                "sig_F"    : t_f.loc[var, "Sig"],
                "OR_hommes": t_m.loc[var, "OR"],
                "p_hommes" : t_m.loc[var, "p-value"],
                "sig_H"    : t_m.loc[var, "Sig"],
            })

        df_comp = pd.DataFrame(rows)
        # Mettre en évidence les variables qui diffèrent selon le sexe
        df_comp["interaction?"] = df_comp.apply(
            lambda r: "⚠️ Oui" if r["sig_F"] != r["sig_H"] else "—", axis=1
        )
        print(df_comp.to_string(index=False))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOSE-RÉPONSE PAR QUINTILE
# ═══════════════════════════════════════════════════════════════════════════════

def run_dose_reponse(df: pd.DataFrame, cfg: AnalyseConfig,
                     polluant: str,
                     n_quintiles: int = 5,
                     tester_tendance: bool = True) -> pd.DataFrame:
    """
    Analyse dose-réponse par quintile d'exposition.

    Paramètres
    ----------
    polluant      : ex "O3_cumul_120m" ou "PM25_pct_sup10"
    n_quintiles   : nombre de tranches (défaut 5)
    tester_tendance: Cochran-Armitage sur les OR par quintile

    Retourne un DataFrame avec OR par quintile.
    """
    from scipy.stats import chi2_contingency

    titre   = _titre(cfg)
    couleur = _couleur(cfg)
    df_f    = _appliquer_filtres(df, cfg)

    g_info = NOMS_GROUPES.get(cfg.groupe)
    if not g_info:
        raise ValueError(f"groupe invalide : {cfg.groupe}")
    g1, g2, col_g, _ = g_info

    df_f = df_f[df_f[col_g].isin([g1, g2])].copy()
    df_f["outcome"] = (df_f[col_g] == g1).astype(int)

    if polluant not in df_f.columns:
        raise ValueError(f"Polluant '{polluant}' absent de df_final.")

    df_f["quintile"] = pd.qcut(df_f[polluant], q=n_quintiles, labels=False, duplicates="drop")
    df_f["quintile"] = df_f["quintile"] + 1  # 1-indexé

    vars_cov = []
    if cfg.inclure_age and "age_diagnostic" in df_f.columns:
        vars_cov.append("age_diagnostic")
    if cfg.inclure_paquet_annee and "paquet_annee" in df_f.columns and cfg.groupe != "CD":
        vars_cov.append("paquet_annee")
    if cfg.inclure_sexe and "sexe_bin" in df_f.columns:
        vars_cov.append("sexe_bin")

    results_q = []
    ref_df = df_f[df_f["quintile"] == 1].copy()

    for q in range(2, df_f["quintile"].max() + 1):
        q_df = df_f[df_f["quintile"] == q].copy()
        sub  = pd.concat([ref_df, q_df]).dropna(subset=["outcome"] + vars_cov)

        if len(sub) < 20:
            continue

        X_q = sm.add_constant(sub[vars_cov])
        try:
            m = sm.Logit(sub["outcome"], X_q).fit(disp=False)
            if "const" in m.params.index:
                or_q  = float(np.exp(m.params["const"]))
                p_q   = float(m.pvalues["const"])
                lo_q  = float(np.exp(m.conf_int().loc["const", 0]))
                hi_q  = float(np.exp(m.conf_int().loc["const", 1]))
            else:
                continue
        except Exception:
            continue

        med_q = df_f.loc[df_f["quintile"] == q, polluant].median()
        results_q.append({
            "Quintile": f"Q{q} vs Q1",
            "Médiane exp.": round(med_q, 2),
            "OR"        : round(or_q, 3),
            "IC inf"    : round(lo_q, 3),
            "IC sup"    : round(hi_q, 3),
            "p-value"   : "<0.001" if p_q < 0.001 else f"{p_q:.3f}",
            "Sig"       : "✅" if p_q < 0.05 else "—",
        })

    df_or = pd.DataFrame(results_q)

    # ── Graphique ─────────────────────────────────────────────────────────────
    if len(df_or) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        qs    = range(len(df_or))
        ors   = df_or["OR"].values
        lo    = df_or["IC inf"].values
        hi    = df_or["IC sup"].values
        sigs  = df_or["Sig"].values

        ax.axhline(y=1, color="black", linestyle="--", lw=1.2, alpha=0.6)
        for i, (OR, l, h, sig) in enumerate(zip(ors, lo, hi, sigs)):
            c = f"#{couleur}" if sig == "✅" else "#AAAAAA"
            ax.plot([i, i], [l, h], color=c, lw=2.5)
            ax.plot(i, OR, "o", color=c, markersize=10)

        ax.set_xticks(range(len(df_or)))
        ax.set_xticklabels(df_or["Quintile"], rotation=15, ha="right")
        ax.set_ylabel("Odds Ratio (IC 95%) vs Q1")
        ax.set_title(f"Dose-réponse — {polluant}\n{titre}", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\nDose-réponse {polluant} — {titre}")
        print(df_or.to_string(index=False))

    return df_or


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ANALYSE SPATIALE EGFR × DISTANCE ICPE
# ═══════════════════════════════════════════════════════════════════════════════

def run_spatial_distance(df: pd.DataFrame, cfg: AnalyseConfig,
                         col_distance: str = "dist_ICPE_SH_m",
                         tranches_km: List[float] = None) -> pd.DataFrame:
    """
    Courbe dose-réponse sur la distance à un site ICPE.
    Calcule OR par tranche de distance vs la tranche la plus éloignée (référence).

    tranches_km : bornes en km. Défaut = [0, 1, 3, 5, 10, 999]
    """
    if tranches_km is None:
        tranches_km = [0, 1, 3, 5, 10, 999]

    titre   = _titre(cfg)
    couleur = _couleur(cfg)
    df_f    = _appliquer_filtres(df, cfg)

    if col_distance not in df_f.columns:
        raise ValueError(f"Colonne '{col_distance}' absente de df_final.")

    # Outcome
    if cfg.outcome_mutation:
        mut_col = f"mutation_{cfg.outcome_mutation}"
        df_f["outcome"] = df_f[mut_col].notna().astype(int)
    else:
        g1, g2, col_g, _ = NOMS_GROUPES[cfg.groupe]
        df_f = df_f[df_f[col_g].isin([g1, g2])].copy()
        df_f["outcome"] = (df_f[col_g] == g1).astype(int)

    # Tranches en mètres
    tranches_m  = [t * 1000 for t in tranches_km]
    labels      = [f"{tranches_km[i]}–{tranches_km[i+1]} km"
                   for i in range(len(tranches_km)-1)]
    df_f["tranche"] = pd.cut(df_f[col_distance],
                              bins=tranches_m, labels=labels, right=True)
    df_f = df_f.dropna(subset=["tranche", "outcome"])

    vars_cov = []
    if cfg.inclure_age and "age_diagnostic" in df_f.columns:
        vars_cov.append("age_diagnostic")
    if cfg.inclure_paquet_annee and "paquet_annee" in df_f.columns and cfg.groupe != "CD":
        vars_cov.append("paquet_annee")
    if cfg.inclure_sexe and "sexe_bin" in df_f.columns:
        vars_cov.append("sexe_bin")

    ref_label = labels[-1]
    ref_df    = df_f[df_f["tranche"] == ref_label].copy()

    results_d = []
    for label in labels[:-1]:
        sub = pd.concat([ref_df, df_f[df_f["tranche"] == label]]).dropna(subset=["outcome"] + vars_cov)
        if len(sub) < 10:
            continue

        X_d = sm.add_constant(sub[vars_cov])
        try:
            m = sm.Logit(sub["outcome"], X_d).fit(disp=False)
            or_d = float(np.exp(m.params["const"]))
            p_d  = float(m.pvalues["const"])
            lo_d = float(np.exp(m.conf_int().loc["const", 0]))
            hi_d = float(np.exp(m.conf_int().loc["const", 1]))
        except Exception:
            continue

        n_t = len(df_f[df_f["tranche"] == label])
        results_d.append({
            "Tranche"  : label,
            "N"        : n_t,
            "OR"       : round(or_d, 3),
            "IC inf"   : round(lo_d, 3),
            "IC sup"   : round(hi_d, 3),
            "p-value"  : "<0.001" if p_d < 0.001 else f"{p_d:.3f}",
            "Sig"      : "✅" if p_d < 0.05 else "—",
        })

    df_dist = pd.DataFrame(results_d)

    if len(df_dist) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x   = range(len(df_dist))
        ors = df_dist["OR"].values
        lo  = df_dist["IC inf"].values
        hi  = df_dist["IC sup"].values
        sig = df_dist["Sig"].values

        ax.axhline(y=1, color="black", linestyle="--", lw=1.2, alpha=0.6)
        for i, (OR, l, h, s) in enumerate(zip(ors, lo, hi, sig)):
            c = f"#{couleur}" if s == "✅" else "#AAAAAA"
            ax.plot([i, i], [l, h], color=c, lw=2.5)
            ax.plot(i, OR, "o", color=c, markersize=10)

        ax.set_xticks(range(len(df_dist)))
        ax.set_xticklabels(df_dist["Tranche"], rotation=20, ha="right")
        ax.set_xlabel(f"Distance à {col_distance.replace('_m','').replace('_',' ')}")
        ax.set_ylabel(f"OR vs tranche {ref_label} (IC 95%)")
        ax.set_title(f"Dose-réponse spatiale\n{titre}", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\nDose-réponse spatiale {col_distance} — {titre}")
        print(df_dist.to_string(index=False))

    return df_dist


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ANALYSE MULTI-FENÊTRES TEMPORELLES
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_fenetres(df: pd.DataFrame, cfg: AnalyseConfig,
                       fenetres: List[int] = None) -> pd.DataFrame:
    """
    Compare les AUC et OR d'un modèle sur plusieurs fenêtres temporelles.
    Nécessite que les variables d'exposition soient recalculées par fenêtre
    (colonnes avec suffixe _Xans si disponibles, sinon utilise les 10 ans).

    Retourne un DataFrame comparatif.
    """
    if fenetres is None:
        fenetres = [5, 10]

    results_f = []

    for f in fenetres:
        # Chercher des colonnes spécifiques à cette fenêtre si disponibles
        # Convention : PM25_cumul_{f*12}m (ex: PM25_cumul_60m pour 5 ans)
        mois = f * 12
        cfg_f = AnalyseConfig(**{
            **cfg.__dict__,
            "fenetre_ans"  : f,
            "polluants_cumul": [
                f"PM25_cumul_{mois}m" if f"PM25_cumul_{mois}m" in df.columns else "PM25_cumul_120m",
                f"O3_cumul_{mois}m"   if f"O3_cumul_{mois}m"   in df.columns else "O3_cumul_120m",
            ],
            "titre_custom" : f"{_titre(cfg)} — Fenêtre {f} ans",
            "verbose"      : False,
        })

        try:
            res = run_analyse(df, cfg_f)
            results_f.append({
                "Fenêtre": f"{f} ans",
                "N"      : res["n"],
                "AUC"    : round(res["auc"], 3),
                "N_sig"  : int((res["tableau"]["Sig"] == "✅").sum()),
            })
        except Exception as e:
            results_f.append({"Fenêtre": f"{f} ans", "N": 0, "AUC": None, "N_sig": 0, "Erreur": str(e)})

    df_f = pd.DataFrame(results_f)
    print(f"\n── Comparaison fenêtres temporelles — {_titre(cfg)} ──")
    print(df_f.to_string(index=False))
    return df_f


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — COMPARAISON DE PLUSIEURS ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════

def comparer_analyses(resultats: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare plusieurs résultats de run_analyse() côte à côte.

    Paramètres
    ----------
    resultats : dict {"label": résultat de run_analyse()}

    Retourne un DataFrame comparatif des AUC et variables significatives.
    """
    rows = []
    for label, res in resultats.items():
        if res is None:
            continue
        sig_vars = res["tableau"][res["tableau"]["Sig"] == "✅"]["Variable"].tolist()
        rows.append({
            "Analyse"   : label,
            "N"         : res["n"],
            "N_outcome" : res["n_outcome"],
            "AUC"       : round(res["auc"], 3),
            "N_sig"     : len(sig_vars),
            "Var. sig." : " | ".join(sig_vars) if sig_vars else "aucune",
        })

    df_comp = pd.DataFrame(rows)
    print("\n── Tableau comparatif des analyses ──")
    print(df_comp.to_string(index=False))
    return df_comp


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — EXEMPLES D'APPELS (commentés)
# ═══════════════════════════════════════════════════════════════════════════════

"""
# ── Analyse standard A vs B ──────────────────────────────────────────────────
cfg = AnalyseConfig(groupe="AB")
res = run_analyse(df_final, cfg)

# ── Analyse uniquement chez les femmes ───────────────────────────────────────
cfg = AnalyseConfig(groupe="AB", filtre_sexe="feminin")
res_f = run_analyse(df_final, cfg)

# ── Analyse EGFR avec interaction sexe × Seveso SH ───────────────────────────
cfg = AnalyseConfig(
    outcome_mutation       = "EGFR",
    interaction_sexe_var   = "dist_ICPE_SH_m",
    vars_icpe              = ["nb_ICPE_total_3km", "dist_ICPE_SH_m"],
)
res_egfr = run_analyse(df_final, cfg)

# ── Dose-réponse O3 dans C vs D ──────────────────────────────────────────────
cfg = AnalyseConfig(groupe="CD")
dr = run_dose_reponse(df_final, cfg, polluant="O3_cumul_120m", n_quintiles=5)

# ── Courbe spatiale EGFR × Seveso SH ─────────────────────────────────────────
cfg = AnalyseConfig(outcome_mutation="EGFR")
ds = run_spatial_distance(df_final, cfg, col_distance="dist_ICPE_SH_m",
                          tranches_km=[0, 1, 3, 5, 10, 999])

# ── Stratification par sexe ───────────────────────────────────────────────────
cfg = AnalyseConfig(groupe="AB")
res_sexe = run_stratifie_sexe(df_final, cfg)

# ── Comparaison multi-fenêtres ────────────────────────────────────────────────
cfg = AnalyseConfig(groupe="AB")
df_fenetres = run_multi_fenetres(df_final, cfg, fenetres=[5, 10])

# ── Comparer femmes vs hommes côte à côte ─────────────────────────────────────
res_f = run_analyse(df_final, AnalyseConfig(groupe="AB", filtre_sexe="feminin", couleur="E76F51"))
res_h = run_analyse(df_final, AnalyseConfig(groupe="AB", filtre_sexe="masculin", couleur="2E86AB"))
df_comp = comparer_analyses({"Femmes": res_f, "Hommes": res_h})

# ── Uniquement adénocarcinomes, non-fumeurs ───────────────────────────────────
cfg = AnalyseConfig(
    groupe              = "AB",
    filtre_histologie   = "adenocarcinome",
    filtre_fumeur       = False,
    pct_o3              = [100, 120],
    vars_icpe           = ["nb_ICPE_SH_3km", "nb_ICPE_SB_3km", "dist_ICPE_SH_m"],
)
res = run_analyse(df_final, cfg)
"""
