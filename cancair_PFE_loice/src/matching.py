import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

class CaseControlMatcher:
    """
    Classe optimisée pour l'appariement cas-témoins avec définition flexible des cas
    et analyse complète des facteurs confondants avant appariement.
    """

    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.confounder_results = None
        self.matching_quality = None
        self.df_matched_final = pd.DataFrame()
        
    def define_cases_controls(self, df, case_definition, control_definition=None):
        """
        Définit les cas et témoins selon des critères flexibles.
        
        Args:
            df: DataFrame
            case_definition: dict avec les critères pour définir les cas
                Exemples:
                - {'statut_deces_a_letude': 'oui'}
                - {'EM': ['aplasie', 'embolie']}
                - {'statut_deces_a_letude': 'oui', 'EM': ['aplasie']}
            control_definition: dict pour définir les témoins (optionnel)
                Si None, les témoins sont tous ceux qui ne sont pas des cas
        """
        print(f"\n{'='*60}")
        print("DÉFINITION DES CAS ET TÉMOINS")
        print(f"{'='*60}")
        
        # Création de la colonne de statut cas/témoin
        df = df.copy()
        df['is_case'] = False
        
        # Application des critères de cas
        case_mask = pd.Series([True] * len(df), index=df.index)
        
        for column, values in case_definition.items():
            if column not in df.columns:
                print(f"⚠️ Attention: Colonne '{column}' non trouvée dans le DataFrame")
                continue
                
            if isinstance(values, list):
                column_mask = df[column].isin(values)
            else:
                column_mask = (df[column] == values)
                
            case_mask = case_mask & column_mask
            
        df.loc[case_mask, 'is_case'] = True
        
        # Application des critères de témoins si spécifiés
        if control_definition:
            control_mask = pd.Series([True] * len(df), index=df.index)
            for column, values in control_definition.items():
                if column not in df.columns:
                    continue
                if isinstance(values, list):
                    column_mask = df[column].isin(values)
                else:
                    column_mask = (df[column] == values)
                control_mask = control_mask & column_mask
            
            # Seuls les individus satisfaisant les critères de témoins ET n'étant pas des cas
            df = df[control_mask | case_mask].copy()
            df.loc[control_mask & ~case_mask, 'is_case'] = False
        
        n_cases = df['is_case'].sum()
        n_controls = (~df['is_case']).sum()
        
        print(f"Critères de cas appliqués: {case_definition}")
        if control_definition:
            print(f"Critères de témoins appliqués: {control_definition}")
        else:
            print("Témoins: tous les individus ne satisfaisant pas les critères de cas")
            
        print(f"\nRésultats:")
        print(f"- Cas identifiés: {n_cases}")
        print(f"- Témoins identifiés: {n_controls}")
        print(f"- Ratio cas/témoins: 1:{n_controls/n_cases:.1f}" if n_cases > 0 else "- Ratio: impossible à calculer (0 cas)")
        
        return df

    def _statistical_test_association(self, df, var, outcome_var='is_case'):
        """
        Effectue un test statistique approprié selon le type de variable.
        """
        try:
            # Supprimer les valeurs manquantes
            temp_df = df[[var, outcome_var]].dropna()
            if len(temp_df) < 50:
                return {'error': 'Insufficient data', 'n_obs': len(temp_df)}
            
            cases = temp_df[temp_df[outcome_var] == True][var]
            controls = temp_df[temp_df[outcome_var] == False][var]
            
            if len(cases) == 0 or len(controls) == 0:
                return {'error': 'No cases or controls', 'n_cases': len(cases), 'n_controls': len(controls)}
            
            result = {'variable': var, 'n_obs': len(temp_df), 'n_cases': len(cases), 'n_controls': len(controls)}
            
            # Test selon le type de variable
            if pd.api.types.is_numeric_dtype(temp_df[var]):
                # Variable continue : t-test ou Mann-Whitney U
                if cases.nunique() <= 10 or controls.nunique() <= 10:
                    # Peu de valeurs uniques, traiter comme catégorielle
                    crosstab = pd.crosstab(temp_df[var], temp_df[outcome_var])
                    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                        chi2, p_value, _, _ = chi2_contingency(crosstab)
                        result.update({
                            'test_type': 'chi2',
                            'test_statistic': chi2,
                            'p_value': p_value,
                            'effect_size': np.sqrt(chi2 / len(temp_df))
                        })
                    else:
                        result['error'] = 'Insufficient variance for chi2'
                else:
                    # Test de normalité simple (Shapiro n'est pas fiable sur de gros échantillons)
                    try:
                        # T-test si les tailles d'échantillon sont raisonnables
                        if len(cases) > 5000 or len(controls) > 5000:
                            # Mann-Whitney U pour de gros échantillons
                            statistic, p_value = mannwhitneyu(cases, controls, alternative='two-sided')
                            result.update({
                                'test_type': 'mannwhitney',
                                'test_statistic': statistic,
                                'p_value': p_value,
                                'cases_median': cases.median(),
                                'controls_median': controls.median(),
                                'effect_size': abs(cases.median() - controls.median()) / temp_df[var].std()
                            })
                        else:
                            statistic, p_value = ttest_ind(cases, controls)
                            result.update({
                                'test_type': 'ttest',
                                'test_statistic': statistic,
                                'p_value': p_value,
                                'cases_mean': cases.mean(),
                                'controls_mean': controls.mean(),
                                'effect_size': abs(cases.mean() - controls.mean()) / temp_df[var].std()
                            })
                    except Exception:
                        # Fallback sur Mann-Whitney
                        statistic, p_value = mannwhitneyu(cases, controls, alternative='two-sided')
                        result.update({
                            'test_type': 'mannwhitney_fallback',
                            'test_statistic': statistic,
                            'p_value': p_value,
                            'cases_median': cases.median(),
                            'controls_median': controls.median()
                        })
            else:
                # Variable catégorielle : Chi2
                crosstab = pd.crosstab(temp_df[var], temp_df[outcome_var])
                if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                    chi2, p_value, _, expected = chi2_contingency(crosstab)
                    # Vérifier les conditions du chi2
                    min_expected = expected.min()
                    result.update({
                        'test_type': 'chi2',
                        'test_statistic': chi2,
                        'p_value': p_value,
                        'effect_size': np.sqrt(chi2 / len(temp_df)),
                        'min_expected_freq': min_expected,
                        'chi2_valid': min_expected >= 5
                    })
                else:
                    result['error'] = 'Insufficient variance for chi2'
            
            # Calcul de l'AUC pour comparaison
            try:
                if pd.api.types.is_numeric_dtype(temp_df[var]):
                    X = temp_df[[var]]
                    y = temp_df[outcome_var].astype(int)
                    if X[var].nunique() > 1:
                        model = LogisticRegression(max_iter=1000, solver='liblinear')
                        model.fit(X, y)
                        y_pred_proba = model.predict_proba(X)[:, 1]
                        auc = roc_auc_score(y, y_pred_proba)
                        result['auc'] = auc
                        result['or'] = np.exp(model.coef_[0][0])
            except Exception:
                pass
                
            return result
            
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}', 'variable': var}

    def analyze_confounders(self, df, potential_confounders, outcome_var='is_case', 
                            significance_threshold=0.05, effect_size_threshold=0.1):
        """
        Analyse tous les facteurs confondants potentiels avec tests statistiques appropriés.
        """
        print(f"\n{'='*60}")
        print("ANALYSE COMPLÈTE DES FACTEURS CONFONDANTS")
        print(f"{'='*60}")
        print(f"Variables à analyser: {len(potential_confounders)}")
        print(f"Seuil de significativité: {significance_threshold}")
        print(f"Seuil de taille d'effet: {effect_size_threshold}")
        
        results = []
        
        # Analyse en parallèle pour optimiser le temps
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_var = {
                executor.submit(self._statistical_test_association, df, var, outcome_var): var 
                for var in potential_confounders if var in df.columns
            }
            
            # Ajout de tqdm ici
            for future in tqdm(as_completed(future_to_var), 
                               total=len(future_to_var), 
                               desc="Analyse confondants"):
                var = future_to_var[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'variable': var, 'error': f'Execution error: {str(e)}'})
        
        # Conversion en DataFrame pour l'analyse
        results_df = pd.DataFrame(results)
        
        # Identification des confondants significatifs
        confounders_identified = []
        
        print(f"\n{'='*50}")
        print("RÉSULTATS DES TESTS STATISTIQUES")
        print(f"{'='*50}")
        
        for _, row in results_df.iterrows():
            if 'error' in row and pd.notna(row['error']):
                print(f"❌ {row['variable']}: {row['error']}")
                continue
                
            var = row['variable']
            p_val = row.get('p_value', 1.0)
            effect_size = row.get('effect_size', 0.0)
            test_type = row.get('test_type', 'unknown')
            
            is_significant = p_val < significance_threshold
            has_effect = effect_size > effect_size_threshold
            
            status = "🔴 CONFONDANT" if (is_significant and has_effect) else "⚪ Non significatif"
            
            print(f"{status} {var}:")
            print(f"   Test: {test_type.upper()}")
            print(f"   p-value: {p_val:.4f}")
            print(f"   Taille d'effet: {effect_size:.4f}")
            
            if test_type == 'ttest':
                print(f"   Moyennes - Cas: {row.get('cases_mean', 'N/A'):.2f}, Témoins: {row.get('controls_mean', 'N/A'):.2f}")
            elif test_type in ['mannwhitney', 'mannwhitney_fallback']:
                print(f"   Médianes - Cas: {row.get('cases_median', 'N/A'):.2f}, Témoins: {row.get('controls_median', 'N/A'):.2f}")
            elif test_type == 'chi2':
                print(f"   Chi2 valide: {row.get('chi2_valid', 'N/A')}")
                
            if 'auc' in row:
                print(f"   AUC: {row['auc']:.3f}")
            if 'or' in row:
                print(f"   OR: {row['or']:.3f}")
                
            print()
            
            if is_significant and has_effect:
                confounders_identified.append(var)
        
        print(f"\n🎯 FACTEURS CONFONDANTS IDENTIFIÉS ({len(confounders_identified)}):")
        for conf in confounders_identified:
            print(f"   • {conf}")
            
        self.confounder_results = {
            'results_df': results_df,
            'confounders_identified': confounders_identified,
            'significance_threshold': significance_threshold,
            'effect_size_threshold': effect_size_threshold,
            'n_cases': df[outcome_var].sum(),
            'n_controls': (~df[outcome_var]).sum()
        }
        
        return self.confounder_results

    def match_cases_controls_optimized(self, df, matching_vars, date_column_for_reference="date_deces"):
        """
        Apparie chaque témoin au cas le plus proche, permettant à un cas de
        servir de référence pour plusieurs témoins.

        Args:
            df (pd.DataFrame): DataFrame contenant une colonne 'is_case'
            matching_vars (list): Variables à utiliser pour l'appariement.
            date_column_for_reference (str): Nom de la colonne de date à utiliser comme date de référence.

        Returns:
            pd.DataFrame: DataFrame contenant tous les témoins avec leur date_reference et cas_apparie,
                          plus les cas originaux avec leur propre date_reference.
        """
        print(f"\n{'='*60}")
        print("APPARIEMENT TÉMOIN-VERS-CAS (PROPAGATION DE DATE)")
        print(f"{'='*60}")

        cases_df = df[df["is_case"] == True].copy()
        controls_df = df[df["is_case"] == False].copy()

        if cases_df.empty:
            print("❌ Aucun cas trouvé. Impossible d'apparier les témoins. Retourne un DataFrame vide.")
            return pd.DataFrame()
        
        if controls_df.empty:
            print("⚠️ Aucun témoin trouvé. Retourne seulement les cas.")
            # Si pas de témoins, on retourne juste les cas avec leur date de référence
            if date_column_for_reference in cases_df.columns:
                cases_df["date_reference"] = cases_df[date_column_for_reference]
            else:
                cases_df["date_reference"] = pd.NaT # Gérer les dates manquantes
            cases_df["cas_apparie"] = None
            # On ajoute ces colonnes pour la cohérence même si pas de témoins
            cases_df["matched_control_ids"] = None
            return cases_df

        if date_column_for_reference not in cases_df.columns:
            print(f"❌ La colonne de date de référence '{date_column_for_reference}' est introuvable dans les données des cas. Assurez-vous qu'elle existe.")
            return pd.DataFrame()

        # Préparation des données pour NearestNeighbors
        df_temp_for_nn = df.copy()
        
        numeric_vars = [v for v in matching_vars if v in df_temp_for_nn.columns and pd.api.types.is_numeric_dtype(df_temp_for_nn[v])]
        categorical_vars = [v for v in matching_vars if v in df_temp_for_nn.columns and not pd.api.types.is_numeric_dtype(df_temp_for_nn[v])]

        # Scaling des variables numériques
        if numeric_vars:
            scaler = StandardScaler()
            df_temp_for_nn[numeric_vars] = scaler.fit_transform(df_temp_for_nn[numeric_vars])
            print(f"✅ Variables numériques ({len(numeric_vars)}) scalées.")
        else:
            print("ℹ️ Aucune variable numérique à scaler parmi les variables d'appariement.")

        # One-hot encoding des variables catégorielles
        encoded_categorical_cols = []
        if categorical_vars:
            df_temp_for_nn_dummies = pd.get_dummies(df_temp_for_nn, columns=categorical_vars, drop_first=True, prefix=categorical_vars)
            encoded_categorical_cols = [col for col in df_temp_for_nn_dummies.columns if any(col.startswith(cat_v + '_') for cat_v in categorical_vars)]
            df_temp_for_nn = df_temp_for_nn_dummies
            print(f"✅ Variables catégorielles ({len(categorical_vars)}) encodées ({len(encoded_categorical_cols)} nouvelles colonnes).")
        else:
            print("ℹ️ Aucune variable catégorielle à encoder parmi les variables d'appariement.")

        vars_for_nn_model = numeric_vars + encoded_categorical_cols

        if not vars_for_nn_model:
            print("❌ Aucune variable valide trouvée pour l'appariement après le traitement. Veuillez vérifier `matching_vars`.")
            return pd.DataFrame()

        # Maintenant, le modèle NN est entraîné sur les CAS
        cases_encoded = df_temp_for_nn.loc[cases_df.index, vars_for_nn_model]
        controls_encoded = df_temp_for_nn.loc[controls_df.index, vars_for_nn_model]
        
        if cases_encoded.empty or controls_encoded.empty:
            print("⚠️ Les DataFrames encodés pour l'appariement sont vides après la sélection des colonnes. Vérifiez vos données et variables de matching.")
            return pd.DataFrame()

        print(f"\n⚙️ Apprentissage NearestNeighbors sur les cas ({len(cases_encoded)} cas) avec {len(vars_for_nn_model)} variables.")
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto") # Chaque témoin cherche 1 cas le plus proche
        nn.fit(cases_encoded)

        print(f"🔄 Recherche du cas le plus proche pour chaque témoin ({len(controls_encoded)} témoins).")
        # distances, indices = nn.kneighbors(controls_encoded) # Ancienne ligne, cause erreur d'attribut si contrôle est vide
        distances, indices = nn.kneighbors(controls_encoded)

        # Création des colonnes d'appariement pour les témoins
        matched_case_ids_for_controls = []
        
        # Le `indices` retourné par kneighbors est un tableau d'index dans `cases_encoded`
        # On doit le mapper aux index originaux des cas pour récupérer les infos complètes.
        case_original_indices = cases_df.index.tolist()

        for i in tqdm(range(len(controls_df)), desc="Appariement des témoins"):
            control_original_index = controls_df.index[i]
            
            # L'index du cas le plus proche dans le DataFrame `cases_encoded`
            closest_case_encoded_idx = indices[i][0]
            
            # L'index original du cas le plus proche dans le DataFrame `df`
            closest_case_original_idx = case_original_indices[closest_case_encoded_idx]
            
            # L'ID pseudo_provisoire du cas le plus proche
            closest_case_id = df.loc[closest_case_original_idx, "pseudo_provisoire"]
            
            matched_case_ids_for_controls.append(closest_case_id)

        # Ajouter les colonnes d'appariement aux témoins
        controls_df["cas_apparie"] = matched_case_ids_for_controls
        controls_df["matched_case_id"] = matched_case_ids_for_controls # Alias pour clarté
        
        # Les témoins n'ont pas de 'matched_control_ids' au sens où les cas en auraient
        controls_df["matched_control_ids"] = None 

        # 🔹 Créer la colonne date_reference pour les cas et témoins
        # Pour les cas, la date de référence est leur propre date d'événement
        if date_column_for_reference in cases_df.columns:
            cases_df["date_reference"] = cases_df[date_column_for_reference]
        else:
            print(f"⚠️ La colonne '{date_column_for_reference}' n'est pas trouvée dans les cas pour la date de référence. Les dates de référence des cas seront vides.")
            cases_df["date_reference"] = pd.NaT 
        cases_df["cas_apparie"] = None # Les cas ne sont pas "appariés à un cas" dans cette logique
        
        # Pour les témoins, la date de référence est celle du cas apparié
        case_id_to_date_reference = cases_df.set_index("pseudo_provisoire")["date_reference"].to_dict()
        controls_df["date_reference"] = controls_df["cas_apparie"].map(case_id_to_date_reference)


        # Les cas ont besoin d'une colonne matched_control_ids pour la cohérence
        # Mais dans cette nouvelle dynamique, un cas peut être apparié à plusieurs témoins.
        # Nous devons donc regrouper les témoins par cas apparié.
        temp_matched_control_ids = controls_df.groupby("cas_apparie")["pseudo_provisoire"].apply(lambda x: ','.join(map(str, x))).to_dict()
        cases_df["matched_control_ids"] = cases_df["pseudo_provisoire"].map(temp_matched_control_ids)

        # 🔹 Concaténation finale
        # On inclut tous les cas et tous les témoins, car chaque témoin a trouvé un cas de référence.
        self.df_matched_final = pd.concat([cases_df, controls_df], ignore_index=True)
        print(f"✅ Appariement terminé. {len(cases_df)} cas et {len(controls_df)} témoins ont été inclus.")

        return self.df_matched_final

    def _check_matching_quality_optimized(self, df_cases_matched, df_controls_matched, matching_vars):
        """
        Vérification optimisée de la qualité d'appariement.
        Cette méthode reste inchangée, mais son interprétation change :
        elle montre si les témoins *appariés* sont similaires aux cas *qui les ont servis de référence*.
        C'est moins une vérification d'équilibre de groupes que de pertinence de l'appariement.
        """
        print(f"\n{'='*40}")
        print("QUALITÉ DE L'APPARIEMENT (SIMILARITÉ TÉMOINS-CAS DE RÉFÉRENCE)")
        print(f"{'='*40}")
        
        if df_cases_matched.empty or df_controls_matched.empty:
            print("⚠️ Aucun cas ou témoin pour vérifier la qualité.")
            return {}

        quality_results = {}
        
        for var in matching_vars:
            if var not in df_cases_matched.columns or var not in df_controls_matched.columns:
                print(f"⚠️ Variable '{var}' non trouvée dans les DataFrames appariés, ignorée pour le contrôle qualité.")
                continue
                
            print(f"\n📊 {var.upper()}:")
            
            if pd.api.types.is_numeric_dtype(df_cases_matched[var]):
                # Variables numériques
                case_mean = df_cases_matched[var].mean()
                control_mean = df_controls_matched[var].mean()
                case_std = df_cases_matched[var].std()
                control_std = df_controls_matched[var].std()
                
                diff = abs(case_mean - control_mean)
                
                print(f"   Cas:    {case_mean:.2f} ± {case_std:.2f}")
                print(f"   Témoins: {control_mean:.2f} ± {control_std:.2f}")
                print(f"   Différence: {diff:.2f}")
                
                quality_results[var] = {
                    'type': 'numeric',
                    'difference': diff,
                    'case_mean': case_mean,
                    'control_mean': control_mean
                }
                
            else:
                # Variables catégorielles
                case_props = df_cases_matched[var].value_counts(normalize=True)
                control_props = df_controls_matched[var].value_counts(normalize=True)
                
                all_categories = set(case_props.index) | set(control_props.index)
                max_diff = 0
                
                print("   Proportions:")
                for cat in sorted(all_categories, key=str): # Utiliser str() pour trier les types mixtes
                    case_prop = case_props.get(cat, 0)
                    control_prop = control_props.get(cat, 0)
                    diff = abs(case_prop - control_prop)
                    max_diff = max(max_diff, diff)
                    
                    print(f"     {cat}: Cas {case_prop:.1%}, Témoins {control_prop:.1%} (Δ{diff:.1%})")
                
                quality_results[var] = {
                    'type': 'categorical',
                    'max_difference': max_diff
                }
        
        # Évaluation globale
        good_quality = True
        for var, results in quality_results.items():
            if results['type'] == 'numeric' and results['difference'] > 2: # Seuil à ajuster
                good_quality = False
            elif results['type'] == 'categorical' and results['max_difference'] > 0.15: # Seuil à ajuster
                good_quality = False
        
        status = "✅ EXCELLENTE" if good_quality else "⚠️ À AMÉLIORER"
        print(f"\n🎯 QUALITÉ GLOBALE: {status}")
        
        self.matching_quality = quality_results
        return quality_results

    def run_complete_analysis(self, df, case_definition, potential_confounders,
                              control_definition=None, auto_matching_vars=True,
                              manual_matching_vars=None, significance_threshold=0.05,
                              effect_size_threshold=0.1, date_column="date_deces"):
        """
        Analyse complète optimisée : définition des cas, analyse des confondants, appariement.
        Note: `max_controls_per_case` n'est plus pertinent pour l'appariement witness-to-case.
        """
        print("🚀 ANALYSE COMPLÈTE CAS-TÉMOINS OPTIMISÉE (PROPAGATION DE DATE)")
        print("="*80)
        
        # Étape 1: Définition des cas et témoins
        df_with_cases = self.define_cases_controls(df, case_definition, control_definition)
        
        if df_with_cases['is_case'].sum() == 0:
            print("❌ Aucun cas identifié avec les critères fournis. Impossible de trouver des dates de référence pour les témoins.")
            return None
            
        # Étape 2: Analyse des confondants
        # Cette étape est toujours pertinente pour identifier les variables importantes pour l'appariement.
        confounder_analysis = self.analyze_confounders(
            df_with_cases, 
            potential_confounders,
            significance_threshold=significance_threshold,
            effect_size_threshold=effect_size_threshold
        )
        
        # Étape 3: Détermination des variables d'appariement
        if auto_matching_vars and confounder_analysis and confounder_analysis['confounders_identified']:
            matching_variables = confounder_analysis['confounders_identified']
            print(f"\n💡 Variables d'appariement automatiques: {matching_variables}")
        elif manual_matching_vars:
            matching_variables = manual_matching_vars
            print(f"\n💡 Variables d'appariement manuelles: {matching_variables}")
        else:
            print("⚠️ Aucun confondant identifié ou `auto_matching_vars` est False et pas de variables manuelles. Utilisation de variables par défaut (age_a_letude, patient_sexe si présentes).")
            matching_variables = [var for var in ['age_a_letude', 'patient_sexe'] if var in df_with_cases.columns]
        
        # S'assurer que la colonne de date de référence existe
        if date_column not in df_with_cases.columns:
            print(f"❌ Erreur: La colonne '{date_column}' spécifiée pour la date de référence est introuvable dans le DataFrame. Veuillez la corriger.")
            return None

        # Étape 4: Appariement optimisé (tous les témoins appariés à un cas)
        df_matched = self.match_cases_controls_optimized(
            df_with_cases,
            matching_variables,
            date_column_for_reference=date_column
        )
        
        # Étape 5: Vérification de la qualité d'appariement (optionnel)
        if not df_matched.empty:
            # Pour la vérification de qualité, on compare les témoins AVEC les cas qui leur ont servi de référence.
            # On prend tous les cas qui ont été utilisés comme référence.
            # Pour les témoins, on ne prend que ceux qui ont été effectivement appariés.
            cases_for_quality_check = df_matched[df_matched['is_case'] == True].copy()
            controls_for_quality_check = df_matched[df_matched['is_case'] == False].copy()
            
            self._check_matching_quality_optimized(cases_for_quality_check, controls_for_quality_check, matching_variables)
        else:
            print("Aucun appariement réussi pour vérifier la qualité.")


        results = {
            'df_original': df,
            'df_with_case_definition': df_with_cases,
            'df_matched_final': df_matched,
            'case_definition': case_definition,
            'control_definition': control_definition,
            'confounder_analysis': confounder_analysis,
            'matching_variables': matching_variables,
            'matching_quality': self.matching_quality
        }
        
        print(f"\n✅ ANALYSE COMPLÈTE TERMINÉE")
        print("="*40)
        
        return results



# # src/matching.py
# import pandas as pd
# import numpy as np

# class CaseControlMatcher:
#     def score_matching(self, cas, temoin):
#         score = 0
#         score += abs((cas['age_a_letude'] - temoin['age_a_letude']) ** 2)
#         score += 100 * (cas['patient_sexe'] != temoin['patient_sexe'])
#         score += 100 * (cas['patho'] != temoin['patho'])
#         score += 100 * (cas['CODE_DEPT'] != temoin['CODE_DEPT'])
#         return score

#     def match_cases_controls(self, df):
#         df_temoins = df[df['statut_deces_a_letude'] == 'non'].copy()
#         df_cas = df[df['statut_deces_a_letude'] == 'oui'].copy()
        
#         df_cas['date_reference'] = pd.to_datetime(df_cas['date_derniere_nouvelle'])
        
#         print(f"Cas: {len(df_cas)}")
#         print(f"Témoins: {len(df_temoins)}")
        
#         dates_ref_temoins = []
#         temoin_cas_ids = []
        
#         for i, temoin in df_temoins.iterrows():
#             subset = df_cas[
#                 (df_cas['age_a_letude'] >= temoin['age_a_letude'] - 3) &
#                 (df_cas['age_a_letude'] <= temoin['age_a_letude'] + 3)
#             ].copy()
        
#             if len(subset) > 0:
#                 subset_sexe = subset[subset['patient_sexe'] == temoin['patient_sexe']]
#                 if len(subset_sexe) > 0:
#                     subset = subset_sexe
        
#             if len(subset) > 0:
#                 subset_patho = subset[subset['patho'] == temoin['patho']]
#                 if len(subset_patho) > 0:
#                     subset = subset_patho
        
#             if len(subset) > 0:
#                 scores = []
#                 cas_indices = []
        
#                 for idx, cas_row in subset.iterrows():
#                     score = self.score_matching(cas_row, temoin)
#                     scores.append(score)
#                     cas_indices.append(idx)
        
#                 best_idx = np.argmin(scores)
#                 chosen_case_idx = cas_indices[best_idx]
#                 chosen_case = df_cas.loc[chosen_case_idx]
        
#                 date_ref = chosen_case['date_reference']
#                 cas_id = chosen_case['pseudo_provisoire']
#             else:
#                 print(f"Aucun appariement trouvé pour le témoin {i}, utilisation d'une date aléatoire")
#                 # sampled_case = df_cas.sample(1)
#                 # date_ref = pd.to_datetime(sampled_case['date_reference'].values[0])
#                 # cas_id = sampled_case['pseudo_provisoire'].values[0]
        
#             dates_ref_temoins.append(date_ref)
#             temoin_cas_ids.append(cas_id)
        
#         # === Une fois la boucle terminée, ajouter les colonnes ===
#         df_temoins['date_reference'] = pd.to_datetime(dates_ref_temoins)
#         df_temoins['cas_apparie'] = temoin_cas_ids

        
#         # ========== Étape 3 : Ajouter la colonne statut_deces ==========
#         # df_cas['statut_deces'] = 1
#         # df_temoins['statut_deces'] = 0
        
#         # ========== Étape 4 : Concaténer ==========
#         df_all = pd.concat([df_cas, df_temoins], ignore_index=True)
        
#         print(f"\nDataset final: {len(df_all)} observations")
#         print(f"- Cas: {len(df_cas)}")
#         print(f"- Témoins: {len(df_temoins)}")
        
#         # ========== Vérification de l'appariement ==========
#         print("\n=== Vérification de l'appariement ===")
        
#         # Âge
#         print("\nDistribution des âges:")
#         print("Cas:")
#         print(df_cas['age_a_letude'].describe())
#         print("Témoins:")
#         print(df_temoins['age_a_letude'].describe())
        
#         # Sexe
#         print("\nDistribution par sexe:")
#         print("Cas:")
#         print(df_cas['patient_sexe'].value_counts())
#         print("Témoins:")
#         print(df_temoins['patient_sexe'].value_counts())
        
#         # Pathologie
#         print("\nDistribution par pathologie:")
#         print("Cas:")
#         print(df_cas['patho'].value_counts())
#         print("Témoins:")
#         print(df_temoins['patho'].value_counts())
        
#         # Dates anormales
#         print("\nDates de référence des témoins hors limites attendues (> 2020):")
#         print(df_temoins[df_temoins['date_reference'].dt.year > 2020][['pseudo_provisoire', 'date_reference', 'cas_apparie']])
    
#         return df_all

