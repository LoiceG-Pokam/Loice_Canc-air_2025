# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime as dt
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline as SklearnPipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
#                            roc_auc_score, precision_recall_curve, average_precision_score,
#                            precision_score, recall_score, f1_score, balanced_accuracy_score, make_scorer)
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import (SMOTE, RandomOverSampler)
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import BalancedRandomForestClassifier
# import json
# import joblib # Import joblib for saving models
# from sklearn.inspection import permutation_importance


# from imblearn.combine import SMOTEENN, SMOTETomek 
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
# from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score, recall_score, precision_score, roc_auc_score

# class EnhancedRigorousMLPipeline:
#     """
#     Pipeline ML rigoureux avec persistance, reprise d'expérience et visualisations complètes
#     Ajout de la capacité d'exécuter sur des sous-groupes de données.
#     """
    
#     def __init__(self, base_output_path="C:/experiments/", experiment_name="rigorous_ml"):
#         self.base_output_path = base_output_path
#         self.experiment_name = experiment_name
#         self.output_dir = None
#         self.results_history = []
#         self.completed_strategies = set()
#         self.experiment_state_file = None
#         self.current_subset_info = "all_data" # Pour la traçabilité des subsets

#     def create_output_directory(self, resume_experiment_id=None, subset_name=None):
#         """
#         Crée un répertoire organisé par date pour les résultats ou reprend une expérience.
#         Peut inclure un sous-répertoire pour les subsets.
#         """
#         if resume_experiment_id:
#             self.output_dir = resume_experiment_id
#             if not os.path.exists(self.output_dir):
#                 raise ValueError(f"Le répertoire d'expérience {resume_experiment_id} n'existe pas")
#             print(f"📁 Reprise de l'expérience : {self.output_dir}")
#         else:
#             today = dt.now().strftime("%Y-%m-%d")
#             timestamp = dt.now().strftime("%H-%M-%S")
            
#             base_run_dir = os.path.join(
#                 self.base_output_path, 
#                 f"{self.experiment_name}_{today}", 
#                 f"run_{timestamp}"
#             )
            
#             if subset_name:
#                 self.output_dir = os.path.join(base_run_dir, f"subset_{subset_name}")
#             else:
#                 self.output_dir = base_run_dir
            
#             os.makedirs(self.output_dir, exist_ok=True)
#             print(f"📁 Nouveau répertoire de sortie créé : {self.output_dir}")
        
#         # Créer les sous-répertoires pour l'organisation
#         self.plots_dir = os.path.join(self.output_dir, "plots")
#         self.models_dir = os.path.join(self.output_dir, "models")
#         self.data_dir = os.path.join(self.output_dir, "data")
#         self.metrics_dir = os.path.join(self.output_dir, "metrics")
        
#         for dir_path in [self.plots_dir, self.models_dir, self.data_dir, self.metrics_dir]:
#             os.makedirs(dir_path, exist_ok=True)
        
#         # Fichier d'état de l'expérience
#         self.experiment_state_file = os.path.join(self.output_dir, "experiment_state.json")
        
#         return self.output_dir
    
#     def save_experiment_state(self, all_results, current_strategy=None):
#         """Sauvegarde l'état actuel de l'expérience, incluant l'info du subset."""
#         state = {
#             'timestamp': dt.now().isoformat(),
#             'completed_strategies': list(self.completed_strategies),
#             'current_strategy': current_strategy,
#             'output_directory': self.output_dir,
#             'total_strategies': len(self.prepare_resampling_strategies()),
#             'progress': len(self.completed_strategies),
#             'current_subset': self.current_subset_info # Ajouter l'info du subset
#         }
        
#         with open(self.experiment_state_file, 'w') as f:
#             json.dump(state, f, indent=2)
    
#     def load_experiment_state(self):
#         """Charge l'état d'une expérience précédente"""
#         if os.path.exists(self.experiment_state_file):
#             with open(self.experiment_state_file, 'r') as f:
#                 state = json.load(f)
            
#             self.completed_strategies = set(state.get('completed_strategies', []))
#             self.current_subset_info = state.get('current_subset', 'all_data') # Charger l'info du subset
#             print(f"📋 État chargé: {len(self.completed_strategies)} stratégies déjà complétées pour subset '{self.current_subset_info}'")
#             print(f"   Stratégies terminées: {', '.join(self.completed_strategies)}")
#             return state
#         return None
    
#     def load_existing_results(self):
#         """Charge les résultats existants depuis le disque"""
#         results_file = os.path.join(self.output_dir, "comprehensive_experiment_results.json")
#         if os.path.exists(results_file):
#             with open(results_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             return data.get('detailed_results', {})
#         return {}
    
#     def prepare_resampling_strategies(self):
#         """Définit différentes stratégies de rééquilibrage"""
#         strategies = {
#             'baseline': None,
#             'class_weight': 'class_weight_only',
#             'random_oversample': RandomOverSampler(random_state=42),
#             'random_undersample': RandomUnderSampler(random_state=42),
#             'smote': SMOTE(random_state=42, k_neighbors=3),
#             'balanced_rf': 'balanced_random_forest',
#             # --- Nouvelles stratégies ---
#             'smoteenn': SMOTEENN(random_state=42), # Combinaison SMOTE et Edited Nearest Neighbors
#             'smotetomek': SMOTETomek(random_state=42) # Combinaison SMOTE et Tomek Links
#     }
#         return strategies
    
#     def create_pipeline(self, numerical_features, categorical_features, strategy_name, strategy):
#         """Crée un pipeline selon la stratégie choisie"""
        
#         preprocessor = ColumnTransformer([
#             ('num', SklearnPipeline([
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ]), numerical_features),
#             ('cat', SklearnPipeline([
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#             ]), categorical_features)
#         ])
        
#         if strategy_name == 'balanced_rf':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', BalancedRandomForestClassifier(
#                     random_state=42,
#                     n_estimators=200,
#                     max_depth=15,
#                     min_samples_split=5,
#                     min_samples_leaf=2
#                 ))
#             ])
#         elif strategy_name == 'class_weight':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', RandomForestClassifier(
#                     random_state=42,
#                     class_weight='balanced',
#                     n_estimators=200,
#                     max_depth=15,
#                     min_samples_split=5,
#                     min_samples_leaf=2
#                 ))
#             ])
#         elif strategy_name == 'baseline':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', RandomForestClassifier(
#                     random_state=42,
#                     n_estimators=200,
#                     max_depth=15,
#                     min_samples_split=5,
#                     min_samples_leaf=2
#                 ))
#             ])
#         else:
#             pipeline = ImbPipeline([
#                 ('preprocessor', preprocessor),
#                 ('resampler', strategy),
#                 ('classifier', RandomForestClassifier(
#                     random_state=42,
#                     n_estimators=200,
#                     max_depth=15,
#                     min_samples_split=5,
#                     min_samples_leaf=2
#                 ))
#             ])
        
#         return pipeline
    

#     def tune_hyperparameters(self, X_train, y_train, numerical_features, categorical_features,
#                              strategy_name, strategy, param_grid, search_type='grid', cv_folds=3, scoring='f1'):
#         """
#         Effectue une recherche d'hyperparamètres (GridSearchCV ou RandomizedSearchCV)
#         pour une stratégie donnée.

#         Args:
#             X_train, y_train: Données d'entraînement pour la recherche.
#             numerical_features, categorical_features: Listes des noms de colonnes.
#             strategy_name (str): Nom de la stratégie (ex: 'balanced_rf', 'smoteenn').
#             strategy: L'objet rééchantillonneur (pour ImbPipeline) ou 'balanced_random_forest'/'class_weight_only'/None.
#             param_grid (dict): Dictionnaire des hyperparamètres à tester.
#             search_type (str): 'grid' pour GridSearchCV ou 'random' pour RandomizedSearchCV.
#             cv_folds (int): Nombre de plis pour la validation croisée stratifiée.
#             scoring (str/scorer): Métrique de scoring à optimiser.

#         Returns:
#             best_pipeline: Le pipeline avec les meilleurs hyperparamètres trouvés.
#             best_params: Le dictionnaire des meilleurs hyperparamètres.
#             best_score: Le meilleur score obtenu.
#             cv_results: Les résultats complets de la recherche.
#         """
#         print(f"\n Démarrage du réglage des hyperparamètres pour {strategy_name} (méthode: {search_type.upper()})...")

#         # Initialisation d'un pipeline avec des paramètres par défaut qui seront écrasés par la recherche
#         # Les kwargs du classifieur seront passés via le param_grid
#         pipeline = self.create_pipeline(numerical_features, categorical_features, strategy_name, strategy, n_estimators=100, max_depth=10)

#         # Assurez-vous que la validation croisée est stratifiée
#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

#         if scoring == 'f1':
#             scorer = make_scorer(f1_score, zero_division=0)
#         elif scoring == 'balanced_accuracy':
#             scorer = 'balanced_accuracy' # Scikit-learn a déjà un scorer pour cela
#         elif scoring == 'roc_auc':
#             scorer = 'roc_auc'
#         elif scoring == 'recall':
#             scorer = make_scorer(recall_score, zero_division=0)
#         elif scoring == 'precision':
#             scorer = make_scorer(precision_score, zero_division=0)
#         else:
#             scorer = scoring # Accepte d'autres strings valides pour sklearn

#         if search_type == 'grid':
#             search = GridSearchCV(
#                 estimator=pipeline,
#                 param_grid=param_grid,
#                 scoring=scorer,
#                 cv=cv,
#                 n_jobs=-1,
#                 verbose=1
#             )
#         elif search_type == 'random':
#             search = RandomizedSearchCV(
#                 estimator=pipeline,
#                 param_distributions=param_grid,
#                 n_iter=50, # Nombre d'itérations à ajuster
#                 scoring=scorer,
#                 cv=cv,
#                 n_jobs=-1,
#                 verbose=1,
#                 random_state=42
#             )
#         else:
#             raise ValueError("search_type doit être 'grid' ou 'random'.")

#         search.fit(X_train, y_train)

#         print(f"✅ Réglage des hyperparamètres terminé pour {strategy_name}.")
#         print(f"   Meilleur score ({scoring}): {search.best_score_:.3f}")
#         print(f"   Meilleurs paramètres: {search.best_params_}")

#         return search.best_estimator_, search.best_params_, search.best_score_, search.cv_results_

    
#     def calculate_feature_importance(self, pipeline, X_val, y_val, strategy_name, features):
#         """Calcule l'importance des features avec plusieurs méthodes"""
#         print(f"🔍 Calcul de l'importance des features pour {strategy_name}...")

#         importance_results = {}

#         try:
#             # Ensure the preprocessor part of the pipeline is fitted.
#             # It should be fitted by the time calculate_feature_importance is called
#             preprocessor_step = pipeline.named_steps['preprocessor']

#             transformed_feature_names = None
#             try:
#                 # This call directly on the preprocessor step of the pipeline is correct
#                 # after the pipeline has been fitted.
#                 transformed_feature_names = preprocessor_step.get_feature_names_out()
#             except AttributeError:
#                 print("   ⚠️ Cannot get transformed feature names. Preprocessor might not be fitted or get_feature_names_out is unavailable.")

#             if transformed_feature_names is not None and hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
#                 importances = pipeline.named_steps['classifier'].feature_importances_

#                 if len(transformed_feature_names) == len(importances):
#                     importance_df = pd.DataFrame({
#                         'feature': transformed_feature_names,
#                         'importance': importances
#                     }).sort_values('importance', ascending=False)

#                     importance_results['built_in'] = importance_df.to_dict('records')
#                 else:
#                     print(f"   ⚠️ Mismatch in built-in feature importance length for {strategy_name}. Expected {len(transformed_feature_names)}, got {len(importances)}.")
#                     # Optionally print samples for debugging:
#                     # print(f"      Transformed feature names sample: {transformed_feature_names[:5]}...")
#                     # print(f"      Importances sample: {importances[:5]}...")

#             print("   Calcul de la permutation importance...")
#             # Permutation importance can use the original X_val columns directly with the full pipeline
#             perm_importance = permutation_importance(
#                 pipeline, X_val, y_val,
#                 n_repeats=5, random_state=42, n_jobs=-1
#             )

#             perm_importance_df = pd.DataFrame({
#                 'feature': X_val.columns.tolist(), # Use original feature names for reporting permutation importance
#                 'importance_mean': perm_importance.importances_mean,
#                 'importance_std': perm_importance.importances_std
#             }).sort_values('importance_mean', ascending=False)

#             importance_results['permutation'] = perm_importance_df.to_dict('records')

#         except Exception as e:
#             print(f"   ⚠️ Erreur lors du calcul d'importance: {str(e)}")
#             importance_results['error'] = str(e)

#         return importance_results
    
#     def plot_comprehensive_analysis(self, strategy_name, y_true, y_pred, y_pred_proba, 
#                                    importance_results, threshold_df, features):
#         """Crée tous les graphiques d'analyse pour une stratégie"""
        
#         plt.rcParams['figure.figsize'] = (15, 10)
        
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle(f'Analyse Complète - {strategy_name} (Subset: {self.current_subset_info})', fontsize=16, fontweight='bold') # Ajout du subset info
        
#         cm = confusion_matrix(y_true, y_pred)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
#         axes[0,0].set_title('Matrice de Confusion')
#         axes[0,0].set_xlabel('Prédiction')
#         axes[0,0].set_ylabel('Réalité')
        
#         fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#         roc_auc = roc_auc_score(y_true, y_pred_proba)
#         axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
#                        label=f'ROC curve (AUC = {roc_auc:.3f})')
#         axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         axes[0,1].set_xlim([0.0, 1.0])
#         axes[0,1].set_ylim([0.0, 1.05])
#         axes[0,1].set_xlabel('Taux de Faux Positifs')
#         axes[0,1].set_ylabel('Taux de Vrais Positifs')
#         axes[0,1].set_title('Courbe ROC')
#         axes[0,1].legend(loc="lower right")
        
#         precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
#         avg_precision = average_precision_score(y_true, y_pred_proba)
#         axes[0,2].plot(recall, precision, color='green', lw=2,
#                        label=f'PR curve (AP = {avg_precision:.3f})')
#         axes[0,2].set_xlim([0.0, 1.0])
#         axes[0,2].set_ylim([0.0, 1.05])
#         axes[0,2].set_xlabel('Recall')
#         axes[0,2].set_ylabel('Precision')
#         axes[0,2].set_title('Courbe Precision-Recall')
#         axes[0,2].legend(loc="lower left")
        
#         axes[1,0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Classe 0', color='blue')
#         axes[1,0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Classe 1', color='red')
#         axes[1,0].set_xlabel('Probabilité prédite')
#         axes[1,0].set_ylabel('Fréquence')
#         axes[1,0].set_title('Distribution des Probabilités')
#         axes[1,0].legend()
        
#         axes[1,1].plot(threshold_df['threshold'], threshold_df['f1'], label='F1-Score', color='blue')
#         axes[1,1].plot(threshold_df['threshold'], threshold_df['balanced_accuracy'], label='Balanced Accuracy', color='green')
#         axes[1,1].plot(threshold_df['threshold'], threshold_df['youden_index'], label='Youden Index', color='red')
#         axes[1,1].set_xlabel('Seuil')
#         axes[1,1].set_ylabel('Score')
#         axes[1,1].set_title('Optimisation des Seuils')
#         axes[1,1].legend()
#         axes[1,1].grid(True, alpha=0.3)
        
#         if 'permutation' in importance_results:
#             perm_data = pd.DataFrame(importance_results['permutation'])
#             top_features = perm_data.head(15) 
            
#             axes[1,2].barh(range(len(top_features)), top_features['importance_mean'])
#             axes[1,2].set_yticks(range(len(top_features)))
#             axes[1,2].set_yticklabels(top_features['feature'])
#             axes[1,2].set_xlabel('Importance (Permutation)')
#             axes[1,2].set_title('Top 15 Features Importantes')
#             axes[1,2].invert_yaxis()
        
#         plt.tight_layout()
        
#         plot_path = os.path.join(self.plots_dir, f'analysis_complete_{strategy_name}_{self.current_subset_info}.png') # Ajout du subset info
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         if 'permutation' in importance_results:
#             self.plot_detailed_feature_importance(strategy_name, importance_results, features)
    
#     def plot_detailed_feature_importance(self, strategy_name, importance_results, features):
#         """Plot détaillé de l'importance des features"""
#         fig, axes = plt.subplots(1, 2, figsize=(20, 8))
#         fig.suptitle(f'Importance des Features - {strategy_name} (Subset: {self.current_subset_info})', fontsize=16, fontweight='bold') # Ajout du subset info
        
#         if 'permutation' in importance_results:
#             perm_data = pd.DataFrame(importance_results['permutation'])
#             top_20 = perm_data.head(20)
            
#             axes[0].barh(range(len(top_20)), top_20['importance_mean'], 
#                          xerr=top_20['importance_std'], capsize=3)
#             axes[0].set_yticks(range(len(top_20)))
#             axes[0].set_yticklabels(top_20['feature'])
#             axes[0].set_xlabel('Importance Moyenne (avec écart-type)')
#             axes[0].set_title('Permutation Importance - Top 20')
#             axes[0].invert_yaxis()
#         axes[0].grid(True, alpha=0.3, axis='x')
        
#         if 'built_in' in importance_results:
#             builtin_data = pd.DataFrame(importance_results['built_in'])
#             top_20_builtin = builtin_data.head(20)
            
#             axes[1].barh(range(len(top_20_builtin)), top_20_builtin['importance'])
#             axes[1].set_yticks(range(len(top_20_builtin)))
#             axes[1].set_yticklabels(top_20_builtin['feature'])
#             axes[1].set_xlabel('Importance Built-in')
#             axes[1].set_title('Random Forest Feature Importance - Top 20')
#             axes[1].invert_yaxis()
#             axes[1].grid(True, alpha=0.3, axis='x')
#         else:
#             axes[1].text(0.5, 0.5, 'Importance built-in\nnon disponible', 
#                          ha='center', va='center', transform=axes[1].transAxes)
#             axes[1].set_title('Built-in Importance - Non disponible')
        
#         plt.tight_layout()
        
#         plot_path = os.path.join(self.plots_dir, f'feature_importance_detailed_{strategy_name}_{self.current_subset_info}.png') # Ajout du subset info
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()
    
#     def optimize_threshold_rigorous(self, y_true, y_pred_proba, strategy_name):
#         """Optimisation rigoureuse des seuils avec validation croisée"""
#         print(f"🎯 Optimisation des seuils pour {strategy_name}...")
        
#         thresholds = np.arange(0.05, 0.95, 0.01)
#         metrics_results = {
#             'threshold': [],
#             'precision': [],
#             'recall': [],
#             'f1': [],
#             'specificity': [],
#             'balanced_accuracy': [],
#             'youden_index': []
#         }
        
#         for threshold in tqdm(thresholds, desc="Test des seuils"):
#             y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
#             precision = precision_score(y_true, y_pred_thresh, zero_division=0)
#             recall = recall_score(y_true, y_pred_thresh, zero_division=0)
#             f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
#             balanced_acc = balanced_accuracy_score(y_true, y_pred_thresh)
            
#             tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
#             fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
#             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
#             youden = recall + specificity - 1
            
#             metrics_results['threshold'].append(threshold)
#             metrics_results['precision'].append(precision)
#             metrics_results['recall'].append(recall)
#             metrics_results['f1'].append(f1)
#             metrics_results['specificity'].append(specificity)
#             metrics_results['balanced_accuracy'].append(balanced_acc)
#             metrics_results['youden_index'].append(youden)
        
#         results_df = pd.DataFrame(metrics_results)
        
#         optimal_thresholds = {
#             'f1': {
#                 'threshold': results_df.loc[results_df['f1'].idxmax(), 'threshold'],
#                 'score': results_df['f1'].max()
#             },
#             'balanced_accuracy': {
#                 'threshold': results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold'],
#                 'score': results_df['balanced_accuracy'].max()
#             },
#             'youden': {
#                 'threshold': results_df.loc[results_df['youden_index'].idxmax(), 'threshold'],
#                 'score': results_df['youden_index'].max()
#             },
#             'precision': {
#                 'threshold': results_df.loc[results_df['precision'].idxmax(), 'threshold'],
#                 'score': results_df['precision'].max()
#             },
#             'recall': {
#                 'threshold': results_df.loc[results_df['recall'].idxmax(), 'threshold'],
#                 'score': results_df['recall'].max()
#             }
#         }
        
#         results_df.to_csv(
#             os.path.join(self.metrics_dir, f'threshold_analysis_{strategy_name}_{self.current_subset_info}.csv'), # Ajout du subset info
#             index=False
#         )
        
#         return optimal_thresholds, results_df
    
#     def cross_validate_strategy(self, X, y, pipeline, strategy_name, cv_folds=5):
#         """Validation croisée rigoureuse d'une stratégie"""
#         print(f"🔄 Validation croisée pour {strategy_name}...")
        
#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
#         scoring = {
#             'roc_auc': 'roc_auc',
#             'precision': make_scorer(precision_score, zero_division=0),
#             'recall': make_scorer(recall_score, zero_division=0),
#             'f1': make_scorer(f1_score, zero_division=0),
#             'balanced_accuracy': 'balanced_accuracy'
#         }
        
#         cv_results = {}
#         for metric_name, scorer in scoring.items():
#             scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=2)
#             cv_results[metric_name] = {
#                 'mean': scores.mean(),
#                 'std': scores.std(),
#                 'scores': scores.tolist()
#             }
#             print(f"   {metric_name}: {scores.mean():.3f} (±{scores.std():.3f})")
        
#         return cv_results
    
#     def holdout_validation(self, df_rf, features, target, patient_id_col='pseudo_provisoire', test_size=0.15, val_size=0.15):
#         """Division rigoureuse Train/Validation/Test basée sur les patients"""
#         print("📊 Division Train/Validation/Test par patients...")

#         unique_patients = df_rf[patient_id_col].unique()
        
#         train_val_patients, holdout_patients = train_test_split(
#             unique_patients, test_size=test_size, random_state=42
#         )
        
#         val_size_adjusted = val_size / (1 - test_size)
#         train_patients, val_patients = train_test_split(
#             train_val_patients, test_size=val_size_adjusted, random_state=42
#         )
        
#         df_train = df_rf[df_rf[patient_id_col].isin(train_patients)]
#         df_val = df_rf[df_rf[patient_id_col].isin(val_patients)]
#         df_holdout = df_rf[df_rf[patient_id_col].isin(holdout_patients)]
        
#         print(f"   Train: {len(df_train)} lignes, {len(train_patients)} patients")
#         print(f"   Val: {len(df_val)} lignes, {len(val_patients)} patients")
#         print(f"   Holdout: {len(df_holdout)} lignes, {len(holdout_patients)} patients")

#         split_data = {
#             'train_patients': train_patients.tolist(),
#             'val_patients': val_patients.tolist(),
#             'holdout_patients': holdout_patients.tolist(),
#         }

#         with open(os.path.join(self.data_dir, f'data_splits_{self.current_subset_info}.json'), 'w') as f: # Ajout du subset info
#             json.dump(split_data, f, indent=2)

#         X_train = df_train[features]
#         y_train = df_train[target]
#         X_val = df_val[features]
#         y_val = df_val[target]
#         X_holdout = df_holdout[features]
#         y_holdout = df_holdout[target]

#         return X_train, X_val, X_holdout, y_train, y_val, y_holdout

    
#     def save_strategy_results(self, strategy_name, results):
#         """Sauvegarde les résultats d'une stratégie individuelle, incluant l'info du subset."""
#         strategy_dir = os.path.join(self.output_dir, f"strategy_{strategy_name}")
#         os.makedirs(strategy_dir, exist_ok=True)
        
#         pipeline_path = os.path.join(self.models_dir, f'pipeline_{strategy_name}_{self.current_subset_info}.pkl') # Ajout du subset info
#         joblib.dump(results['pipeline'], pipeline_path) # Use joblib.dump
        
#         metrics_path = os.path.join(self.metrics_dir, f'metrics_{strategy_name}_{self.current_subset_info}.json') # Ajout du subset info
#         metrics_data = {
#             'cv_results': results['cv_results'],
#             'validation_metrics': results['validation_metrics'],
#             'optimal_thresholds': results['optimal_thresholds'],
#             'feature_importance': results.get('feature_importance', {}),
#             'timestamp': dt.now().isoformat(),
#             'subset_info': self.current_subset_info # Ajouter l'info du subset ici
#         }
        
#         with open(metrics_path, 'w') as f:
#             json.dump(metrics_data, f, indent=2, default=str)
        
#         print(f"💾 Résultats de {strategy_name} pour subset '{self.current_subset_info}' sauvegardés")
    
#     def evaluate_all_strategies(self, df_rf, features, target='etat_critique', 
#                                 resume_experiment_id=None, output_dir=None,
#                                 subset_col=None, subset_values=None):
#         """
#         Évaluation complète et rigoureuse de toutes les stratégies avec persistance,
#         avec la possibilité d'exécuter sur des sous-groupes de données.
#         """
        
#         print("🚀 DÉBUT DE L'ÉVALUATION RIGOUREUSE")
#         print("="*60)

#         # Définir les subsets à traiter
#         if subset_col and subset_values:
#             subsets_to_process = {value: df_rf[df_rf[subset_col] == value] for value in subset_values}
#         elif subset_col:
#             subsets_to_process = {value: df_rf[df_rf[subset_col] == value] for value in df_rf[subset_col].unique()}
#         else:
#             subsets_to_process = {"all_data": df_rf}

#         all_overall_results = {} # Pour stocker les résultats de tous les subsets

#         for subset_name, subset_df in subsets_to_process.items():
#             self.current_subset_info = subset_name
#             print(f"\n{'#'*70}")
#             print(f"## Démarrage de l'évaluation pour le subset: '{subset_name}' ({len(subset_df)} lignes) ##")
#             print(f"{'#'*70}")

#             if subset_df.empty:
#                 print(f"⚠️ Le subset '{subset_name}' est vide, passage au suivant.")
#                 continue

#             # Créer ou reprendre le répertoire de sortie pour ce subset
#             # Si resume_experiment_id est fourni, on le passe pour reprendre, sinon on crée un nouveau dossier par subset
#             current_output_base = output_dir if output_dir else self.base_output_path
#             self.create_output_directory(resume_experiment_id=resume_experiment_id, subset_name=subset_name)

#             if resume_experiment_id:
#                 self.load_experiment_state()
            
#             X = subset_df[features]
#             numerical_features = X.select_dtypes(include=['int64', 'float64','float32','int32']).columns.tolist()
#             categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
#             print(f"Features numériques: {len(numerical_features)}")
#             print(f"Features catégorielles: {len(categorical_features)}")
            
#             X_train, X_val, X_holdout, y_train, y_val, y_holdout = self.holdout_validation(
#                 subset_df, features, target
#             )
            
#             strategies = self.prepare_resampling_strategies()
            
#             all_results_for_subset = self.load_existing_results() # Charger les résultats spécifiques à ce subset
            
#             for strategy_name, strategy in strategies.items():
                
#                 if strategy_name in self.completed_strategies:
#                     print(f"\n⏭️ {strategy_name} déjà complétée pour le subset '{subset_name}', passage à la suivante...")
#                     continue
                
#                 print(f"\n{'='*60}")
#                 print(f"🧪 ÉVALUATION DE LA STRATÉGIE: {strategy_name.upper()} pour subset '{subset_name}'")
#                 print(f"{'='*60}")
                
#                 try:
#                     self.save_experiment_state(all_results_for_subset, strategy_name)
                    
#                     pipeline = self.create_pipeline(numerical_features, categorical_features, strategy_name, strategy)
                    
#                     cv_results = self.cross_validate_strategy(X_train, y_train, pipeline, strategy_name)
                    
#                     print(f"🏋️ Entraînement du modèle {strategy_name} pour subset '{subset_name}'...")
                    
#                     if hasattr(strategy, 'fit_resample') and strategy is not None:
#                         preprocessor = pipeline.named_steps['preprocessor']
#                         X_train_processed = preprocessor.fit_transform(X_train)
#                         X_train_resampled, y_train_resampled = strategy.fit_resample(X_train_processed, y_train)
                        
#                         classifier = pipeline.named_steps['classifier']
#                         classifier.fit(X_train_resampled, y_train_resampled)
                        
#                         X_val_processed = preprocessor.transform(X_val)
#                         y_val_pred_proba = classifier.predict_proba(X_val_processed)[:, 1]
#                         y_val_pred = classifier.predict(X_val_processed)
#                     else:
#                         pipeline.fit(X_train, y_train)
#                         y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
#                         y_val_pred = pipeline.predict(X_val)
                    
#                     val_metrics = {
#                         'roc_auc': roc_auc_score(y_val, y_val_pred_proba),
#                         'avg_precision': average_precision_score(y_val, y_val_pred_proba),
#                         'precision': precision_score(y_val, y_val_pred, zero_division=0),
#                         'recall': recall_score(y_val, y_val_pred, zero_division=0),
#                         'f1': f1_score(y_val, y_val_pred, zero_division=0),
#                         'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
#                         'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
#                     }
                    
#                     optimal_thresholds, threshold_df = self.optimize_threshold_rigorous(
#                         y_val, y_val_pred_proba, strategy_name
#                     )
                    
#                     feature_importance = self.calculate_feature_importance(
#                         pipeline, X_val, y_val, strategy_name, features
#                     )
                    
#                     self.plot_comprehensive_analysis(
#                         strategy_name, y_val, y_val_pred, y_val_pred_proba,
#                         feature_importance, threshold_df, features
#                     )
                    
#                     strategy_results = {
#                         'cv_results': cv_results,
#                         'validation_metrics': val_metrics,
#                         'optimal_thresholds': optimal_thresholds,
#                         'feature_importance': feature_importance,
#                         'pipeline': pipeline,
#                         'threshold_analysis': threshold_df.to_dict('records')
#                     }
#                     all_results_for_subset[strategy_name] = strategy_results
#                     self.save_strategy_results(strategy_name, strategy_results)
#                     self.completed_strategies.add(strategy_name)
#                     print(f"✅ {strategy_name} terminé avec succès pour subset '{subset_name}'")
#                     print(f" ROC-AUC: {val_metrics['roc_auc']:.3f}")
#                     print(f" F1-Score: {val_metrics['f1']:.3f}")
#                     print(f" Balanced Accuracy: {val_metrics['balanced_accuracy']:.3f}")
#                     self.save_experiment_state(all_results_for_subset)
#                 except Exception as e:
#                     print(f"❌ Erreur avec {strategy_name} pour subset '{subset_name}': {str(e)}")
#                     error_info = {
#                         'strategy': strategy_name,
#                         'subset': subset_name, # Ajouter le subset à l'erreur
#                         'error': str(e),
#                         'timestamp': dt.now().isoformat()
#                     }
#                     error_file = os.path.join(self.output_dir, 'errors.json')
#                     if os.path.exists(error_file):
#                         with open(error_file, 'r') as f:
#                             errors = json.load(f)
#                     else:
#                         errors = []
#                     errors.append(error_info)
#                     with open(error_file, 'w') as f:
#                         json.dump(errors, f, indent=2)
#                     continue
            
#             # Après avoir traité toutes les stratégies pour un subset, analyser et sauvegarder les résultats globaux de ce subset
#             comparison_df, holdout_metrics = self.analyze_and_compare_results(all_results_for_subset, X_holdout, y_holdout)
#             self.save_comprehensive_results(all_results_for_subset, features, target, comparison_df, holdout_metrics)
#             self.create_final_report(all_results_for_subset, comparison_df, holdout_metrics, features, target)

#             all_overall_results[subset_name] = { # Stocker les résultats agrégés pour ce subset
#                 'comparison_df': comparison_df.to_dict('records'),
#                 'holdout_metrics': holdout_metrics
#             }
#             # Réinitialiser les stratégies complétées pour le prochain subset si ce n'est pas une reprise
#             self.completed_strategies = set()
#             print(f"\n### Fin de l'évaluation pour le subset: '{subset_name}' ###")

#         print(f"\n{'='*60}")
#         print("✅ ÉVALUATION RIGOUREUSE COMPLÉTÉE POUR TOUS LES SUBSETS")
#         print(f"{'='*60}")
#         return all_overall_results

#     def analyze_and_compare_results(self, all_results, X_holdout, y_holdout):
#         """Analyse comparative de toutes les stratégies"""
#         print(f"\n{'='*60}")
#         print(f"📊 ANALYSE COMPARATIVE DES RÉSULTATS (Subset: {self.current_subset_info})") # Ajout du subset info
#         print(f"{'='*60}")
        
#         comparison_data = []
#         for strategy_name, results in all_results.items():
#             if 'cv_results' in results:
#                 cv_results = results['cv_results']
#                 val_metrics = results['validation_metrics']
#                 comparison_data.append({
#                     'Strategy': strategy_name,
#                     'CV_ROC_AUC_Mean': cv_results['roc_auc']['mean'],
#                     'CV_ROC_AUC_Std': cv_results['roc_auc']['std'],
#                     'CV_F1_Mean': cv_results['f1']['mean'],
#                     'CV_F1_Std': cv_results['f1']['std'],
#                     'CV_Precision_Mean': cv_results['precision']['mean'],
#                     'CV_Recall_Mean': cv_results['recall']['mean'],
#                     'CV_Balanced_Acc_Mean': cv_results['balanced_accuracy']['mean'],
#                     'Val_ROC_AUC': val_metrics['roc_auc'],
#                     'Val_F1': val_metrics['f1'],
#                     'Val_Balanced_Acc': val_metrics['balanced_accuracy'],
#                     'Val_Precision': val_metrics['precision'],
#                     'Val_Recall': val_metrics['recall'],
#                     'Val_Avg_Precision': val_metrics['avg_precision']
#                 })
        
#         comparison_df = pd.DataFrame(comparison_data)

#         if not comparison_df.empty:
#             # MODIFICATION ICI: Trier par Recall, puis F1-Score, puis ROC-AUC
#             comparison_df = comparison_df.sort_values(
#                 ['Val_Recall', 'Val_F1', 'Val_ROC_AUC'], 
#                 ascending=[False, False, False]
#             )
#             print("\n🏆 CLASSEMENT DES STRATÉGIES (par Rappel, F1-Score, puis ROC-AUC sur validation):")
#             print(comparison_df[['Strategy', 'Val_Recall', 'Val_F1', 'Val_ROC_AUC', 'Val_Balanced_Acc', 'Val_Precision']].round(3))
#         else:
#             print("Aucun résultat à comparer.")

#         if not comparison_df.empty:
#             comparison_df.to_csv(os.path.join(self.metrics_dir, f'strategy_comparison_detailed_{self.current_subset_info}.csv'), index=False) # Ajout du subset info
        
#         self.plot_strategy_comparison(comparison_df)
        
#         top_3_strategies = comparison_df.head(3)['Strategy'].tolist() if not comparison_df.empty else []

#         holdout_results = {}
#         print(f"\n🎯 ÉVALUATION FINALE SUR HOLDOUT - TOP {len(top_3_strategies)} STRATÉGIES (Subset: {self.current_subset_info})") # Ajout du subset info
#         for strategy_name in top_3_strategies:
#             if strategy_name in all_results and 'pipeline' in all_results[strategy_name]:
#                 pipeline = all_results[strategy_name]['pipeline']
#                 print(f"\n🧪 Test final de {strategy_name} sur données holdout pour subset '{self.current_subset_info}'...")
                
#                 # S'assurer que le préprocesseur et le classifieur sont dans le bon état pour la prédiction
#                 # Le pipeline doit déjà être ajusté ici
#                 if hasattr(pipeline.named_steps.get('resampler'), 'fit_resample'):
#                     # Si c'est un pipeline avec resampler, on doit faire passer X_holdout par le preprocessor
#                     # avant de le passer au classifier (qui est déjà entraîné)
#                     preprocessor_holdout = pipeline.named_steps['preprocessor']
#                     X_holdout_processed = preprocessor_holdout.transform(X_holdout)
#                     classifier_holdout = pipeline.named_steps['classifier']
#                     y_holdout_pred_proba = classifier_holdout.predict_proba(X_holdout_processed)[:, 1]
#                     y_holdout_pred = classifier_holdout.predict(X_holdout_processed)
#                 else:
#                     # Si c'est un pipeline standard (baseline, class_weight, balanced_rf),
#                     # la prédiction se fait directement sur le pipeline complet
#                     y_holdout_pred_proba = pipeline.predict_proba(X_holdout)[:, 1]
#                     y_holdout_pred = pipeline.predict(X_holdout)


#                 holdout_metrics = {
#                     'roc_auc': roc_auc_score(y_holdout, y_holdout_pred_proba),
#                     'avg_precision': average_precision_score(y_holdout, y_holdout_pred_proba),
#                     'precision': precision_score(y_holdout, y_holdout_pred, zero_division=0),
#                     'recall': recall_score(y_holdout, y_holdout_pred, zero_division=0),
#                     'f1': f1_score(y_holdout, y_holdout_pred, zero_division=0),
#                     'balanced_accuracy': balanced_accuracy_score(y_holdout, y_holdout_pred),
#                     'classification_report': classification_report(y_holdout, y_holdout_pred, output_dict=True)
#                 }
#                 holdout_results[strategy_name] = holdout_metrics
#                 print(f"📊 RÉSULTATS HOLDOUT - {strategy_name} (Subset: {self.current_subset_info}):") # Ajout du subset info
#                 for metric, value in holdout_metrics.items():
#                     if metric != 'classification_report':
#                         print(f"   {metric}: {value:.3f}")
#                 self.plot_holdout_analysis(strategy_name, y_holdout, y_holdout_pred, y_holdout_pred_proba)

#         if holdout_results:
#             with open(os.path.join(self.metrics_dir, f'holdout_results_{self.current_subset_info}.json'), 'w') as f: # Ajout du subset info
#                 json.dump(holdout_results, f, indent=2, default=str)
        
#         return comparison_df, holdout_results

#     def plot_holdout_analysis(self, strategy_name, y_true, y_pred, y_pred_proba):
#         """Crée un graphique d'analyse simple pour les résultats holdout."""
#         plt.rcParams['figure.figsize'] = (10, 6)
#         fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#         fig.suptitle(f'Analyse Holdout - {strategy_name} (Subset: {self.current_subset_info})', fontsize=16, fontweight='bold')

#         # Matrice de confusion
#         cm = confusion_matrix(y_true, y_pred)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
#         axes[0].set_title('Matrice de Confusion (Holdout)')
#         axes[0].set_xlabel('Prédiction')
#         axes[0].set_ylabel('Réalité')

#         # Courbe ROC
#         fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#         roc_auc = roc_auc_score(y_true, y_pred_proba)
#         axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
#         axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         axes[1].set_xlim([0.0, 1.0])
#         axes[1].set_ylim([0.0, 1.05])
#         axes[1].set_xlabel('Taux de Faux Positifs')
#         axes[1].set_ylabel('Taux de Vrais Positifs')
#         axes[1].set_title('Courbe ROC (Holdout)')
#         axes[1].legend(loc="lower right")

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plot_path = os.path.join(self.plots_dir, f'holdout_analysis_{strategy_name}_{self.current_subset_info}.png')
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()

#     def plot_strategy_comparison(self, comparison_df):
#         """Crée des graphiques de comparaison entre stratégies"""
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         fig.suptitle(f'Comparaison des Stratégies de Rééquilibrage (Subset: {self.current_subset_info})', fontsize=16, fontweight='bold') # Ajout du subset info

#         strategies = comparison_df['Strategy']
        
#         # Plot ROC-AUC
#         axes[0,0].bar(strategies, comparison_df['Val_ROC_AUC'], color='skyblue', alpha=0.7)
#         axes[0,0].set_title('ROC-AUC sur Validation')
#         axes[0,0].set_ylabel('ROC-AUC')
#         axes[0,0].tick_params(axis='x', rotation=45)
#         axes[0,0].grid(True, alpha=0.3)

#         # Plot F1-Score
#         axes[0,1].bar(strategies, comparison_df['Val_F1'], color='lightgreen', alpha=0.7)
#         axes[0,1].set_title('F1-Score sur Validation')
#         axes[0,1].set_ylabel('F1-Score')
#         axes[0,1].tick_params(axis='x', rotation=45)
#         axes[0,1].grid(True, alpha=0.3)

#         # Plot Balanced Accuracy
#         axes[1,0].bar(strategies, comparison_df['Val_Balanced_Acc'], color='orange', alpha=0.7)
#         axes[1,0].set_title('Balanced Accuracy sur Validation')
#         axes[1,0].set_ylabel('Balanced Accuracy')
#         axes[1,0].tick_params(axis='x', rotation=45)
#         axes[1,0].grid(True, alpha=0.3)

#         # Plot Precision vs Recall
#         # Utilise la taille pour indiquer le F1-score et la couleur pour le ROC-AUC
#         s_values = comparison_df['Val_F1'] * 500 # Scale F1 for marker size
#         scatter = axes[1,1].scatter(comparison_df['Val_Recall'], comparison_df['Val_Precision'], 
#                                     s=s_values, alpha=0.7, 
#                                     c=comparison_df['Val_ROC_AUC'], cmap='viridis', edgecolors='w', linewidth=0.5)
        
#         # Ajouter les labels de stratégie
#         for i, strategy in enumerate(strategies):
#             axes[1,1].annotate(strategy, 
#                                (comparison_df.iloc[i]['Val_Recall'], comparison_df.iloc[i]['Val_Precision']), 
#                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        
#         axes[1,1].set_xlabel('Recall')
#         axes[1,1].set_ylabel('Precision')
#         axes[1,1].set_title('Precision vs Recall (taille=F1, couleur=ROC-AUC)')
#         axes[1,1].grid(True, alpha=0.3)
        
#         # Ajouter une barre de couleur pour le ROC-AUC
#         cbar = fig.colorbar(scatter, ax=axes[1,1], orientation='vertical', fraction=0.046, pad=0.04)
#         cbar.set_label('ROC-AUC')

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le titre global
#         plot_path = os.path.join(self.plots_dir, f'strategy_comparison_{self.current_subset_info}.png') # Ajout du subset info
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         self.plot_radar_comparison(comparison_df)

#     def plot_radar_comparison(self, comparison_df):
#         """Crée un graphique radar pour comparer les top 5 stratégies"""
#         top_5 = comparison_df.head(5) # Prendre le top 5 selon le nouveau tri
        
#         # Utiliser les métriques les plus pertinentes pour le radar
#         metrics = ['Val_Recall', 'Val_F1', 'Val_Balanced_Acc', 'Val_Precision', 'Val_ROC_AUC']
#         metric_labels = ['Recall', 'F1-Score', 'Balanced Acc', 'Precision', 'ROC-AUC']
        
#         # Normaliser les métriques si elles sont sur des échelles très différentes
#         # Pour les métriques de 0 à 1, ce n'est pas strictement nécessaire, mais c'est une bonne pratique
#         df_normalized = top_5[metrics].copy()
#         for col in metrics:
#             # Simple min-max scaling si nécessaire
#             min_val = df_normalized[col].min()
#             max_val = df_normalized[col].max()
#             if max_val > min_val: # Avoid division by zero if all values are the same
#                 df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
#             else:
#                 df_normalized[col] = 0.5 # Or some other neutral value if all are same

#         fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
#         angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#         angles += angles[:1] # Compléter le cercle

#         # Plot chaque stratégie
#         for i, strategy_name in enumerate(top_5['Strategy']):
#             values = df_normalized.iloc[i][metrics].tolist()
#             values += values[:1] # Compléter le cercle pour le tracé
#             ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=strategy_name)
#             ax.fill(angles, values, alpha=0.25)

#         ax.set_theta_offset(np.pi / 2)
#         ax.set_theta_direction(-1)
        
#         ax.set_rlabel_position(0) # Position des labels sur l'axe radial
#         plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
#         ax.set_ylim(0, 1)

#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(metric_labels)

#         ax.set_title(f'Comparaison Radar des Top 5 Stratégies (Subset: {self.current_subset_info})', 
#                      fontsize=16, fontweight='bold', va='bottom', pad=20)
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

#         plt.tight_layout()
#         plot_path = os.path.join(self.plots_dir, f'strategy_comparison_radar_{self.current_subset_info}.png') # Ajout du subset info
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()

#     def save_comprehensive_results(self, all_results_for_subset, features, target, comparison_df, holdout_metrics):
#         """Sauvegarde tous les résultats agrégés dans un fichier JSON."""
#         comprehensive_results = {
#             'timestamp': dt.now().isoformat(),
#             'subset_info': self.current_subset_info,
#             'features_used': features,
#             'target_column': target,
#             'detailed_results': {
#                 s_name: {
#                     'cv_results': res['cv_results'],
#                     'validation_metrics': res['validation_metrics'],
#                     'optimal_thresholds': res['optimal_thresholds'],
#                     'feature_importance': res['feature_importance'],
#                     'threshold_analysis': res['threshold_analysis'] # Inclure l'analyse des seuils
#                 } for s_name, res in all_results_for_subset.items() if 'pipeline' in res # Exclure le pipeline pour la sauvegarde JSON
#             },
#             'strategy_comparison': comparison_df.to_dict('records'),
#             'holdout_results': holdout_metrics
#         }
        
#         results_file = os.path.join(self.output_dir, f"comprehensive_experiment_results_{self.current_subset_info}.json") # Ajout du subset info
#         with open(results_file, 'w', encoding='utf-8') as f:
#             json.dump(comprehensive_results, f, indent=2, default=str)
#         print(f"📄 Résultats complets de l'expérience sauvegardés dans {results_file}")

#     def create_final_report(self, all_results_for_subset, comparison_df, holdout_metrics, features, target):
#         """Génère un rapport final markdown ou texte."""
#         report_path = os.path.join(self.output_dir, f"experiment_report_{self.current_subset_info}.md") # Ajout du subset info
#         print(f"\n📝 Génération du rapport final pour subset '{self.current_subset_info}' dans {report_path}...")

#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(f"# Rapport d'Expérience ML - Random Forest (Subset: {self.current_subset_info})\n\n") # Ajout du subset info
#             f.write(f"Date de l'exécution : {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Répertoire de sortie : {self.output_dir}\n")
#             f.write(f"Variable cible : `{target}`\n")
#             f.write(f"Features utilisées : {', '.join(features)}\n\n")

#             f.write("## 📈 Vue d'ensemble des Stratégies (Validation Croisée & Validation)\n\n")
#             if not comparison_df.empty:
#                 f.write("Le tableau ci-dessous présente les performances moyennes des différentes stratégies de rééquilibrage sur les ensembles de validation (validation croisée et ensemble de validation dédié).\n\n")
#                 f.write(comparison_df[['Strategy', 'Val_Recall', 'Val_F1', 'Val_ROC_AUC', 'Val_Balanced_Acc', 'Val_Precision', 'Val_Avg_Precision']].round(3).to_markdown(index=False))
#                 f.write("\n\n")
#                 f.write("Le classement est effectué en priorisant le **Rappel (Recall)**, puis le **F1-Score**, et enfin l'**AUC ROC** sur l'ensemble de validation.\n\n")
#             else:
#                 f.write("Aucune donnée de comparaison de stratégie disponible.\n\n")

#             f.write("## 🚀 Résultats sur l'Ensemble Holdout (Top Stratégies)\n\n")
#             if holdout_metrics:
#                 f.write("Les stratégies les plus performantes sur l'ensemble de validation ont été évaluées sur un ensemble de test indépendant (holdout) pour une évaluation finale impartiale.\n\n")
#                 for strategy_name, metrics in holdout_metrics.items():
#                     f.write(f"### Stratégie : {strategy_name}\n\n")
#                     f.write(f"- ROC-AUC : {metrics['roc_auc']:.3f}\n")
#                     f.write(f"- Average Precision : {metrics['avg_precision']:.3f}\n")
#                     f.write(f"- Precision : {metrics['precision']:.3f}\n")
#                     f.write(f"- Recall : {metrics['recall']:.3f}\n")
#                     f.write(f"- F1-Score : {metrics['f1']:.3f}\n")
#                     f.write(f"- Balanced Accuracy : {metrics['balanced_accuracy']:.3f}\n")
#                     f.write("\n")
#                     f.write("#### Rapport de Classification :\n")
#                     f.write("```\n")
#                     # Formater le classification_report pour qu'il soit lisible dans markdown
#                     report_str = ""
#                     for label, class_metrics in metrics['classification_report'].items():
#                         if isinstance(class_metrics, dict):
#                             report_str += f"Class {label}:\n"
#                             for metric_name, value in class_metrics.items():
#                                 if isinstance(value, float):
#                                     report_str += f"  {metric_name}: {value:.3f}\n"
#                                 else:
#                                     report_str += f"  {metric_name}: {value}\n"
#                         else: # for accuracy, macro avg, weighted avg
#                             report_str += f"{label}: {class_metrics:.3f}\n" if isinstance(class_metrics, float) else f"{label}: {class_metrics}\n"
#                     f.write(report_str)
#                     f.write("```\n\n")
#             else:
#                 f.write("Aucun résultat sur l'ensemble holdout disponible.\n\n")

#             f.write("## 🖼️ Graphiques et Visualisations Clés\n\n")
#             f.write("Tous les graphiques générés sont disponibles dans le sous-répertoire `plots/`.\n\n")
#             f.write(f"Voici les liens directs vers les graphiques de comparaison généraux pour le subset '{self.current_subset_info}' :\n")
            
#             # Vérifier l'existence des fichiers avant de les lister
#             comparison_plot_name = f'strategy_comparison_{self.current_subset_info}.png'
#             radar_plot_name = f'strategy_comparison_radar_{self.current_subset_info}.png'
            
#             comparison_plot_path = os.path.join(self.plots_dir, comparison_plot_name)
#             radar_plot_path = os.path.join(self.plots_dir, radar_plot_name)

#             if os.path.exists(comparison_plot_path):
#                 f.write(f"- Comparaison des stratégies (barres) : [Lien]({os.path.join('plots', comparison_plot_name)})\n")
#             if os.path.exists(radar_plot_path):
#                 f.write(f"- Comparaison radar des Top 5 stratégies : [Lien]({os.path.join('plots', radar_plot_name)})\n")
            
#             # Liens vers les analyses complètes et importances des features des top stratégies
#             if holdout_metrics:
#                 f.write("\nEt des exemples pour les top stratégies :\n")
#                 for strategy_name in holdout_metrics.keys(): # Keys sont les top stratégies évaluées sur holdout
#                     analysis_plot_name = f'analysis_complete_{strategy_name}_{self.current_subset_info}.png'
#                     feature_importance_plot_name = f'feature_importance_detailed_{strategy_name}_{self.current_subset_info}.png'
                    
#                     analysis_plot_path = os.path.join(self.plots_dir, analysis_plot_name)
#                     feature_importance_plot_path = os.path.join(self.plots_dir, feature_importance_plot_name)

#                     f.write(f"### {strategy_name}\n")
#                     if os.path.exists(analysis_plot_path):
#                         f.write(f"- Analyse complète (Matrice de Confusion, ROC, PR, etc.) : [Lien]({os.path.join('plots', analysis_plot_name)})\n")
#                     if os.path.exists(feature_importance_plot_path):
#                         f.write(f"- Importance des Features : [Lien]({os.path.join('plots', feature_importance_plot_name)})\n")
#                     f.write("\n")
#             else:
#                 f.write("\nAucun graphique détaillé pour les stratégies individuelles car aucun résultat holdout n'a été produit.\n")
            
#             f.write("## 📂 Fichiers de Sortie Générés\n\n")
#             f.write("Tous les fichiers de sortie sont stockés dans le répertoire de l'expérience et ses sous-répertoires :\n")
#             f.write(f"- `data_splits_{self.current_subset_info}.json` : Informations sur la division Train/Validation/Holdout par patients\n")
#             f.write(f"- `comprehensive_experiment_results_{self.current_subset_info}.json` : Résultats JSON complets de l'expérience\n")
#             f.write(f"- `experiment_metadata_{self.current_subset_info}.json` : Métadonnées complètes de l'expérience\n")
#             f.write(f"- `metrics_*.json` : Métriques détaillées pour chaque stratégie et subset\n")
#             f.write(f"- `threshold_analysis_*.csv` : Comparaison des seuils pour chaque stratégie et subset\n")
#             f.write(f"- `strategy_comparison_detailed_{self.current_subset_info}.csv` : Comparaison agrégée des stratégies\n")
#             f.write(f"- `holdout_results_{self.current_subset_info}.json` : Résultats détaillés sur l'ensemble holdout\n")
#             f.write(f"- `plots/` : Répertoire contenant tous les graphiques (matrice de confusion, ROC, PR, importance des features, comparaisons)\n\n")
            
#             f.write("## 💡 Recommandations Cliniques\n\n")
#             f.write("### Choix du Seuil selon l'Usage :\n\n")
#             f.write("- **Pour minimiser les faux négatifs (ne pas rater de décès)** : Utiliser le seuil optimisé pour le rappel\n")
#             f.write("- **Pour minimiser les faux positifs (éviter les alertes inutiles)** : Utiliser le seuil optimisé pour la précision\n")
#             f.write("- **Pour un équilibre général** : Utiliser le seuil optimisé pour le F1-Score ou la Balanced Accuracy\n")
#             # else:
#             #     f.write("Aucun résultat d'expérience n'a été généré pour ce subset. Veuillez vérifier l'exécution.")
#             print("✅ Rapport final généré avec succès.")

#-------------------------------------------------------------------------------------------------------------------------


# rf_1.txt
import optuna
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.utils import resample
from tqdm import tqdm
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, accuracy_score,
                           roc_auc_score, precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score, balanced_accuracy_score, make_scorer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (SMOTE, RandomOverSampler)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier # Import BalancedRandomForestClassifier
import json
import joblib # Import joblib for saving models
from sklearn.inspection import permutation_importance
import copy
import traceback

class EnhancedRigorousMLPipeline:
    def __init__(self, df, features, target, categorical_features: list, patient_id_col=None, 
                 output_path="C:/experiments/", experiment_name="rigorous_ml",
                 random_state=42, test_size=0.2, scoring='f1',
                 param_grid_base=None, n_trials=10):
        
        self.df = df
        self.features = [f for f in features if f != patient_id_col]
        self.target = target
        self.categorical_features = categorical_features
        self.patient_id_col = patient_id_col
        self.random_state = random_state
        self.test_size = test_size
        self.scoring = scoring
        self.param_grid_base = param_grid_base if param_grid_base is not None else self._default_param_grid()
        self.n_trials = n_trials
        
        self.base_output_path = output_path
        self.experiment_name = experiment_name
        self.experiment_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.base_output_path, self.experiment_name, self.experiment_timestamp)
        self.plots_base_dir = os.path.join(self.output_dir, "plots")
        self.models_base_dir = os.path.join(self.output_dir, "models")
        self.feature_importance_dir = os.path.join(self.output_dir, "feature_importance")

        os.makedirs(self.plots_base_dir, exist_ok=True)
        os.makedirs(self.models_base_dir, exist_ok=True)
        os.makedirs(self.feature_importance_dir, exist_ok=True)
        
        self.results = {}
        self.models = {}
        self.feature_importances = {}
        self.current_subset_info = "global"

        self.numerical_features = [f for f in self.features if f not in self.categorical_features]
        self.preprocessor = self._create_preprocessor()
        
        self.report_path = os.path.join(self.output_dir, f"experiment_report_{self.experiment_timestamp}.md")
        self.metadata = {
            "experiment_start_time": self.experiment_timestamp,
            "target": self.target,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "patient_id_column": self.patient_id_col,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "scoring_metric": self.scoring,
            "optuna_n_trials": self.n_trials
        }
        self.all_metrics = {} # Initialize dictionary to store all metrics for all strategies/subsets
        self._write_report_header()

    def _create_preprocessor(self):
        numerical_transformer = SklearnPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = SklearnPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor_transformers = []
        if self.numerical_features:
            preprocessor_transformers.append(('num', numerical_transformer, self.numerical_features))
        if self.categorical_features:
            preprocessor_transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        return ColumnTransformer(
            transformers=preprocessor_transformers,
            remainder='passthrough'
        )
    
    def _default_param_grid(self):
        # Default parameter grid for RandomForestClassifier using Optuna distributions
        return {
            'randomforestclassifier__n_estimators': optuna.distributions.IntDistribution(50, 500),
            'randomforestclassifier__max_depth': optuna.distributions.IntDistribution(3, 10),
            'randomforestclassifier__min_samples_split': optuna.distributions.IntDistribution(2, 20),
            'randomforestclassifier__min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
            'randomforestclassifier__max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', 0.5, 0.7]),
            'randomforestclassifier__class_weight': optuna.distributions.CategoricalDistribution(['balanced', 'balanced_subsample', None])
        }

    def load_data(self, X, y):
        self.X = X
        self.y = y
        self.metadata["data_shape"] = {"X": self.X.shape, "y": self.y.shape}
        print(f"Données chargées. X_shape: {self.X.shape}, y_shape: {self.y.shape}")

    def prepare_data(self):
        print("\n⚙️ Préparation des données...")
        
        # Étape 1: Gérer l'indexation par ID patient si nécessaire
        df_to_split = self.df.copy()
        if self.patient_id_col and self.patient_id_col in df_to_split.columns:
            # Utiliser l'ID patient pour diviser les données de manière robuste
            patient_ids = df_to_split[self.patient_id_col].unique()
            train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_size, random_state=self.random_state)
            
            # Filtrer le DataFrame pour obtenir les sous-ensembles d'entraînement et de test
            X_train_df = df_to_split[df_to_split[self.patient_id_col].isin(train_ids)].set_index(self.patient_id_col)
            X_test_df = df_to_split[df_to_split[self.patient_id_col].isin(test_ids)].set_index(self.patient_id_col)

            y_train = X_train_df[self.target]
            y_test = X_test_df[self.target]

            X_train_df = X_train_df[self.features]
            X_test_df = X_test_df[self.features]

        else:
            # Séparation standard si pas d'ID patient
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                df_to_split[self.features], df_to_split[self.target], 
                test_size=self.test_size, random_state=self.random_state, stratify=df_to_split[self.target]
            )

        # Étape 2: Prétraitement des données
        self.preprocessor.fit(X_train_df)
        X_train_processed = self.preprocessor.transform(X_train_df)
        X_test_processed = self.preprocessor.transform(X_test_df)
        
        # Étape 3: Obtenir les noms des colonnes après le prétraitement
        try:
            self.processed_feature_names = self.preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"⚠️ Avertissement: Impossible d'obtenir les noms de features traités via get_feature_names_out. Erreur: {e}")
            self.processed_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

        # Assurer que y_test est une série avec l'index des patients pour le calcul des métriques
        self.X_train_processed = X_train_processed
        self.y_train = y_train
        self.X_test_processed = X_test_processed
        self.y_test = y_test

        print(f"✅ Données prétraitées et divisées. X_train_processed.shape: {self.X_train_processed.shape}, X_test_processed.shape: {self.X_test_processed.shape}")
        print(f"DEBUG: Nombre de features traitées: {len(self.processed_feature_names)}")
        print(f"DEBUG: Premiers noms de features traités: {self.processed_feature_names[:5]}")

        # Vérification de l'encodage One-Hot
        ohe_test = any('__' in f_name for f_name in self.processed_feature_names)
        if not ohe_test and self.categorical_features:
            print("⚠️ Avertissement: Les noms de features traités ne semblent pas contenir les suffixes OHE. Cela peut affecter l'interprétabilité des importances.")

    def _calculate_metrics(self, y_true, y_pred, y_proba):
        # Convertir les probabilités pour la classe positive (classe 1)
        if y_proba is not None and y_proba.ndim > 1:
            y_proba_positive = y_proba[:, 1]
        elif y_proba is not None:
            y_proba_positive = y_proba
        else:
            y_proba_positive = None

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='binary', zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
        
        if y_proba_positive is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba_positive)
            except ValueError:
                metrics["roc_auc"] = np.nan
            try:
                metrics["avg_precision_score"] = average_precision_score(y_true, y_proba_positive)
            except ValueError:
                metrics["avg_precision_score"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["avg_precision_score"] = np.nan

        return metrics

    def _calculate_and_store_metrics(self, strategy_name, subset_info, y_test, y_pred, y_proba):
        """
        Calcule et stocke les métriques complètes pour une stratégie et un sous-ensemble donnés.
        """
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })

        patient_ids_by_category = {}
        try:
            # Assuming y_test has an index that can be used for patient_id mapping
            y_test_series = pd.Series(y_test)
            y_pred_series = pd.Series(y_pred, index=y_test_series.index)

            tp_indices = y_test_series[(y_test_series == 1) & (y_pred_series == 1)].index.tolist()
            patient_ids_by_category['true_positives_patients_ids'] = tp_indices

            fn_indices = y_test_series[(y_test_series == 1) & (y_pred_series == 0)].index.tolist()
            patient_ids_by_category['false_negatives_patients_ids'] = fn_indices

            tn_indices = y_test_series[(y_test_series == 0) & (y_pred_series == 0)].index.tolist()
            patient_ids_by_category['true_negatives_patients_ids'] = tn_indices

            fp_indices = y_test_series[(y_test_series == 0) & (y_pred_series == 1)].index.tolist()
            patient_ids_by_category['false_positives_patients_ids'] = fp_indices

            metrics['patient_classification_breakdown'] = patient_ids_by_category
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération des IDs patients pour le stockage des métriques: {e}")

        if subset_info not in self.all_metrics:
            self.all_metrics[subset_info] = {}
        self.all_metrics[subset_info][strategy_name] = metrics
        print(f"✅ Métriques stockées pour {strategy_name} ({subset_info}).")


    def save_all_metrics(self):
        """
        Sauvegarde toutes les métriques collectées dans un fichier JSON.
        """
        metrics_filepath = os.path.join(self.output_dir, f"all_metrics_{self.experiment_timestamp}.json")
        try:
            # Convert numpy arrays/values to Python lists/types for JSON serialization
            serializable_metrics = copy.deepcopy(self.all_metrics)
            for subset_info, strategies_metrics in serializable_metrics.items():
                for strategy_name, metrics_dict in strategies_metrics.items():
                    for key, value in metrics_dict.items():
                        if isinstance(value, np.float32) or isinstance(value, np.float64):
                            metrics_dict[key] = float(value)
                        elif isinstance(value, np.int64) or isinstance(value, np.int32):
                            metrics_dict[key] = int(value)
                        elif isinstance(value, np.ndarray):
                            metrics_dict[key] = value.tolist()
            
            with open(metrics_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=4)
            print(f"💾 Toutes les métriques sauvegardées dans : {metrics_filepath}")
            self._write_report_content(f"- **Toutes les métriques sauvegardées** : [`all_metrics_{self.experiment_timestamp}.json`]({os.path.basename(metrics_filepath)})\n")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde de toutes les métriques: {e}")
            traceback.print_exc() # Print full stack trace for debugging


    def _optimize_hyperparameters(self, X, y, strategy_name):
        def objective(trial):
            # Paramètres de base pour RandomForestClassifier ou BalancedRandomForestClassifier
            classifier_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                'random_state': self.random_state,
            }

            # Handle class_weight differently based on strategy
            if strategy_name == "BalancedRandomForest":
                # BalancedRandomForestClassifier handles class imbalance internally,
                # and doesn't typeaksally use 'class_weight' directly in the same way as RF.
                # It uses 'sampling_strategy' internally.
                # So, we don't add 'class_weight' to classifier_params for BRFC.
                classifier = BalancedRandomForestClassifier(**classifier_params)
                pipeline = SklearnPipeline([('balancedrandomforestclassifier', classifier)])
            else:
                classifier_params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
                classifier = RandomForestClassifier(**classifier_params)
                if strategy_name == "NoResampling":
                    pipeline = SklearnPipeline([('randomforestclassifier', classifier)])
                else:
                    sampler = self._get_sampler(strategy_name)
                    pipeline = ImbPipeline([
                        ('sampler', sampler),
                        ('randomforestclassifier', classifier)
                    ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            scorer = make_scorer(f1_score, zero_division=0)
            
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        return study.best_params

    def _get_sampler(self, strategy_name):
        if strategy_name == "RandomOverSampler":
            return RandomOverSampler(random_state=self.random_state)
        elif strategy_name == "RandomUnderSampler":
            return RandomUnderSampler(random_state=self.random_state)
        # elif strategy_name == "SMOTE":
        #     return SMOTE(random_state=self.random_state)
        elif strategy_name == "SMOTEENN":
            return SMOTEENN(random_state=self.random_state)
        # elif strategy_name == "SMOTETomek":
        #     return SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Stratégie de rééchantillonnage inconnue : {strategy_name}")
    
    # Placeholder for test_simple_plot. Assuming it exists elsewhere or is meant to be implemented.
    def test_simple_plot(self, plot_name, custom_plots_base_dir):
        """
        Placeholder function for testing plot generation. 
        In a real scenario, this would generate a simple plot to verify Matplotlib setup.
        """
        try:
            plt.figure(figsize=(6, 4))
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Test Line')
            plt.title(f"Test Plot: {plot_name}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plot_filepath = os.path.join(custom_plots_base_dir, f"{plot_name}.png")
            plt.savefig(plot_filepath)
            plt.close()
            return plot_filepath
        except Exception as e:
            print(f"Error generating test plot: {e}")
            return None

    def _plot_roc_pr_curves_in_dir(self, y_test, y_proba_test, y_train, y_proba_train, strategy_name, custom_plots_base_dir):
        """
        Génère et sauvegarde un plot avec les courbes ROC et Precision-Recall superposées pour train et test.
        """
        plt.figure(figsize=(14, 6)) # Adjust size for two side-by-side plots

        # --- ROC Curve Subplot ---
        plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
        
        # Plot ROC for Test Set
        if y_proba_test is not None and y_proba_test.ndim > 1:
            fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test[:, 1])
            roc_auc_test = roc_auc_score(y_test, y_proba_test[:, 1])
            plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.2f})')
        else:
            plt.text(0.5, 0.6, "Test ROC curve not available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        # Plot ROC for Train Set
        if y_proba_train is not None and y_proba_train.ndim > 1:
            fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train[:, 1])
            roc_auc_train = roc_auc_score(y_train, y_proba_train[:, 1])
            plt.plot(fpr_train, tpr_train, color='red', lw=2, linestyle='--', label=f'Train ROC (AUC = {roc_auc_train:.2f})')
        else:
            plt.text(0.5, 0.4, "Train ROC curve not available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Courbe ROC ({strategy_name})')
        plt.legend(loc="lower right")
        plt.grid(True)

        # --- Precision-Recall Curve Subplot ---
        plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot

        # Plot Precision-Recall for Test Set
        if y_proba_test is not None and y_proba_test.ndim > 1:
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_proba_test[:, 1])
            avg_precision_test = average_precision_score(y_test, y_proba_test[:, 1])
            plt.plot(recall_test, precision_test, color='blue', lw=2, label=f'Test PR (AP = {avg_precision_test:.2f})')
        else:
            plt.text(0.5, 0.6, "Test Precision-Recall curve not available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        # Plot Precision-Recall for Train Set
        if y_proba_train is not None and y_proba_train.ndim > 1:
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_proba_train[:, 1])
            avg_precision_train = average_precision_score(y_train, y_proba_train[:, 1])
            plt.plot(recall_train, precision_train, color='green', lw=2, linestyle='--', label=f'Train PR (AP = {avg_precision_train:.2f})')
        else:
            plt.text(0.5, 0.4, "Train Precision-Recall curve not available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Courbe Precision-Recall ({strategy_name})')
        plt.legend(loc="lower left")
        plt.grid(True)

        plt.tight_layout()
        plot_filename = f"roc_pr_curves_{strategy_name}_{self.current_subset_info}.png"
        plot_path = os.path.join(custom_plots_base_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        return plot_path


    def _plot_confusion_matrix_in_dir(self, y_true, y_pred, strategy_name, custom_plots_base_dir):
        """Génère et sauvegarde la matrice de confusion."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Prédit 0', 'Prédit 1'], 
                        yticklabels=['Réel 0', 'Réel 1'])
            plt.xlabel('Prédiction')
            plt.ylabel('Vraie Valeur')
            plt.title(f'Matrice de Confusion - {strategy_name}')
            plt.tight_layout()
            plot_filename = f"confusion_matrix_{strategy_name}_{self.current_subset_info}.png"
            plot_path = os.path.join(custom_plots_base_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la matrice de confusion pour {strategy_name}: {e}")
            traceback.print_exc()
            return None

    def _plot_probability_distribution_in_dir(self, y_true, y_proba_positive, strategy_name, custom_plots_base_dir):
        """Génère et sauvegarde la distribution des probabilités prédites avec des bords aux histogrammes."""
        try:
            plt.figure(figsize=(10, 6))
            # Ajout de 'edgecolor' pour les bords
            sns.histplot(y_proba_positive[y_true == 0], color='blue', label='Classe 0 (Négatif)', kde=True, stat="density", linewidth=1, edgecolor='black')
            sns.histplot(y_proba_positive[y_true == 1], color='red', label='Classe 1 (Positif)', kde=True, stat="density", linewidth=1, edgecolor='black')
            plt.title(f'Distribution des Probabilités Prédites - {strategy_name}')
            plt.xlabel('Probabilité Prédite de la Classe Positive')
            plt.ylabel('Densité')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"probability_distribution_{strategy_name}_{self.current_subset_info}.png"
            plot_path = os.path.join(custom_plots_base_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la distribution de probabilités pour {strategy_name}: {e}")
            traceback.print_exc()
            return None

    def _plot_learning_curve_in_dir(self, estimator, X, y, strategy_name, custom_plots_base_dir):
        """Génère et sauvegarde la courbe d'apprentissage."""
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5),
                scoring=make_scorer(f1_score, zero_division=0)
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")

            plt.title(f"Courbe d'Apprentissage - {strategy_name}")
            plt.xlabel("Taille de l'ensemble d'entraînement")
            plt.ylabel("Score F1")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"learning_curve_{strategy_name}_{self.current_subset_info}.png"
            plot_path = os.path.join(custom_plots_base_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la courbe d'apprentissage pour {strategy_name}: {e}")
            traceback.print_exc()
            return None

    def _plot_threshold_optimization_in_dir(self, y_true, y_proba_positive, strategy_name, custom_plots_base_dir):
        """
        Génère et sauvegarde un plot d'optimisation du seuil.
        """
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba_positive)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # Add epsilon to avoid division by zero
            
            # Calculate balanced accuracy for each threshold
            balanced_accuracies = []
            for t in thresholds:
                y_pred_threshold = (y_proba_positive >= t).astype(int)
                balanced_accuracies.append(balanced_accuracy_score(y_true, y_pred_threshold))

            # Ensure that all arrays have the same length before creating DataFrame
            min_len = min(len(thresholds), len(precision), len(recall), len(f1_scores), len(balanced_accuracies))
            
            threshold_df = pd.DataFrame({
                'threshold': thresholds[:min_len],
                'precision': precision[:min_len],
                'recall': recall[:min_len],
                'f1_score': f1_scores[:min_len],
                'balanced_accuracy': balanced_accuracies[:min_len] 
            })
            
            # Find optimal F1-score threshold
            # Check if threshold_df is empty or if f1_score column has non-finite values only
            if not threshold_df.empty and threshold_df['f1_score'].dropna().any():
                optimal_f1_threshold = threshold_df['threshold'].iloc[np.argmax(threshold_df['f1_score'])]
            else:
                optimal_f1_threshold = None
                print(f"⚠️ Avertissement: Impossible de trouver un seuil F1 optimal pour {strategy_name}. La colonne 'f1_score' est vide ou contient seulement des NaN/Inf.")
            
            plt.figure(figsize=(10, 6))
            plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision')
            plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall')
            plt.plot(threshold_df['threshold'], threshold_df['f1_score'], label='F1-Score')
            plt.plot(threshold_df['threshold'], threshold_df['balanced_accuracy'], label='Balanced Accuracy')
            
            if optimal_f1_threshold is not None:
                plt.axvline(x=optimal_f1_threshold, color='red', linestyle='--', label=f'Optimal F1 Threshold ({optimal_f1_threshold:.2f})')
            
            plt.title(f'Optimisation du Seuil - {strategy_name}')
            plt.xlabel('Seuil de Probabilité')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            plot_filename = f"threshold_optimization_{strategy_name}_{self.current_subset_info}.png"
            plot_path = os.path.join(custom_plots_base_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération du plot d'optimisation du seuil pour {strategy_name}: {e}")
            traceback.print_exc()
            return None
    
    def _aggregate_ohe_importances(self, importances, processed_feature_names, original_categorical_features):
        """
        Agrège les importances des features pour les colonnes encodées en One-Hot.
        """
        aggregated_importances = {}
        
        # Initialize with original numerical features
        for i, name in enumerate(processed_feature_names):
            is_categorical_ohe = False
            for cat_col in original_categorical_features:
                # Check for the correct OHE prefix generated by ColumnTransformer
                # which is typeaksally "cat__original_column_name_category_value"
                if name.startswith(f"cat__{cat_col}_"):
                    is_categorical_ohe = True
                    break
            if not is_categorical_ohe:
                # Remove the 'num__' prefix if it exists
                aggregated_importances[name.replace('num__', '')] = importances[i]

        # Aggregate OHE features
        for cat_col in original_categorical_features:
            ohe_cols = [name for name in processed_feature_names if name.startswith(f"cat__{cat_col}_")]
            if ohe_cols:
                # Sum importances for all OHE parts of the original categorical feature
                summed_importance = sum(importances[list(processed_feature_names).index(ohe_col)] for ohe_col in ohe_cols)
                aggregated_importances[cat_col] = summed_importance
            elif cat_col in processed_feature_names: # Fallback: Handle case where OHE might not have been applied or named differently
                aggregated_importances[cat_col] = importances[list(processed_feature_names).index(cat_col)]

        return aggregated_importances

    def compute_feature_importance_in_dir(self, pipeline, X_train, y_train, strategy_name, custom_feature_importance_dir, processed_feature_names):
        """
        Calcule et sauvegarde l'importance des features (built-in et permutation) et leurs plots.
        Cette version enregistre et trace également les importances brutes (non agrégées)
        pour les fonctionnalités One-Hot Encoded.
        """
        feature_importances_data = {}
        base_classifier = None

        # 1. Find the base classifier (RandomForestClassifier or BalancedRandomForestClassifier) in the pipeline
        try:
            for step_name, step_obj in pipeline.named_steps.items():
                if isinstance(step_obj, (RandomForestClassifier, BalancedRandomForestClassifier)):
                    base_classifier = step_obj
                    break
                elif isinstance(step_obj, ImbPipeline): # If it's an imblearn pipeline
                    for sub_step_name, sub_step_obj in step_obj.named_steps.items():
                        if isinstance(sub_step_obj, (RandomForestClassifier, BalancedRandomForestClassifier)):
                            base_classifier = sub_step_obj
                            break
                    if base_classifier: # If found in sub-pipeline, break outer loop too
                        break
        except Exception as e:
            print(f"❌ Erreur lors de la recherche du classifieur de base dans le pipeline pour {strategy_name}: {e}")
            traceback.print_exc()

        # 2. Built-in Feature Importance
        try:
            if base_classifier and hasattr(base_classifier, 'feature_importances_'):
                built_in_importances_raw = base_classifier.feature_importances_
                
                # Store raw built-in importances
                feature_importances_data['built_in_importance_raw'] = {
                    processed_feature_names[i]: float(built_in_importances_raw[i]) 
                    for i in range(len(built_in_importances_raw))
                }
                print(f"Stored raw built-in importances for {strategy_name}.")

                # Aggregate OHE feature importances
                built_in_importances_aggregated = self._aggregate_ohe_importances(
                    built_in_importances_raw, processed_feature_names, self.categorical_features
                )
                feature_importances_data['built_in_importance'] = {
                    k: float(v) for k, v in built_in_importances_aggregated.items()
                }
                print(f"Computed aggregated built-in importances for {strategy_name}.")
            else:
                print(f"⚠️ Modèle {strategy_name} n'a pas d'attribut 'feature_importances_'. Skipping built-in importance calculation.")
                feature_importances_data['built_in_importance'] = None
                feature_importances_data['built_in_importance_raw'] = None

        except Exception as e:
            print(f"❌ Erreur lors du calcul de l'importance built-in pour {strategy_name}: {e}")
            traceback.print_exc()
            feature_importances_data['built_in_importance'] = None
            feature_importances_data['built_in_importance_raw'] = None

        # 3. Permutation Importance
        try:
            result = permutation_importance(
                pipeline, self.X_test_processed, self.y_test, n_repeats=10, random_state=self.random_state, n_jobs=-1
            )
            
            permutation_importances_raw_mean = result.importances_mean
            permutation_importances_raw_std = result.importances_std

            # Store raw permutation importances
            feature_importances_data['permutation_importance_raw'] = {
                processed_feature_names[i]: {'mean': float(permutation_importances_raw_mean[i]), 'std': float(permutation_importances_raw_std[i])}
                for i in range(len(permutation_importances_raw_mean))
                if not pd.isna(permutation_importances_raw_mean[i]) and np.isfinite(permutation_importances_raw_mean[i])
            }
            print(f"Stored raw permutation importances for {strategy_name}.")

            # Aggregate OHE permutation importances
            permutation_importances_aggregated = {}
            for i, name in enumerate(processed_feature_names):
                is_categorical_ohe = False
                for cat_col in self.categorical_features:
                    if name.startswith(f"cat__{cat_col}_"):
                        is_categorical_ohe = True
                        break
                if not is_categorical_ohe:
                    permutation_importances_aggregated[name.replace('num__', '')] = {
                        'mean': permutation_importances_raw_mean[i],
                        'std': permutation_importances_raw_std[i]
                    }

            for cat_col in self.categorical_features:
                ohe_indices = [i for i, name in enumerate(processed_feature_names) if name.startswith(f"cat__{cat_col}_")]
                if ohe_indices:
                    summed_mean = sum(permutation_importances_raw_mean[idx] for idx in ohe_indices)
                    combined_std = np.sqrt(sum(permutation_importances_raw_std[idx]**2 for idx in ohe_indices))
                    permutation_importances_aggregated[cat_col] = {'mean': float(summed_mean), 'std': float(combined_std)}
                elif cat_col in processed_feature_names:
                     idx = list(processed_feature_names).index(cat_col)
                     permutation_importances_aggregated[cat_col] = {'mean': float(permutation_importances_raw_mean[idx]), 'std': float(permutation_importances_raw_std[idx])}

            feature_importances_data['permutation_importance'] = {
                k: {'mean': float(v['mean']), 'std': float(v['std'])} 
                for k, v in permutation_importances_aggregated.items()
                if not pd.isna(v['mean']) and np.isfinite(v['mean'])
            }
            print(f"Computed aggregated permutation importances for {strategy_name}.")
        except Exception as e:
            print(f"❌ Erreur lors du calcul de l'importance par permutation pour {strategy_name}: {e}")
            traceback.print_exc()
            feature_importances_data['permutation_importance'] = None
            feature_importances_data['permutation_importance_raw'] = None

        # Save feature importances to JSON
        json_filename = f"feature_importance_{strategy_name}_{self.current_subset_info}.json"
        json_path = os.path.join(custom_feature_importance_dir, json_filename)
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(feature_importances_data, f, indent=4)
            print(f"💾 Importances des features sauvegardées : {json_path}")
            self._write_report_content(f"- **Importances des features sauvegardées** : [`feature_importance_{strategy_name}_{self.current_subset_info}.json`]({os.path.basename(json_path)})\n")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du JSON d'importance des features: {e}")
            traceback.print_exc()

        return feature_importances_data


    def _plot_top_n_feature_importances_in_dir(self, importances_dict, n, plot_type, strategy_name, custom_plots_base_dir):
        """
        Génère et sauvegarde un plot des N features les plus importantes.
        importances_dict: dict (feature_name -> score)
        """
        if not importances_dict:
            print(f"Pas de données d'importance pour le plot '{plot_type}' pour {strategy_name}.")
            return None
        
        try:
            # Sort features by importance score
            sorted_importances = sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
            top_n_features = sorted_importances[:n]
            
            features = [item[0] for item in top_n_features]
            scores = [item[1] for item in top_n_features]

            plt.figure(figsize=(12, 8))
            sns.barplot(x=scores, y=features, palette='viridis')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {n} Importances des Features ({plot_type}) - {strategy_name}')
            plt.tight_layout()
            
            plot_filename = f"feature_importance_{plot_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{strategy_name}_{self.current_subset_info}.png"
            plot_path = os.path.join(custom_plots_base_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération du plot d'importance des features ({plot_type}) pour {strategy_name}: {e}")
            traceback.print_exc()
            return None


    def run_pipeline(self, subset_X=None, subset_y=None, subset_info="global"):
        self.current_subset_info = subset_info
        X_train_current, y_train_current = (self.X_train_processed, self.y_train) if subset_X is None else (subset_X, subset_y)
        
        print(f"\n🚀 Démarrage du pipeline pour subset: {subset_info}")
        
        if subset_info != "global":
            base_plots_for_this_run = os.path.join(self.plots_base_dir, subset_info)
            subset_models_base_dir = os.path.join(self.models_base_dir, subset_info)  
            subset_feature_importance_dir = os.path.join(self.feature_importance_dir, subset_info)
        else:
            base_plots_for_this_run = self.plots_base_dir
            subset_models_base_dir = self.models_base_dir
            subset_feature_importance_dir = self.feature_importance_dir
        
        os.makedirs(base_plots_for_this_run, exist_ok=True)
        os.makedirs(subset_models_base_dir, exist_ok=True)
        os.makedirs(subset_feature_importance_dir, exist_ok=True)
        
        print(f"📁 Répertoires de base créés pour les résultats de {subset_info}:")
        print(f"   - Plots (base): {base_plots_for_this_run}")
        print(f"   - Models: {subset_models_base_dir}")
        print(f"   - Features: {subset_feature_importance_dir}")
        
        test_plot_path = self.test_simple_plot(f"initial_{subset_info}", base_plots_for_this_run)
        if test_plot_path:
            print(f"✅ Matplotlib fonctionne correctement (plot test: {os.path.basename(test_plot_path)})")
        else:
            print(f"❌ Problème avec Matplotlib - vérifiez votre environnement")
        
        strategies = {
            "NoResampling": "Pipeline sans rééchantillonnage (RandomForest direct)",
            "BalancedRandomForest": "Pipeline avec BalancedRandomForestClassifier", # New strategy
            "RandomOverSampler": "Pipeline avec RandomOverSampler",
            "RandomUnderSampler": "Pipeline avec RandomUnderSampler",
            "SMOTEENN": "Pipeline avec SMOTEENN",
            # "SMOTETomek": "Pipeline avec SMOTETomek",
            # "SMOTE": "Pipeline avec SMOTE"
        }

        subset_results = {"info": subset_info}

        for strategy_name, strategy_description in strategies.items():
            print(f"\n🔄 === Exécution de la stratégie : {strategy_name} ===")
            self._write_report_section(f"### Stratégie : {strategy_name}")
            self._write_report_content(f"{strategy_description}.\n")

            current_strategy_plots_dir = os.path.join(base_plots_for_this_run, strategy_name)
            os.makedirs(current_strategy_plots_dir, exist_ok=True)
            print(f"   📁 Répertoire des plots pour {strategy_name}: {current_strategy_plots_dir}")

            try:
                print(f"🔍 Optimisation Optuna pour {strategy_name}...")
                best_params = self._optimize_hyperparameters(X_train_current, y_train_current, strategy_name)
                print(f"✅ Optuna terminé pour {strategy_name}. Meilleurs paramètres: {best_params}")
                
                best_params_str = str(best_params) if isinstance(best_params, dict) else str(best_params)
                self._write_report_content(f"- **Meilleurs hyperparamètres Optuna** : `{best_params_str}`\n")
                
                processed_best_params = {}
                if isinstance(best_params, dict):
                    if strategy_name == "BalancedRandomForest":
                        for k, v in best_params.items():
                            processed_best_params[k] = v
                    else: # For RandomForestClassifier and resamplers
                        for k, v in best_params.items():
                            if k.startswith('randomforestclassifier__'):
                                processed_best_params[k.replace('randomforestclassifier__', '')] = v
                            elif k.startswith('sampler__'):
                                processed_best_params[k.replace('sampler__', '')] = v
                            else:
                                processed_best_params[k] = v

                print(f"🏗️ Construction du pipeline pour {strategy_name}...")
                
                if strategy_name == "BalancedRandomForest":
                    classifier = BalancedRandomForestClassifier(**processed_best_params, random_state=self.random_state)
                    pipeline = SklearnPipeline([('balancedrandomforestclassifier', classifier)])
                else:
                    classifier = RandomForestClassifier(**processed_best_params, random_state=self.random_state)
                    if strategy_name == "NoResampling":
                        pipeline = SklearnPipeline([('randomforestclassifier', classifier)])
                    else:
                        sampler = self._get_sampler(strategy_name)
                        pipeline = ImbPipeline([
                            ('sampler', sampler),
                            ('randomforestclassifier', classifier)
                        ])

                print(f"📚 Entraînement du modèle pour {strategy_name}...")
                pipeline.fit(X_train_current, y_train_current)
                self.models[strategy_name] = pipeline

                # Sauvegarde du modèle
                model_path = os.path.join(subset_models_base_dir, f"model_{strategy_name}_{subset_info}.joblib")
                joblib.dump(pipeline, model_path)
                print(f"💾 Modèle sauvegardé : {model_path}")
                self._write_report_content(f"- **Modèle sauvegardé** : [`model_{strategy_name}_{subset_info}.joblib`]({os.path.basename(model_path)})\n")

                # Évaluation sur l'ensemble de test
                print(f"📊 Évaluation pour {strategy_name}...")
                y_pred_test = pipeline.predict(self.X_test_processed)
                y_proba_test = pipeline.predict_proba(self.X_test_processed)
                
                # Probabilities for training set for plots
                y_proba_train = pipeline.predict_proba(X_train_current)

                # Calcul et stockage des métriques complètes
                self._calculate_and_store_metrics(strategy_name, subset_info, self.y_test, y_pred_test, y_proba_test)
                metrics = self.all_metrics[subset_info][strategy_name] # Retrieve the comprehensive metrics

                subset_results[strategy_name] = metrics # Keep this for current run's reporting

                print(f"📈 Métriques pour {strategy_name} (Test Set):")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            print(f"   {metric}: {value:.4f}")
                            self._write_report_content(f"   - **{metric.replace('_', ' ').title()}** : `{value:.4f}`\n")
                        else:
                            print(f"   {metric}: {value}")
                            self._write_report_content(f"   - **{metric.replace('_', ' ').title()}** : `{value}`\n")
                    
                print(f"🎨 Génération des visualisations pour {strategy_name}...")
                
                plot_results = {}
                
                # ROC/PR Curves (both test and train) - now superposed
                print(f"    📊 Génération courbes ROC/PR (Test et Train) superposées...")
                roc_pr_path = self._plot_roc_pr_curves_in_dir(
                    self.y_test, y_proba_test, y_train_current, y_proba_train, 
                    strategy_name, current_strategy_plots_dir
                )
                plot_results['roc_pr'] = roc_pr_path
                
                # Confusion Matrix
                print(f"    🔲 Génération matrice de confusion...")
                cm_path = self._plot_confusion_matrix_in_dir(
                    self.y_test, y_pred_test, strategy_name, current_strategy_plots_dir
                )
                plot_results['confusion_matrix'] = cm_path
                
                # Probability Distribution
                print(f"    📈 Génération distribution probabilités...")
                if y_proba_test is not None and y_proba_test.ndim > 1: # Ensure positive class probabilities are available
                    prob_path = self._plot_probability_distribution_in_dir(
                        self.y_test, y_proba_test[:, 1], strategy_name, current_strategy_plots_dir
                    )
                else:
                    print("     ⚠️ Distribution de probabilités non générée: y_proba_test n'est pas disponible ou mal formé.")
                    prob_path = None
                plot_results['probability_dist'] = prob_path

                # Threshold Optimization
                print(f"    📏 Génération optimisation du seuil...")
                if y_proba_test is not None and y_proba_test.ndim > 1:
                    threshold_path = self._plot_threshold_optimization_in_dir(
                        self.y_test, y_proba_test[:, 1], strategy_name, current_strategy_plots_dir
                    )
                else:
                    print("     ⚠️ Optimisation du seuil non générée: y_proba_test n'est pas disponible ou mal formé.")
                    threshold_path = None
                plot_results['threshold_optimization'] = threshold_path
                
                # Learning Curve
                print(f"    📚 Génération courbes d'apprentissage...")
                lc_path = self._plot_learning_curve_in_dir(
                    pipeline, X_train_current, y_train_current, strategy_name, current_strategy_plots_dir
                )
                plot_results['learning_curve'] = lc_path
                
                # IMPORTANCE DES FEATURES
                print(f"🎯 Calcul importance des features pour {strategy_name}...")
                feature_importances = self.compute_feature_importance_in_dir(
                    pipeline, X_train_current, y_train_current, strategy_name, 
                    subset_feature_importance_dir, self.processed_feature_names
                )
                self.feature_importances[strategy_name] = feature_importances

                if 'built_in_importance' in feature_importances and feature_importances['built_in_importance']:
                    print(f"    🎯 Génération plot importance built-in (agrégée)...")
                    builtin_path = self._plot_top_n_feature_importances_in_dir(
                        feature_importances['built_in_importance'],
                        n=15, plot_type="Built-in (Agrégée)", strategy_name=strategy_name,
                        custom_plots_base_dir=current_strategy_plots_dir
                    )
                    plot_results['builtin_importance_aggregated'] = builtin_path
                else:
                    print(f"⚠️ Skipping built-in aggregated importance plot: 'built_in_importance' not found or empty for {strategy_name}.")
                    plot_results['builtin_importance_aggregated'] = None
                
                if 'built_in_importance_raw' in feature_importances and feature_importances['built_in_importance_raw']:
                    print(f"    🎯 Génération plot importance built-in (brute OHE)...")
                    builtin_raw_path = self._plot_top_n_feature_importances_in_dir(
                        feature_importances['built_in_importance_raw'],
                        n=15, plot_type="Built-in (OHE Brute)", strategy_name=strategy_name,
                        custom_plots_base_dir=current_strategy_plots_dir
                    )
                    plot_results['builtin_importance_raw'] = builtin_raw_path
                else:
                    print(f"⚠️ Skipping built-in raw (OHE) importance plot: 'built_in_importance_raw' not found or empty for {strategy_name}.")
                    plot_results['builtin_importance_raw'] = None


                if 'permutation_importance' in feature_importances and feature_importances['permutation_importance']:
                    permutation_means = {f: data['mean'] for f, data in feature_importances['permutation_importance'].items() 
                                         if not pd.isna(data['mean']) and np.isfinite(data['mean'])}
                    print(f"    🔄 Génération plot importance par permutation (agrégée)...")
                    perm_path = self._plot_top_n_feature_importances_in_dir(
                        permutation_means, n=15, plot_type="Permutation (Agrégée)", strategy_name=strategy_name,
                        custom_plots_base_dir=current_strategy_plots_dir
                    )
                    plot_results['permutation_importance_aggregated'] = perm_path
                else:
                    print(f"⚠️ Skipping permutation aggregated importance plot: 'permutation_importance' not found or empty for {strategy_name}.")
                    plot_results['permutation_importance_aggregated'] = None

                if 'permutation_importance_raw' in feature_importances and feature_importances['permutation_importance_raw']:
                    permutation_raw_means = {f: data['mean'] for f, data in feature_importances['permutation_importance_raw'].items() 
                                         if not pd.isna(data['mean']) and np.isfinite(data['mean'])}
                    print(f"    🔄 Génération plot importance par permutation (brute OHE)...")
                    perm_raw_path = self._plot_top_n_feature_importances_in_dir(
                        permutation_raw_means, n=15, plot_type="Permutation (OHE Brute)", strategy_name=strategy_name,
                        custom_plots_base_dir=current_strategy_plots_dir
                    )
                    plot_results['permutation_importance_raw'] = perm_raw_path
                else:
                    print(f"⚠️ Skipping permutation raw (OHE) importance plot: 'permutation_importance_raw' not found or empty for {strategy_name}.")
                    plot_results['permutation_importance_raw'] = None

                print(f"🔍 Vérification des plots générés pour {strategy_name}...")
                successful_plots = []
                failed_plots = []
                for plot_type, plot_path in plot_results.items():
                    if plot_path and os.path.exists(plot_path):
                        successful_plots.append(f"{plot_type}: {plot_path}")
                        print(f" ✅ {plot_type}: {os.path.basename(plot_path)}")
                    else:
                        failed_plots.append(plot_type)
                        print(f" ❌ {plot_type}: Non généré")
                
                self._write_report_content(f"- **Plots générés** : {len(successful_plots)}/{len(plot_results)}\n")
                for plot_info in successful_plots:
                    # plot_info is like "roc_pr: C:/path/to/plot.png"
                    # Need to extract the path and then make it relative
                    actual_plot_path = plot_info.split(': ')[1]
                    relative_plot_path = os.path.relpath(actual_plot_path, self.output_dir)
                    self._write_report_content(f" - ✅ [{plot_info.split(':')[0]}]({relative_plot_path})\n")
                if failed_plots:
                    self._write_report_content(f"- **Plots échoués** : {', '.join(failed_plots)}\n")
                self._write_report_content("\n---\n\n")
                print(f"✅ Stratégie {strategy_name} terminée avec succès!")
                print(f" 📊 Plots générés: {len(successful_plots)}/{len(plot_results)}")
            except Exception as e:
                error_msg = f"❌ Erreur lors de l'exécution de la stratégie {strategy_name}: {str(e)}"
                print(error_msg)
                print(f"📋 Stack trace: {traceback.format_exc()}")
                subset_results[strategy_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                self._write_report_content(f"- **❌ ERREUR** : {str(e)}\n")
                self._write_report_content("- **Statut** : Échec de l'exécution\n\n---\n\n")
                continue
        
        print(f"\n🏁 Pipeline terminé pour subset: {subset_info}")
        successful_strategies = [s for s in subset_results.keys() if s != 'info' and isinstance(subset_results[s], dict) and 'error' not in subset_results[s]]
        failed_strategies = [s for s in subset_results.keys() if s != 'info' and isinstance(subset_results[s], dict) and 'error' in subset_results[s]]
        print(f"📊 Résumé final:")
        print(f" ✅ Stratégies réussies: {len(successful_strategies)}")
        print(f" ❌ Stratégies échouées: {len(failed_strategies)}")
        if successful_strategies:
            print(f" 🏆 Stratégies réussies: {', '.join(successful_strategies)}")
        if failed_strategies:
            print(f" 💥 Stratégies échouées: {', '.join(failed_strategies)}")
        
        self.results[subset_info] = subset_results

        # Call save_all_metrics after all strategies for the current subset are processed
        self.save_all_metrics()

        if len(successful_strategies) > 1:
            try:
                comparison_plot_base_dir = self.plots_base_dir if subset_info == "global" else os.path.join(self.plots_base_dir, subset_info)
                self._compare_strategies(subset_info, comparison_plot_base_dir)
                print(f"✅ Comparaison des stratégies (barres) générée")
                
                # New: Generate Radar Chart for strategy comparison
                print(f"🎡 Génération du graphique en cercle concentrique pour la comparaison des stratégies...")
                radar_plot_path = self._plot_radar_chart_comparison_in_dir(subset_info, comparison_plot_base_dir)
                if radar_plot_path:
                    print(f"✅ Graphique en cercle concentrique généré: {os.path.basename(radar_plot_path)}")
                    self._write_report_content(f"- **Graphique en cercle concentrique (comparaison des stratégies)** : [`strategy_comparison_radar_{subset_info}.png`]({os.path.basename(radar_plot_path)})\n")
                else:
                    print(f"❌ Échec de la génération du graphique en cercle concentrique.")

            except Exception as e:
                print(f"❌ Erreur lors de la comparaison des stratégies: {str(e)}")

        print(f"🎉 Pipeline complet terminé pour {subset_info}!")
        return subset_results

    def run_on_subsets(self, subset_column, subsets_to_run):
        print(f"\n--- Exécution du pipeline sur les sous-ensembles de '{subset_column}' ---")
        self._write_report_section(f"## Analyse par Sous-ensembles : {subset_column}")
        for value in subsets_to_run:
            print(f"\nTraitement du sous-ensemble : {subset_column} = {value}")
            self._write_report_section(f"### Sous-ensemble : {subset_column} = {value}")
            subset_df = self.df[self.df[subset_column] == value].copy()
            if subset_df.empty:
                print(f"Sous-ensemble vide pour {subset_column} = {value}. Saut.")
                self.results[f"{subset_column}_{value}"] = {"info": f"{subset_column}_{value}", "error": "Sous-ensemble vide"}
                self._write_report_content(f"Le sous-ensemble pour `{subset_column} = {value}` est vide.\n")
                continue

            X_subset = subset_df[self.features]
            y_subset = subset_df[self.target]
            from sklearn.model_selection import train_test_split # Re-import here if not already global

            # Temporarily store original test sets and processed feature names from the main pipeline
            original_X_test_processed = self.X_test_processed
            original_y_test = self.y_test
            original_processed_feature_names = self.processed_feature_names

            if self.patient_id_col and self.patient_id_col in subset_df.columns:
                patient_ids = subset_df[self.patient_id_col].unique()
                train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_size, random_state=self.random_state)

                X_train_subset_df = subset_df[subset_df[self.patient_id_col].isin(train_ids)].drop(columns=[self.patient_id_col], errors='ignore')
                y_train_subset = y_subset[subset_df[self.patient_id_col].isin(train_ids)]
                print(f"Taille du sous-ensemble '{value}' : X_train_subset.shape={X_train_subset_df.shape}, y_train_subset.shape={y_train_subset.shape}")

                X_test_subset_df = subset_df[subset_df[self.patient_id_col].isin(test_ids)].drop(columns=[self.patient_id_col], errors='ignore')
                y_test_subset = y_subset[subset_df[self.patient_id_col].isin(test_ids)]

                X_train_processed_subset = self.preprocessor.fit_transform(X_train_subset_df[self.features])
                X_test_processed_subset = self.preprocessor.transform(X_test_subset_df[self.features])

                # Update self.processed_feature_names for the current subset's transformation
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    self.processed_feature_names = self.preprocessor.get_feature_names_out(self.features)
                else:
                    print(f"⚠️ Avertissement: (Sous-ensemble {value}) Impossible d'obtenir les noms de features traités via get_feature_names_out. Les noms de colonnes par défaut seront utilisés.")
                    self.processed_feature_names = [f"feature_{i}" for i in range(X_train_processed_subset.shape[1])]
            else:
                X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
                    X_subset[self.features], y_subset, test_size=self.test_size, random_state=self.random_state, stratify=y_subset
                )
                X_train_processed_subset = self.preprocessor.fit_transform(X_train_subset)
                X_test_processed_subset = self.preprocessor.transform(X_test_subset)

                # Update self.processed_feature_names for the current subset's transformation
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    self.processed_feature_names = self.preprocessor.get_feature_names_out(self.features)
                else:
                    print(f"⚠️ Avertissement: (Sous-ensemble {value}) Impossible d'obtenir les noms de features traités via get_feature_names_out. Les noms de colonnes par défaut seront utilisés.")
                    self.processed_feature_names = [f"feature_{i}" for i in range(X_train_processed_subset.shape[1])]
            
            # Set current test sets for the subset run
            self.X_test_processed = X_test_processed_subset
            self.y_test = y_test_subset
            # self.processed_feature_names is already updated by the blocks above for the current subset's preprocessor.

            subset_results_dict = self.run_pipeline(X_train_processed_subset, y_train_subset, subset_info=f"{subset_column}_{value}")
            self.results[f"{subset_column}_{value}"] = subset_results_dict

            # Restore original test sets and processed feature names for the main pipeline context
            self.X_test_processed = original_X_test_processed
            self.y_test = original_y_test
            self.processed_feature_names = original_processed_feature_names

        print(f"\n--- Exécution sur les sous-ensembles de '{subset_column}' terminée ---")

    def _write_report_header(self):
        """ Écrit l'en-tête initial du rapport d'expérimentation.
        """
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Rapport d'Expérimentation ML Rigoureuse: {self.experiment_name}\n")
            f.write(f"**Date de l'expérimentation** : {self.experiment_timestamp}\n")
            f.write(f"**Cible** : `{self.target}`\n")
            f.write(f"**Features utilisées** : {len(self.features)} ({', '.join(self.features[:5])}...)\n")
            f.write(f"**Taille de l'ensemble de test** : {self.test_size * 100}%\n")
            f.write(f"**Graine aléatoire** : {self.random_state}\n")
            f.write("\n## Résumé de l'Expérimentation\n")
            f.write("Ce rapport détaille les performances de divers pipelines de classification "
                    "avec rééchantillonnage pour un problème de classification binaire, utilisant RandomForest.\n\n")

    def _write_report_section(self, title):
        """ Ajoute une section au rapport.
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{title}\n\n")

    def _write_report_content(self, content):
        """ Ajoute du contenu au rapport.
        """
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(content)

    def _compare_strategies(self, subset_info, comparison_plot_base_dir):
        """
        Génère un plot comparatif des métriques clés pour différentes stratégies.
        """
        if subset_info not in self.all_metrics or not self.all_metrics[subset_info]:
            print(f"Aucune métrique disponible pour la comparaison des stratégies dans le sous-ensemble: {subset_info}.")
            return

        metrics_to_compare = ['f1_score', 'roc_auc', 'balanced_accuracy', 'precision', 'recall']
        
        data_for_plot = []
        for strategy_name, metrics in self.all_metrics[subset_info].items():
            if 'error' not in metrics: # Only include successful runs
                row = {'Strategy': strategy_name}
                for metric_name in metrics_to_compare:
                    row[metric_name] = metrics.get(metric_name)
                data_for_plot.append(row)
        
        if not data_for_plot:
            print(f"Pas de données de métriques valides pour la comparaison des stratégies dans le sous-ensemble: {subset_info}.")
            return

        df_comparison = pd.DataFrame(data_for_plot)
        df_melted = df_comparison.melt(id_vars='Strategy', var_name='Metric', value_name='Score')

        plt.figure(figsize=(14, 8))
        sns.barplot(x='Metric', y='Score', hue='Strategy', data=df_melted, palette='muted')
        plt.title(f'Comparaison des Métriques Clés par Stratégie - {subset_info}')
        plt.ylabel('Score')
        plt.xlabel('Métrique')
        plt.ylim(0, 1.0)
        plt.legend(title='Stratégie', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        
        plot_filename = f"strategy_comparison_{subset_info}.png"
        plot_path = os.path.join(comparison_plot_base_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Plot de comparaison des stratégies généré: {plot_path}")
        self._write_report_content(f"- **Plot de comparaison des stratégies (barres)** : [`strategy_comparison_{subset_info}.png`]({os.path.basename(plot_path)})\n")

    def _plot_radar_chart_comparison_in_dir(self, subset_info, custom_plots_base_dir):
        """
        Génère un graphique en cercle concentrique (radar chart) pour comparer les stratégies.
        """
        if subset_info not in self.all_metrics or not self.all_metrics[subset_info]:
            print(f"Aucune métrique disponible pour le radar chart dans le sous-ensemble: {subset_info}.")
            return None

        metrics_to_compare = ['f1_score', 'roc_auc', 'balanced_accuracy', 'precision', 'recall']
        
        strategies_data = {}
        for strategy_name, metrics in self.all_metrics[subset_info].items():
            if 'error' not in metrics: # Only include successful runs
                values = [metrics.get(metric_name, 0.0) for metric_name in metrics_to_compare]
                strategies_data[strategy_name] = values
        
        if not strategies_data:
            print(f"Pas de données de métriques valides pour le radar chart dans le sous-ensemble: {subset_info}.")
            return None

        num_vars = len(metrics_to_compare)
        # Calculate angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # The plot needs to be a closed circle, so we need to "complete" the loop
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each strategy
        for strategy_name, values in strategies_data.items():
            # Complete the loop for values as well
            plot_values = values + values[:1]
            ax.plot(angles, plot_values, linewidth=2, linestyle='solid', label=strategy_name)
            ax.fill(angles, plot_values, alpha=0.25)

        # Set labels for each axis
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_compare)

        # Set y-axis limits and labels (0 to 1 as metrics are scaled)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f"{val:.1f}" for val in np.arange(0, 1.1, 0.2)], color="gray", size=7)
        ax.tick_params(axis='y', colors='gray', labelleft=False) # Hide default y-tick labels on the right

        plt.title(f'Comparaison des Stratégies (Radar Chart) - {subset_info}', size=16, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        plot_filename = f"strategy_comparison_radar_{subset_info}.png"
        plot_path = os.path.join(custom_plots_base_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        return plot_path

# Exemple d'utilisation (décommenter et adapter pour tester)
# if __name__ == "__main__":
#     # Créer un DataFrame de démonstration
#     data = {
#         'feature1': np.random.rand(100),
#         'feature2': np.random.rand(100) * 10,
#         'feature3': np.random.randint(0, 5, 100),
#         'gender': np.random.choice(['M', 'F'], 100),
#         'patient_id': range(100),
#         'target': np.random.randint(0, 2, 100)
#     }
#     df_demo = pd.DataFrame(data)
    
#     # Simuler un déséquilibre
#     df_demo_balanced = pd.concat([
#         df_demo[df_demo['target'] == 0].sample(50, replace=True, random_state=42),
#         df_demo[df_demo['target'] == 1].sample(10, replace=True, random_state=42)
#     ])
#     df_demo_balanced = df_demo_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#     # Définir les features et la cible
#     features = ['feature1', 'feature2', 'feature3', 'gender']
#     target = 'target'
#     categorical_features = ['gender']
#     patient_id_col = 'patient_id' # Ou None si pas de split par ID patient

#     # Exécuter l'analyse
#     # Pour une exécution globale :
#     # results_global = run_rf_analysis(
#     #     df=df_demo_balanced,
#     #     features=features,
#     #     target=target,
#     #     categorical_features=categorical_features,
#     #     patient_id_col=patient_id_col,
#     #     output_path="C:/rf_experiments/",
#     #     experiment_name="RF_Global_Analysis",
#     #     n_trials=5 # Réduire pour un test rapide
#     # )
#     # print("Résultats de l'analyse globale:", results_global)

#     # Pour une exécution par sous-ensemble (par exemple, par 'gender') :
#     # results_subsets = run_rf_analysis(
#     #     df=df_demo_balanced,
#     #     features=features,
#     #     target=target,
#     #     categorical_features=categorical_features,
#     #     patient_id_col=patient_id_col,
#     #     output_path="C:/rf_experiments/",
#     #     experiment_name="RF_Gender_Subsets",
#     #     n_trials=3, # Réduire pour un test rapide
#     #     subset_column='gender',
#     #     subsets_to_run=['M', 'F']
#     # )
#     # print("Résultats de l'analyse par sous-ensembles:", results_subsets)


# Exemple d'utilisation (décommenter et adapter pour tester)
# if __name__ == "__main__":
#     # Créer un DataFrame de démonstration
#     data = {
#         'feature1': np.random.rand(100),
#         'feature2': np.random.rand(100) * 10,
#         'feature3': np.random.randint(0, 5, 100),
#         'gender': np.random.choice(['M', 'F'], 100),
#         'patient_id': range(100),
#         'target': np.random.randint(0, 2, 100)
#     }
#     df_demo = pd.DataFrame(data)
    
#     # Simuler un déséquilibre
#     df_demo_balanced = pd.concat([
#         df_demo[df_demo['target'] == 0].sample(50, replace=True, random_state=42),
#         df_demo[df_demo['target'] == 1].sample(10, replace=True, random_state=42)
#     ])
#     df_demo_balanced = df_demo_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#     # Définir les features et la cible
#     features = ['feature1', 'feature2', 'feature3', 'gender']
#     target = 'target'
#     categorical_features = ['gender']
#     patient_id_col = 'patient_id' # Ou None si pas de split par ID patient

#     # Exécuter l'analyse
#     # Pour une exécution globale :
#     # results_global = run_rf_analysis(
#     #     df=df_demo_balanced,
#     #     features=features,
#     #     target=target,
#     #     categorical_features=categorical_features,
#     #     patient_id_col=patient_id_col,
#     #     output_path="C:/rf_experiments/",
#     #     experiment_name="RF_Global_Analysis",
#     #     n_trials=5 # Réduire pour un test rapide
#     # )
#     # print("Résultats de l'analyse globale:", results_global)

#     # Pour une exécution par sous-ensemble (par exemple, par 'gender') :
#     # results_subsets = run_rf_analysis(
#     #     df=df_demo_balanced,
#     #     features=features,
#     #     target=target,
#     #     categorical_features=categorical_features,
#     #     patient_id_col=patient_id_col,
#     #     output_path="C:/rf_experiments/",
#     #     experiment_name="RF_Gender_Subsets",
#     #     n_trials=3, # Réduire pour un test rapide
#     #     subset_column='gender',
#     #     subsets_to_run=['M', 'F']
#     # )
#     # print("Résultats de l'analyse par sous-ensembles:", results_subsets)



# Exemple d'appel de main_rf_pipeline (si vous avez un DataFrame df_pca et des listes features_ml, cat_var)
# Supposons que vous avez df_pca, features_pca, cat_var définis comme dans vos exemples précédents
# features_ml = features_pca + cat_var
# main_rf_pipeline(df_pca, features_ml, cat_var, target='statut_deces_boolean')

# def main_rf_pipeline(df_clusters, cat_var, features_ml, output_dir=None, subset_col=None, subset_values=None):
#     # Important : assurez-vous que 'pseudo_provisoire' est inclus dans features_ml si patient_id_column est utilisé
#     # et si cette colonne est nécessaire pour le split holdout par patient_id
#     if 'pseudo_provisoire' not in features_ml:
#         features_ml.append('pseudo_provisoire')

#     features_ml = list(set(features_ml)) # Supprime les doublons après ajout

#     print("conversion des variables catégorielles en type 'category'")
#     df_clusters = conversion_to_cat(df_clusters, cat_var)
#     print("Distribution de la colonne cible :")
#     print(df_clusters['statut_deces_boolean'].value_counts())
#     print("conversion terminée")
#     print("="*50)
#     print("prédiction des deces a partir des variables de pollution et de température")

#     # Passer cat_var à predict_deces pour qu'elle puisse identifier les features catégorielles
#     predict_deces(
#         df_clusters,
#         features_ml,
#         'statut_deces_boolean',
#         cat_var=cat_var, # Ajoutez cette ligne
#         patient_id_column='pseudo_provisoire',
#         output_dir=output_dir,
#         subset_col=subset_col,
#         subset_values=subset_values
#     )


# Assurez-vous que toutes les importations nécessaires sont présentes en haut du fichier
# Si ce fichier est exécuté seul, ajoutez des exemples d'utilisation ou un bloc main
        

#---------------------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime as dt
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline as SklearnPipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
#                              roc_auc_score, precision_recall_curve, average_precision_score,
#                              precision_score, recall_score, f1_score, balanced_accuracy_score, make_scorer)
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import (SMOTE, RandomOverSampler)
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.combine import SMOTEENN, SMOTETomek
# import json
# import joblib
# from sklearn.inspection import permutation_importance

# class EnhancedRigorousMLPipeline:
#     """
#     Pipeline ML rigoureux avec persistance, reprise d'expérience et visualisations complètes.
#     Intègre la gestion du déséquilibre des classes, l'optimisation des hyperparamètres,
#     l'analyse d'importance des features et l'évaluation sur l'ensemble de test (holdout).
#     """

#     def __init__(self, base_output_path="C:/experiments/", experiment_name="rigorous_ml"):
#         self.base_output_path = base_output_path
#         self.experiment_name = experiment_name
#         self.output_dir = None
#         self.results_history = []
#         self.completed_strategies = set()
#         self.experiment_state_file = None
#         self.current_subset_info = "all_data" # Pour la traçabilité des subsets
#         self.holdout_data = {} # Pour stocker X_holdout, y_holdout par subset

#     def create_output_directory(self, resume_experiment_id=None, subset_name=None):
#         """
#         Crée un répertoire organisé par date pour les résultats ou reprend une expérience.
#         Peut inclure un sous-répertoire pour les subsets.
#         """
#         if resume_experiment_id:
#             self.output_dir = resume_experiment_id
#             if not os.path.exists(self.output_dir):
#                 raise ValueError(f"Le répertoire d'expérience {resume_experiment_id} n'existe pas")
#             print(f"📁 Reprise de l'expérience : {self.output_dir}")
#         else:
#             today = dt.now().strftime("%Y-%m-%d")
#             timestamp = dt.now().strftime("%H-%M-%S")

#             base_run_dir = os.path.join(
#                 self.base_output_path,
#                 f"{self.experiment_name}_{today}",
#                 f"run_{timestamp}"
#             )

#             if subset_name:
#                 self.output_dir = os.path.join(base_run_dir, f"subset_{subset_name}")
#             else:
#                 self.output_dir = base_run_dir

#             os.makedirs(self.output_dir, exist_ok=True)
#             print(f"📁 Nouveau répertoire de sortie créé : {self.output_dir}")

#         # Créer les sous-répertoires pour l'organisation
#         self.plots_dir = os.path.join(self.output_dir, "plots")
#         self.models_dir = os.path.join(self.output_dir, "models")
#         self.data_dir = os.path.join(self.output_dir, "data")
#         self.metrics_dir = os.path.join(self.output_dir, "metrics")

#         for dir_path in [self.plots_dir, self.models_dir, self.data_dir, self.metrics_dir]:
#             os.makedirs(dir_path, exist_ok=True)

#         # Fichier d'état de l'expérience
#         self.experiment_state_file = os.path.join(self.output_dir, "experiment_state.json")

#         return self.output_dir

#     def save_experiment_state(self, all_results, current_strategy=None):
#         """Sauvegarde l'état actuel de l'expérience, incluant l'info du subset."""
#         state = {
#             'timestamp': dt.now().isoformat(),
#             'completed_strategies': list(self.completed_strategies),
#             'current_strategy': current_strategy,
#             'output_directory': self.output_dir,
#             'total_strategies': len(self.prepare_resampling_strategies()),
#             'progress': len(self.completed_strategies),
#             'current_subset': self.current_subset_info # Ajouter l'info du subset
#         }

#         with open(self.experiment_state_file, 'w') as f:
#             json.dump(state, f, indent=2)

#     def load_experiment_state(self):
#         """Charge l'état d'une expérience précédente"""
#         if os.path.exists(self.experiment_state_file):
#             with open(self.experiment_state_file, 'r') as f:
#                 state = json.load(f)

#             self.completed_strategies = set(state.get('completed_strategies', []))
#             self.current_subset_info = state.get('current_subset', 'all_data') # Charger l'info du subset
#             print(f"📋 État chargé: {len(self.completed_strategies)} stratégies déjà complétées pour subset '{self.current_subset_info}'")
#             print(f"    Stratégies terminées: {', '.join(self.completed_strategies)}")
#             return state
#         return None

#     def load_existing_results(self):
#         """Charge les résultats existants depuis le disque"""
#         results_file = os.path.join(self.output_dir, "comprehensive_experiment_results.json")
#         if os.path.exists(results_file):
#             with open(results_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             return data.get('detailed_results', {})
#         return {}

#     def prepare_resampling_strategies(self):
#         """Définit différentes stratégies de rééquilibrage"""
#         strategies = {
#             'baseline': None, # Uncomment to re-enable
#             'class_weight': 'class_weight_only', # Uncomment to re-enable
#             'random_oversample': RandomOverSampler(random_state=42), # Uncomment to re-enable
#             'random_undersample': RandomUnderSampler(random_state=42), # Uncomment to re-enable
#             'smote': SMOTE(random_state=42, k_neighbors=3), # Uncomment to re-enable
#             'balanced_rf': 'balanced_random_forest', # Uncomment to re-enable
#             # 'smoteenn': SMOTEENN(random_state=42), # Uncomment to re-enable
#             'smotetomek': SMOTETomek(random_state=42) # Only this one is active currently based on your provided code
#         }
#         return strategies

#     def create_pipeline(self, numerical_features, categorical_features, strategy_name, strategy, **classifier_params):
#         """
#         Crée un pipeline selon la stratégie choisie,
#         permettant le passage de paramètres personnalisés pour le classifieur.
#         """

#         preprocessor = ColumnTransformer([
#             ('num', SklearnPipeline([
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ]), numerical_features),
#             ('cat', SklearnPipeline([
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#             ]), categorical_features)
#         ])

#         default_rf_params = {
#             'random_state': 42,
#             'n_estimators': 200,
#             'max_depth': 15,
#             'min_samples_split': 5,
#             'min_samples_leaf': 2
#         }
#         classifier_final_params = {**default_rf_params, **classifier_params}

#         if strategy_name == 'balanced_rf':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', BalancedRandomForestClassifier(**classifier_final_params))
#             ])
#         elif strategy_name == 'class_weight':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', RandomForestClassifier(class_weight='balanced', **classifier_final_params))
#             ])
#         elif strategy_name == 'baseline':
#             pipeline = SklearnPipeline([
#                 ('preprocessor', preprocessor),
#                 ('classifier', RandomForestClassifier(**classifier_final_params))
#             ])
#         else:
#             pipeline = ImbPipeline([
#                 ('preprocessor', preprocessor),
#                 ('resampler', strategy),
#                 ('classifier', RandomForestClassifier(**classifier_final_params))
#             ])

#         return pipeline

#     def tune_hyperparameters(self, X_train, y_train, numerical_features, categorical_features,
#                              strategy_name, strategy, param_grid, search_type='random', cv_folds=3, scoring='f1'):
#         """
#         Effectue une recherche d'hyperparamètres (GridSearchCV ou RandomizedSearchCV)
#         pour une stratégie donnée.
#         """
#         print(f"\n⚙️ Démarrage du réglage des hyperparamètres pour {strategy_name} (méthode: {search_type.upper()})...")

#         pipeline = self.create_pipeline(numerical_features, categorical_features, strategy_name, strategy, n_estimators=100, max_depth=10)

#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

#         if scoring == 'f1':
#             scorer = make_scorer(f1_score, zero_division=0)
#         elif scoring == 'balanced_accuracy':
#             scorer = 'balanced_accuracy'
#         elif scoring == 'roc_auc':
#             scorer = 'roc_auc'
#         elif scoring == 'recall':
#             scorer = make_scorer(recall_score, zero_division=0)
#         elif scoring == 'precision':
#             scorer = make_scorer(precision_score, zero_division=0)
#         else:
#             scorer = scoring

#         if search_type == 'grid':
#             search = GridSearchCV(
#                 estimator=pipeline,
#                 param_grid=param_grid,
#                 scoring=scorer,
#                 cv=cv,
#                 n_jobs=-1,
#                 verbose=1
#             )
#         elif search_type == 'random':
#             search = RandomizedSearchCV(
#                 estimator=pipeline,
#                 param_distributions=param_grid,
#                 n_iter=50,
#                 scoring=scorer,
#                 cv=cv,
#                 n_jobs=-1,
#                 verbose=1,
#                 random_state=42
#             )
#         else:
#             raise ValueError("search_type doit être 'grid' ou 'random'.")

#         search.fit(X_train, y_train)

#         print(f"✅ Réglage des hyperparamètres terminé pour {strategy_name}.")
#         print(f"    Meilleur score ({scoring}): {search.best_score_:.3f}")
#         print(f"    Meilleurs paramètres: {search.best_params_}")

#         return search.best_estimator_, search.best_params_, search.best_score_, search.cv_results_

#     def calculate_feature_importance(self, pipeline, X_val, y_val, strategy_name, features):
#         """Calcule l'importance des features avec plusieurs méthodes"""
#         print(f"🔍 Calcul de l'importance des features pour {strategy_name}...")

#         importance_results = {}

#         try:
#             preprocessor_step = pipeline.named_steps['preprocessor']
#             transformed_feature_names = None
#             try:
#                 # Obtenir les noms des features après transformation
#                 # Ceci nécessite que le preprocessor ait été fit
#                 transformed_feature_names = preprocessor_step.get_feature_names_out(X_val.columns)
#             except AttributeError:
#                 print("    ⚠️ Impossible d'obtenir les noms des features transformées. Le préprocesseur n'est peut-être pas ajusté ou la méthode get_feature_names_out est indisponible.")
#             except Exception as e:
#                 print(f"    ⚠️ Erreur lors de l'obtention des noms de features transformées : {e}")

#             if transformed_feature_names is not None and hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
#                 importances = pipeline.named_steps['classifier'].feature_importances_

#                 if len(transformed_feature_names) == len(importances):
#                     importance_df = pd.DataFrame({
#                         'feature': transformed_feature_names,
#                         'importance': importances
#                     }).sort_values('importance', ascending=False)
#                     importance_results['built_in'] = importance_df.to_dict('records')
#                 else:
#                     print(f"    ⚠️ Discordance dans la longueur de l'importance intégrée des features pour {strategy_name}. Attendu {len(transformed_feature_names)}, obtenu {len(importances)}.")

#             print("    Calcul de la permutation importance...")
#             # La permutation importance peut être coûteuse, ajustez n_repeats si nécessaire
#             perm_importance = permutation_importance(
#                 pipeline, X_val, y_val,
#                 n_repeats=5, random_state=42, n_jobs=-1
#             )

#             perm_importance_df = pd.DataFrame({
#                 'feature': X_val.columns.tolist(),
#                 'importance_mean': perm_importance.importances_mean,
#                 'importance_std': perm_importance.importances_std
#             }).sort_values('importance_mean', ascending=False)

#             importance_results['permutation'] = perm_importance_df.to_dict('records')

#         except Exception as e:
#             print(f"    ⚠️ Erreur lors du calcul d'importance des features: {str(e)}")
#             importance_results['error'] = str(e)

#         return importance_results

#     def plot_comprehensive_analysis(self, strategy_name, y_true, y_pred, y_pred_proba,
#                                     importance_results, threshold_df, features, eval_set_name="Validation"):
#         """Crée tous les graphiques d'analyse pour une stratégie"""

#         plt.rcParams['figure.figsize'] = (15, 10)

#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle(f'Analyse Complète - {strategy_name} ({eval_set_name} Set, Subset: {self.current_subset_info})', fontsize=16, fontweight='bold')

#         cm = confusion_matrix(y_true, y_pred)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
#         axes[0,0].set_title('Matrice de Confusion')
#         axes[0,0].set_xlabel('Prédiction')
#         axes[0,0].set_ylabel('Réalité')

#         fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#         roc_auc = roc_auc_score(y_true, y_pred_proba)
#         axes[0,1].plot(fpr, tpr, color='darkorange', lw=2,
#                          label=f'ROC curve (AUC = {roc_auc:.3f})')
#         axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         axes[0,1].set_xlim([0.0, 1.0])
#         axes[0,1].set_ylim([0.0, 1.05])
#         axes[0,1].set_xlabel('Taux de Faux Positifs')
#         axes[0,1].set_ylabel('Taux de Vrais Positifs')
#         axes[0,1].set_title('Courbe ROC')
#         axes[0,1].legend(loc="lower right")

#         precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
#         avg_precision = average_precision_score(y_true, y_pred_proba)
#         axes[0,2].plot(recall, precision, color='green', lw=2,
#                          label=f'PR curve (AP = {avg_precision:.3f})')
#         axes[0,2].set_xlim([0.0, 1.0])
#         axes[0,2].set_ylim([0.0, 1.05])
#         axes[0,2].set_xlabel('Recall')
#         axes[0,2].set_ylabel('Precision')
#         axes[0,2].set_title('Courbe Precision-Recall')
#         axes[0,2].legend(loc="lower left")

#         axes[1,0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Classe 0', color='blue')
#         axes[1,0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Classe 1', color='red')
#         axes[1,0].set_xlabel('Probabilité prédite')
#         axes[1,0].set_ylabel('Fréquence')
#         axes[1,0].set_title('Distribution des Probabilités')
#         axes[1,0].legend()

#         # Vérifier si threshold_df est vide avant de plotter
#         if not threshold_df.empty:
#             axes[1,1].plot(threshold_df['threshold'], threshold_df['f1'], label='F1-Score', color='blue')
#             axes[1,1].plot(threshold_df['threshold'], threshold_df['balanced_accuracy'], label='Balanced Accuracy', color='green')
#             axes[1,1].plot(threshold_df['threshold'], threshold_df['youden_index'], label='Youden Index', color='red')
#             axes[1,1].set_xlabel('Seuil')
#             axes[1,1].set_ylabel('Score')
#             axes[1,1].set_title('Optimisation des Seuils')
#             axes[1,1].legend()
#             axes[1,1].grid(True, alpha=0.3)
#         else:
#             axes[1,1].text(0.5, 0.5, 'Analyse des seuils\nnon disponible',
#                              ha='center', va='center', transform=axes[1,1].transAxes)
#             axes[1,1].set_title('Optimisation des Seuils - Non disponible')


#         if 'permutation' in importance_results:
#             perm_data = pd.DataFrame(importance_results['permutation'])
#             top_features = perm_data.head(15)

#             axes[1,2].barh(range(len(top_features)), top_features['importance_mean'])
#             axes[1,2].set_yticks(range(len(top_features)))
#             axes[1,2].set_yticklabels(top_features['feature'])
#             axes[1,2].set_xlabel('Importance (Permutation)')
#             axes[1,2].set_title('Top 15 Features Importantes (Permutation)')
#             axes[1,2].invert_yaxis()
#         else:
#             axes[1,2].text(0.5, 0.5, 'Permutation Importance\nnon disponible',
#                              ha='center', va='center', transform=axes[1,2].transAxes)
#             axes[1,2].set_title('Permutation Importance - Non disponible')

#         plt.tight_layout()

#         plot_path = os.path.join(self.plots_dir, f'analysis_complete_{strategy_name}_{self.current_subset_info}_{eval_set_name}.png')
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()

#         if 'permutation' in importance_results or 'built_in' in importance_results:
#             self.plot_detailed_feature_importance(strategy_name, importance_results, eval_set_name)

#     def plot_detailed_feature_importance(self, strategy_name, importance_results, eval_set_name="Validation"):
#         """Plot détaillé de l'importance des features"""
#         fig, axes = plt.subplots(1, 2, figsize=(20, 8))
#         fig.suptitle(f'Importance des Features - {strategy_name} ({eval_set_name} Set, Subset: {self.current_subset_info})', fontsize=16, fontweight='bold')

#         if 'permutation' in importance_results and importance_results['permutation']:
#             perm_data = pd.DataFrame(importance_results['permutation'])
#             top_20 = perm_data.head(20)

#             axes[0].barh(range(len(top_20)), top_20['importance_mean'],
#                          xerr=top_20['importance_std'], capsize=3)
#             axes[0].set_yticks(range(len(top_20)))
#             axes[0].set_yticklabels(top_20['feature'])
#             axes[0].set_xlabel('Importance Moyenne (avec écart-type)')
#             axes[0].set_title('Permutation Importance - Top 20')
#             axes[0].invert_yaxis()
#             axes[0].grid(True, alpha=0.3, axis='x')
#         else:
#             axes[0].text(0.5, 0.5, 'Permutation Importance\nnon disponible',
#                              ha='center', va='center', transform=axes[0].transAxes)
#             axes[0].set_title('Permutation Importance - Non disponible')

#         if 'built_in' in importance_results and importance_results['built_in']:
#             builtin_data = pd.DataFrame(importance_results['built_in'])
#             top_20_builtin = builtin_data.head(20)

#             axes[1].barh(range(len(top_20_builtin)), top_20_builtin['importance'])
#             axes[1].set_yticks(range(len(top_20_builtin)))
#             axes[1].set_yticklabels(top_20_builtin['feature'])
#             axes[1].set_xlabel('Importance Built-in')
#             axes[1].set_title('Random Forest Feature Importance - Top 20')
#             axes[1].invert_yaxis()
#             axes[1].grid(True, alpha=0.3, axis='x')
#         else:
#             axes[1].text(0.5, 0.5, 'Importance built-in\nnon disponible',
#                              ha='center', va='center', transform=axes[1].transAxes)
#             axes[1].set_title('Built-in Importance - Non disponible')

#         plt.tight_layout()

#         plot_path = os.path.join(self.plots_dir, f'feature_importance_detailed_{strategy_name}_{self.current_subset_info}_{eval_set_name}.png')
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()

#     def optimize_threshold_rigorous(self, y_true, y_pred_proba, strategy_name, eval_set_name="Validation"):
#         """Optimisation rigoureuse des seuils"""
#         print(f"🎯 Optimisation des seuils pour {strategy_name} ({eval_set_name} Set)...")

#         thresholds = np.arange(0.05, 0.95, 0.01)
#         metrics_results = {
#             'threshold': [],
#             'precision': [],
#             'recall': [],
#             'f1': [],
#             'specificity': [],
#             'balanced_accuracy': [],
#             'youden_index': []
#         }

#         for threshold in tqdm(thresholds, desc=f"Test des seuils ({eval_set_name})"):
#             y_pred_thresh = (y_pred_proba >= threshold).astype(int)

#             precision = precision_score(y_true, y_pred_thresh, zero_division=0)
#             recall = recall_score(y_true, y_pred_thresh, zero_division=0)
#             f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
#             balanced_acc = balanced_accuracy_score(y_true, y_pred_thresh)

#             tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
#             fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
#             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#             youden = recall + specificity - 1

#             metrics_results['threshold'].append(threshold)
#             metrics_results['precision'].append(precision)
#             metrics_results['recall'].append(recall)
#             metrics_results['f1'].append(f1)
#             metrics_results['specificity'].append(specificity)
#             metrics_results['balanced_accuracy'].append(balanced_acc)
#             metrics_results['youden_index'].append(youden)

#         results_df = pd.DataFrame(metrics_results)

#         optimal_thresholds = {
#             'f1': {
#                 'threshold': results_df.loc[results_df['f1'].idxmax(), 'threshold'],
#                 'score': results_df['f1'].max()
#             },
#             'balanced_accuracy': {
#                 'threshold': results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold'],
#                 'score': results_df['balanced_accuracy'].max()
#             },
#             'youden': {
#                 'threshold': results_df.loc[results_df['youden_index'].idxmax(), 'threshold'],
#                 'score': results_df['youden_index'].max()
#             },
#             'precision': {
#                 'threshold': results_df.loc[results_df['precision'].idxmax(), 'threshold'],
#                 'score': results_df['precision'].max()
#             },
#             'recall': {
#                 'threshold': results_df.loc[results_df['recall'].idxmax(), 'threshold'],
#                 'score': results_df['recall'].max()
#             }
#         }

#         results_df.to_csv(
#             os.path.join(self.metrics_dir, f'threshold_analysis_{strategy_name}_{self.current_subset_info}_{eval_set_name}.csv'),
#             index=False
#         )

#         return optimal_thresholds, results_df

#     def cross_validate_strategy(self, X, y, pipeline, strategy_name, cv_folds=5):
#         """Validation croisée rigoureuse d'une stratégie"""
#         print(f"🔄 Validation croisée pour {strategy_name}...")

#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

#         scoring = {
#             'roc_auc': 'roc_auc',
#             'precision': make_scorer(precision_score, zero_division=0),
#             'recall': make_scorer(recall_score, zero_division=0),
#             'f1': make_scorer(f1_score, zero_division=0),
#             'balanced_accuracy': 'balanced_accuracy'
#         }

#         cv_results = {}
#         for metric_name, scorer in scoring.items():
#             scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
#             cv_results[metric_name] = {
#                 'mean': scores.mean(),
#                 'std': scores.std(),
#                 'scores': scores.tolist()
#             }
#             print(f"    {metric_name}: {scores.mean():.3f} (±{scores.std():.3f})")

#         return cv_results

#     def holdout_validation(self, df_rf, features, target, patient_id_col='pseudo_provisoire', test_size=0.15, val_size=0.15):
#         """Division rigoureuse Train/Validation/Test basée sur les patients"""
#         print("📊 Division Train/Validation/Test par patients...")

#         unique_patients = df_rf[patient_id_col].unique()

#         train_val_patients, holdout_patients = train_test_split(
#             unique_patients, test_size=test_size, random_state=42
#         )

#         val_size_adjusted = val_size / (1 - test_size)
#         train_patients, val_patients = train_test_split(
#             train_val_patients, test_size=val_size_adjusted, random_state=42
#         )

#         df_train = df_rf[df_rf[patient_id_col].isin(train_patients)]
#         df_val = df_rf[df_rf[patient_id_col].isin(val_patients)]
#         df_holdout = df_rf[df_rf[patient_id_col].isin(holdout_patients)]

#         print(f"    Train: {len(df_train)} lignes, {len(train_patients)} patients")
#         print(f"    Val: {len(df_val)} lignes, {len(val_patients)} patients")
#         print(f"    Holdout: {len(df_holdout)} lignes, {len(holdout_patients)} patients")

#         split_data = {
#             'train_patients': train_patients.tolist(),
#             'val_patients': val_patients.tolist(),
#             'holdout_patients': holdout_patients.tolist(),
#         }

#         with open(os.path.join(self.data_dir, f'data_splits_{self.current_subset_info}.json'), 'w') as f:
#             json.dump(split_data, f, indent=2)

#         X_train = df_train[features]
#         y_train = df_train[target]
#         X_val = df_val[features]
#         y_val = df_val[target]
#         X_holdout = df_holdout[features]
#         y_holdout = df_holdout[target]

#         # Stocker les données holdout dans l'instance pour une utilisation ultérieure
#         self.holdout_data[self.current_subset_info] = {'X': X_holdout, 'y': y_holdout}

#         return X_train, X_val, X_holdout, y_train, y_val, y_holdout


#     def save_strategy_results(self, strategy_name, results):
#         """Sauvegarde les résultats d'une stratégie individuelle, incluant l'info du subset."""
#         strategy_dir = os.path.join(self.output_dir, f"strategy_{strategy_name}")
#         os.makedirs(strategy_dir, exist_ok=True)

#         pipeline_path = os.path.join(self.models_dir, f'pipeline_{strategy_name}_{self.current_subset_info}.pkl')
#         joblib.dump(results['pipeline'], pipeline_path)

#         metrics_path = os.path.join(self.metrics_dir, f'metrics_{strategy_name}_{self.current_subset_info}.json')
#         metrics_data = {
#             'cv_results': results['cv_results'],
#             'validation_metrics': results['validation_metrics'],
#             'optimal_thresholds': results['optimal_thresholds'],
#             'feature_importance': results.get('feature_importance', {}),
#             'timestamp': dt.now().isoformat(),
#             'subset_info': self.current_subset_info,
#             'hyperparameter_tuning': results.get('hyperparameter_tuning', {})
#         }

#         with open(metrics_path, 'w') as f:
#             json.dump(metrics_data, f, indent=2, default=str)

#         print(f"💾 Résultats de {strategy_name} pour subset '{self.current_subset_info}' sauvegardés")

#     def evaluate_all_strategies(self, df_rf, features, target='etat_critique',
#                                  resume_experiment_id=None, output_dir=None,
#                                  subset_col=None, subset_values=None):
#         """
#         Évaluation complète et rigoureuse de toutes les stratégies avec persistance,
#         avec la possibilité d'exécuter sur des sous-groupes de données.
#         """

#         print("🚀 DÉBUT DE L'ÉVALUATION RIGOUREUSE")
#         print("="*60)

#         if subset_col and subset_values:
#             subsets_to_process = {value: df_rf[df_rf[subset_col] == value] for value in subset_values}
#         elif subset_col:
#             subsets_to_process = {value: df_rf[df_rf[subset_col] == value] for value in df_rf[subset_col].unique()}
#         else:
#             subsets_to_process = {"all_data": df_rf}

#         all_overall_results = {}

#         # Grilles de paramètres mises à jour pour SMOTEENN et SMOTETomek
#         # Basé sur la convention d'Imblearn pour l'accès aux sous-objets
#         param_grids = {
#             'balanced_rf': {
#                 'classifier__n_estimators': [200, 300, 500],
#                 'classifier__max_depth': [10, 15, 20, None],
#                 'classifier__min_samples_split': [2, 5, 10],
#                 'classifier__min_samples_leaf': [1, 2, 4],
#                 'classifier__max_features': ['sqrt', 0.8],
#                 'classifier__sampling_strategy': ['auto', 'majority'],
#                 'classifier__replacement': [True, False]
#             },
#             'random_oversample': {
#                 'resampler__sampling_strategy': ['auto', 0.5, 0.75, 1.0],
#                 'classifier__n_estimators': [200, 300, 500],
#                 'classifier__max_depth': [10, 15, 20, None],
#                 'classifier__min_samples_split': [2, 5, 10],
#                 'classifier__min_samples_leaf': [1, 2, 4],
#                 'classifier__max_features': ['sqrt', 0.8]
#             },
#             'random_undersample': {
#                 'resampler__sampling_strategy': ['auto', 0.5, 0.75, 1.0],
#                 'classifier__n_estimators': [200, 300, 500],
#                 'classifier__max_depth': [10, 15, 20, None],
#                 'classifier__min_samples_split': [2, 5, 10],
#                 'classifier__min_samples_leaf': [1, 2, 4],
#                 'classifier__max_features': ['sqrt', 0.8]
#             },
#             'smote': {
#                 'resampler__sampling_strategy': ['auto', 0.75, 1.0],
#                 'resampler__k_neighbors': [3, 5, 7],
#                 'classifier__n_estimators': [200, 400],
#                 'classifier__max_depth': [10, 15],
#                 'classifier__max_features': ['sqrt']
#             },
#             'smoteenn': {
#                 # SMOTEENN est une combinaison de SMOTE et EditedNearestNeighbours (ENN)
#                 # Ses paramètres internes sont accessibles via smote__ et enn__
#                 'resampler__smote__k_neighbors': [3, 5],
#                 'resampler__enn__n_neighbors': [3, 5], # n_neighbors pour EditedNearestNeighbours
#                 'classifier__n_estimators': [200, 400],
#                 'classifier__max_depth': [10, 15],
#                 'classifier__max_features': ['sqrt']
#             },
#             'smotetomek': {
#                 # SMOTETomek est une combinaison de SMOTE et TomekLinks
#                 # Ses paramètres internes sont accessibles via smote__ et tomek__
#                 'resampler__smote__k_neighbors': [3, 5],
#                 'resampler__tomek__sampling_strategy': ['auto', 'majority'],
#                 'classifier__n_estimators': [200, 400],
#                 'classifier__max_depth': [10, 15],
#                 'classifier__max_features': ['sqrt']
#             }
#         }


#         for subset_name, subset_df in subsets_to_process.items():
#             self.current_subset_info = subset_name
#             print(f"\n{'#'*70}")
#             print(f"## Démarrage de l'évaluation pour le subset: '{subset_name}' ({len(subset_df)} lignes) ##")
#             print(f"{'#'*70}")

#             if subset_df.empty:
#                 print(f"⚠️ Le subset '{subset_name}' est vide, passage au suivant.")
#                 continue

#             current_output_base = output_dir if output_dir else self.base_output_path
#             self.create_output_directory(resume_experiment_id=resume_experiment_id, subset_name=subset_name)

#             if resume_experiment_id:
#                 self.load_experiment_state()

#             X = subset_df[features]
#             numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
#             categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

#             print(f"Features numériques: {len(numerical_features)}") # Continuation of the print statement
#             print(f"Features catégorielles: {len(categorical_features)}")

#             # Division des données en Train/Validation/Holdout
#             X_train, X_val, X_holdout, y_train, y_val, y_holdout = \
#                 self.holdout_validation(subset_df, features, target)

#             # Enregistrement des données holdout pour ce subset
#             self.holdout_data[subset_name] = {'X': X_holdout, 'y': y_holdout}


#             # Pour chaque stratégie de rééquilibrage
#             strategies = self.prepare_resampling_strategies()
#             all_results_for_subset = self.load_existing_results() # Charger les résultats existants pour le subset

#             for strategy_name, strategy in strategies.items():
#                 if strategy_name in self.completed_strategies:
#                     print(f"\n⏭️ {strategy_name} déjà complétée pour le subset '{subset_name}', passage à la suivante...")
#                     continue

#                 print(f"\n{'='*60}")
#                 print(f"🧪 ÉVALUATION DE LA STRATÉGIE: {strategy_name.upper()} pour subset '{subset_name}'")
#                 print(f"{'='*60}")

#                 pipeline_to_evaluate = None
#                 best_params = {}
#                 best_score = None
#                 search_cv_results = {}

#                 try:
#                     # Correction ici : Passez 'current_strategy' comme argument nommé
#                     self.save_experiment_state(all_results_for_subset, current_strategy=strategy_name)

#                     if strategy_name in param_grids:
#                         pipeline_to_evaluate, best_params, best_score, search_cv_results = self.tune_hyperparameters(
#                             X_train, y_train, numerical_features, categorical_features,
#                             strategy_name, strategy, param_grids[strategy_name],
#                             search_type='random',
#                             scoring='f1'
#                         )
#                         print(f"Best pipeline pour {strategy_name} obtenu après tuning.")
#                     else:
#                         print(f"Pas de réglage d'hyperparamètres défini pour {strategy_name}. Utilisation des paramètres par défaut.")
#                         # Créer un pipeline avec les paramètres par défaut
#                         pipeline_to_evaluate = self.create_pipeline(numerical_features, categorical_features, strategy_name, strategy)

#                     # Cross-validation sur l'ensemble d'entraînement pour le pipeline optimal
#                     cv_metrics = self.cross_validate_strategy(X_train, y_train, pipeline_to_evaluate, strategy_name)

#                     # Évaluation sur l'ensemble de validation
#                     y_val_pred = pipeline_to_evaluate.predict(X_val)
#                     y_val_proba = pipeline_to_evaluate.predict_proba(X_val)[:, 1]

#                     val_clf_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
#                     val_roc_auc = roc_auc_score(y_val, y_val_proba)
#                     val_avg_precision = average_precision_score(y_val, y_val_proba)
#                     val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)

#                     val_metrics = {
#                         'classification_report': val_clf_report,
#                         'roc_auc': val_roc_auc,
#                         'average_precision_score': val_avg_precision,
#                         'balanced_accuracy': val_balanced_acc
#                     }
#                     print(f"\n📈 Métriques de validation pour {strategy_name} :")
#                     print(f"   ROC AUC: {val_roc_auc:.3f}")
#                     print(f"   Balanced Accuracy: {val_balanced_acc:.3f}")
#                     print(f"   F1-score (Class 1): {val_clf_report['1']['f1-score']:.3f}")

#                     # Optimisation des seuils sur l'ensemble de validation
#                     optimal_thresholds, threshold_df = self.optimize_threshold_rigorous(y_val, y_val_proba, strategy_name)
#                     print(f"   Seuils optimaux (F1-score): {optimal_thresholds['f1']['threshold']:.2f}")

#                     # Calcul de l'importance des features
#                     feature_importance_results = self.calculate_feature_importance(
#                         pipeline_to_evaluate, X_val, y_val, strategy_name, features
#                     )

#                     # Sauvegarde des résultats
#                     results_to_save = {
#                         'pipeline': pipeline_to_evaluate,
#                         'best_params': best_params,
#                         'best_score': best_score,
#                         'cv_results': cv_metrics,
#                         'validation_metrics': val_metrics,
#                         'optimal_thresholds': optimal_thresholds,
#                         'feature_importance': feature_importance_results,
#                         'hyperparameter_tuning': search_cv_results
#                     }
#                     self.save_strategy_results(strategy_name, results_to_save)

#                     # Ajout aux résultats complets
#                     all_results_for_subset[strategy_name] = {
#                         'best_score': best_score,
#                         'best_params': best_params,
#                         'validation_metrics': val_metrics,
#                         'optimal_thresholds': optimal_thresholds
#                     }

#                     # Plots
#                     self.plot_comprehensive_analysis(
#                         strategy_name, y_val, y_val_pred, y_val_proba,
#                         feature_importance_results, threshold_df, features, eval_set_name="Validation"
#                     )

#                     # Ajout à la liste des stratégies complétées
#                     self.completed_strategies.add(strategy_name)
#                     # Enregistrez l'état après chaque stratégie réussie
#                     self.save_experiment_state(all_results_for_subset, current_strategy=strategy_name)


#                 except Exception as e:
#                     print(f"❌ Erreur avec {strategy_name} pour subset '{subset_name}': {str(e)}")
#                     error_info = {
#                         'strategy': strategy_name,
#                         'subset': subset_name,
#                         'error': str(e),
#                         'timestamp': dt.now().isoformat()
#                     }
#                     all_results_for_subset[strategy_name] = {'error': str(e), 'timestamp': dt.now().isoformat()}
#                     # Correction ici : Passez 'current_strategy' comme argument nommé pour l'état d'erreur
#                     self.save_experiment_state(all_results_for_subset, current_strategy="ERROR")
#                     continue # Passer à la stratégie suivante en cas d'erreur


#             # Évaluation finale sur l'ensemble Holdout une fois toutes les stratégies terminées
#             print(f"\n{'='*70}")
#             print(f"## ÉVALUATION SUR L'ENSEMBLE HOLDOUT pour le subset: '{subset_name}' ##")
#             print(f"{'='*70}")

#             if subset_name in self.holdout_data:
#                 X_holdout = self.holdout_data[subset_name]['X']
#                 y_holdout = self.holdout_data[subset_name]['y']

#                 for strategy_name, strategy_results in all_results_for_subset.items():
#                     if 'pipeline' in strategy_results: # Vérifier si le pipeline a été sauvegardé
#                         try:
#                             # Charger le pipeline depuis le disque si ce n'est pas déjà le cas
#                             pipeline_path = os.path.join(self.models_dir, f'pipeline_{strategy_name}_{self.current_subset_info}.pkl')
#                             if os.path.exists(pipeline_path):
#                                 final_pipeline = joblib.load(pipeline_path)
#                             else:
#                                 print(f"⚠️ Pipeline pour {strategy_name} non trouvé à {pipeline_path}. Skip holdout evaluation.")
#                                 continue

#                             print(f"\n📊 Évaluation Holdout pour {strategy_name}:")
#                             y_holdout_pred = final_pipeline.predict(X_holdout)
#                             y_holdout_proba = final_pipeline.predict_proba(X_holdout)[:, 1]

#                             holdout_clf_report = classification_report(y_holdout, y_holdout_pred, output_dict=True, zero_division=0)
#                             holdout_roc_auc = roc_auc_score(y_holdout, y_holdout_proba)
#                             holdout_avg_precision = average_precision_score(y_holdout, y_holdout_proba)
#                             holdout_balanced_acc = balanced_accuracy_score(y_holdout, y_holdout_pred)

#                             holdout_metrics = {
#                                 'classification_report': holdout_clf_report,
#                                 'roc_auc': holdout_roc_auc,
#                                 'average_precision_score': holdout_avg_precision,
#                                 'balanced_accuracy': holdout_balanced_acc
#                             }
#                             print(f"   ROC AUC (Holdout): {holdout_roc_auc:.3f}")
#                             print(f"   Balanced Accuracy (Holdout): {holdout_balanced_acc:.3f}")
#                             print(f"   F1-score (Class 1, Holdout): {holdout_clf_report['1']['f1-score']:.3f}")

#                             # Optional: Recalculate optimal thresholds on holdout if needed, or use validation's best threshold
#                             # For a rigorous holdout, you generally fix the threshold from validation set.
#                             # For simplicity, here we'll just log the metrics with default (0.5) threshold.
#                             # You might want to apply the optimal threshold found on validation set here.

#                             # Recalcul de l'importance des features sur Holdout si désiré (attention aux coûts)
#                             # holdout_feature_importance_results = self.calculate_feature_importance(
#                             #     final_pipeline, X_holdout, y_holdout, strategy_name, features
#                             # )

#                             # Plots pour l'ensemble Holdout
#                             # Re-optimize threshold for plotting on holdout only if needed, otherwise use best from validation
#                             optimal_thresholds_holdout, threshold_df_holdout = self.optimize_threshold_rigorous(y_holdout, y_holdout_proba, strategy_name, eval_set_name="Holdout")
#                             # Decide if you want to use y_holdout_pred based on optimal_thresholds_holdout['f1']['threshold'] or default 0.5
#                             y_holdout_pred_optimized = (y_holdout_proba >= optimal_thresholds_holdout['f1']['threshold']).astype(int)

#                             self.plot_comprehensive_analysis(
#                                 strategy_name, y_holdout, y_holdout_pred_optimized, y_holdout_proba,
#                                 feature_importance_results, threshold_df_holdout, features, eval_set_name="Holdout"
#                             )

#                             all_results_for_subset[strategy_name]['holdout_metrics'] = holdout_metrics

#                         except Exception as e:
#                             print(f"❌ Erreur lors de l'évaluation Holdout pour {strategy_name} : {str(e)}")
#                             all_results_for_subset[strategy_name]['holdout_metrics'] = {'error': str(e)}
#             else:
#                 print(f"⚠️ Données Holdout non disponibles pour le subset '{subset_name}'. Skipping holdout evaluation.")

#             # Enregistrement des résultats globaux pour le subset
#             all_overall_results[subset_name] = all_results_for_subset
#             with open(os.path.join(self.output_dir, f"comprehensive_experiment_results_{subset_name}.json"), 'w') as f:
#                 json.dump({'detailed_results': all_overall_results[subset_name]}, f, indent=2, default=str)

#         print("\n============================================================")
#         print("🎉 ÉVALUATION RIGOUREUSE TERMINÉE POUR TOUS LES SUBSETS 🎉")
#         print("============================================================")
#         return all_overall_results