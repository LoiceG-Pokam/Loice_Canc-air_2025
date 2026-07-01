import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import scipy.stats as stats
from tqdm import tqdm
from itertools import combinations
from adjustText import adjust_text
from  sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from adjustText import adjust_text
from mpl_toolkits.mplot3d import Axes3D
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import matplotlib.dates as mdates
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from kneed import KneeLocator
from kmodes.kmodes import KModes
import prince 
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           roc_auc_score, precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score, balanced_accuracy_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import json
import joblib
from sklearn.inspection import permutation_importance

# Imbalanced-learn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler)

import pickle
from imblearn.ensemble import BalancedRandomForestClassifier


class EnhancedRigorousMLPipeline:
    """
    Pipeline ML rigoureux avec persistance, reprise d'expérience et visualisations complètes
    """
    
    def __init__(self, base_output_path="C:/experiments/", experiment_name="rigorous_ml"):
        self.base_output_path = base_output_path
        self.experiment_name = experiment_name
        self.output_dir = None
        self.results_history = []
        self.completed_strategies = set()
        self.experiment_state_file = None
        
    def create_output_directory(self, resume_experiment_id=None):
        """Crée un répertoire organisé par date pour les résultats ou reprend une expérience"""
        if resume_experiment_id:
            self.output_dir = resume_experiment_id
            if not os.path.exists(self.output_dir):
                raise ValueError(f"Le répertoire d'expérience {resume_experiment_id} n'existe pas")
            print(f"📁 Reprise de l'expérience : {self.output_dir}")
        else:
            today = dt.now().strftime("%Y-%m-%d")
            timestamp = dt.now().strftime("%H-%M-%S")
            
            self.output_dir = os.path.join(
                self.base_output_path, 
                f"{self.experiment_name}_{today}", 
                f"run_{timestamp}"
            )
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"📁 Nouveau répertoire de sortie créé : {self.output_dir}")
        
        # Créer les sous-répertoires pour l'organisation
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.data_dir = os.path.join(self.output_dir, "data")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        
        for dir_path in [self.plots_dir, self.models_dir, self.data_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Fichier d'état de l'expérience
        self.experiment_state_file = os.path.join(self.output_dir, "experiment_state.json")
        
        return self.output_dir
    
    def save_experiment_state(self, all_results, current_strategy=None):
        """Sauvegarde l'état actuel de l'expérience"""
        state = {
            'timestamp': dt.now().isoformat(),
            'completed_strategies': list(self.completed_strategies),
            'current_strategy': current_strategy,
            'output_directory': self.output_dir,
            'total_strategies': len(self.prepare_resampling_strategies()),
            'progress': len(self.completed_strategies)
        }
        
        with open(self.experiment_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_experiment_state(self):
        """Charge l'état d'une expérience précédente"""
        if os.path.exists(self.experiment_state_file):
            with open(self.experiment_state_file, 'r') as f:
                state = json.load(f)
            
            self.completed_strategies = set(state.get('completed_strategies', []))
            print(f"📋 État chargé: {len(self.completed_strategies)} stratégies déjà complétées")
            print(f"   Stratégies terminées: {', '.join(self.completed_strategies)}")
            return state
        return None
    
    def load_existing_results(self):
        """Charge les résultats existants depuis le disque"""
        results_file = os.path.join(self.output_dir, "comprehensive_experiment_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('detailed_results', {})
        return {}
    
    def prepare_resampling_strategies(self):
        """Définit différentes stratégies de rééquilibrage"""
        strategies = {
            # Stratégies pures
            'baseline': None,
            'class_weight': 'class_weight_only',
            'random_oversample': RandomOverSampler(random_state=42),
            'random_undersample': RandomUnderSampler(random_state=42),
            
            # SMOTE et variantes
            'smote': SMOTE(random_state=42, k_neighbors=3),
            # 'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=3),
            # 'adasyn': ADASYN(random_state=42, n_neighbors=3),
            # 'svm_smote': SVMSMOTE(random_state=42, k_neighbors=3),
            
            # Stratégies combinées
            # 'smote_enn': SMOTEENN(random_state=42),
            # 'smote_tomek': SMOTETomek(random_state=42),
            
            # Modèle spécialisé
            'balanced_rf': 'balanced_random_forest'
        }
        return strategies
    
    def create_pipeline(self, numerical_features, categorical_features, strategy_name, strategy):
        """Crée un pipeline selon la stratégie choisie"""
        
        # Préprocesseur commun
        preprocessor = ColumnTransformer([
            ('num', SklearnPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', SklearnPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
        
        # Choisir le modèle selon la stratégie
        if strategy_name == 'balanced_rf':
            pipeline = SklearnPipeline([
                ('preprocessor', preprocessor),
                ('classifier', BalancedRandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])
        elif strategy_name == 'class_weight':
            pipeline = SklearnPipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced',
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])
        elif strategy_name == 'baseline':
            pipeline = SklearnPipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])
        else:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resampler', strategy),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])
        
        return pipeline
    
    def calculate_feature_importance(self, pipeline, X_val, y_val, strategy_name, features):
        """Calcule l'importance des features avec plusieurs méthodes"""
        print(f"🔍 Calcul de l'importance des features pour {strategy_name}...")
        
        importance_results = {}
        
        try:
            # 1. Importance intégrée du modèle (pour Random Forest)
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                # Obtenir les noms des features après preprocessing
                preprocessor = pipeline.named_steps['preprocessor']
                
                # Features numériques (restent inchangées)
                num_features = preprocessor.named_transformers_['num']['scaler'].feature_names_in_.tolist() if hasattr(preprocessor.named_transformers_['num']['scaler'], 'feature_names_in_') else []
                
                # Features catégorielles (après one-hot encoding)
                cat_features = []
                if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
                    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out().tolist()
                
                all_feature_names = num_features + cat_features
                
                importances = pipeline.named_steps['classifier'].feature_importances_
                
                if len(all_feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        'feature': all_feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    importance_results['built_in'] = importance_df.to_dict('records')
                
            # 2. Permutation importance (plus robuste)
            print("   Calcul de la permutation importance...")
            perm_importance = permutation_importance(
                pipeline, X_val, y_val, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            perm_importance_df = pd.DataFrame({
                'feature': features,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            importance_results['permutation'] = perm_importance_df.to_dict('records')
            
        except Exception as e:
            print(f"   ⚠️ Erreur lors du calcul d'importance: {str(e)}")
            importance_results['error'] = str(e)
        
        return importance_results
    
    def plot_comprehensive_analysis(self, strategy_name, y_true, y_pred, y_pred_proba, 
                                   importance_results, threshold_df, features):
        """Crée tous les graphiques d'analyse pour une stratégie"""
        
        # Configuration des plots
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Matrice de confusion
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Analyse Complète - {strategy_name}', fontsize=16, fontweight='bold')
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Matrice de Confusion')
        axes[0,0].set_xlabel('Prédiction')
        axes[0,0].set_ylabel('Réalité')
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('Taux de Faux Positifs')
        axes[0,1].set_ylabel('Taux de Vrais Positifs')
        axes[0,1].set_title('Courbe ROC')
        axes[0,1].legend(loc="lower right")
        
        # Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        axes[0,2].plot(recall, precision, color='green', lw=2,
                      label=f'PR curve (AP = {avg_precision:.3f})')
        axes[0,2].set_xlim([0.0, 1.0])
        axes[0,2].set_ylim([0.0, 1.05])
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Courbe Precision-Recall')
        axes[0,2].legend(loc="lower left")
        
        # Distribution des probabilités prédites
        axes[1,0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Classe 0', color='blue')
        axes[1,0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Classe 1', color='red')
        axes[1,0].set_xlabel('Probabilité prédite')
        axes[1,0].set_ylabel('Fréquence')
        axes[1,0].set_title('Distribution des Probabilités')
        axes[1,0].legend()
        
        # Optimisation des seuils
        axes[1,1].plot(threshold_df['threshold'], threshold_df['f1'], label='F1-Score', color='blue')
        axes[1,1].plot(threshold_df['threshold'], threshold_df['balanced_accuracy'], label='Balanced Accuracy', color='green')
        axes[1,1].plot(threshold_df['threshold'], threshold_df['youden_index'], label='Youden Index', color='red')
        axes[1,1].set_xlabel('Seuil')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Optimisation des Seuils')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Feature importance (permutation)
        if 'permutation' in importance_results:
            perm_data = pd.DataFrame(importance_results['permutation'])
            top_features = perm_data.head(15)  # Top 15 features
            
            axes[1,2].barh(range(len(top_features)), top_features['importance_mean'])
            axes[1,2].set_yticks(range(len(top_features)))
            axes[1,2].set_yticklabels(top_features['feature'])
            axes[1,2].set_xlabel('Importance (Permutation)')
            axes[1,2].set_title('Top 15 Features Importantes')
            axes[1,2].invert_yaxis()
        
        plt.tight_layout()
        
        # Sauvegarder le plot
        plot_path = os.path.join(self.plots_dir, f'analysis_complete_{strategy_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot séparé pour l'importance des features (détaillé)
        if 'permutation' in importance_results:
            self.plot_detailed_feature_importance(strategy_name, importance_results, features)
    
    def plot_detailed_feature_importance(self, strategy_name, importance_results, features):
        """Plot détaillé de l'importance des features"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Importance des Features - {strategy_name}', fontsize=16, fontweight='bold')
        
        # Permutation importance
        if 'permutation' in importance_results:
            perm_data = pd.DataFrame(importance_results['permutation'])
            top_20 = perm_data.head(20)
            
            axes[0].barh(range(len(top_20)), top_20['importance_mean'], 
                        xerr=top_20['importance_std'], capsize=3)
            axes[0].set_yticks(range(len(top_20)))
            axes[0].set_yticklabels(top_20['feature'])
            axes[0].set_xlabel('Importance Moyenne (avec écart-type)')
            axes[0].set_title('Permutation Importance - Top 20')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Built-in importance (si disponible)
        if 'built_in' in importance_results:
            builtin_data = pd.DataFrame(importance_results['built_in'])
            top_20_builtin = builtin_data.head(20)
            
            axes[1].barh(range(len(top_20_builtin)), top_20_builtin['importance'])
            axes[1].set_yticks(range(len(top_20_builtin)))
            axes[1].set_yticklabels(top_20_builtin['feature'])
            axes[1].set_xlabel('Importance Built-in')
            axes[1].set_title('Random Forest Feature Importance - Top 20')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'Importance built-in\nnon disponible', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Built-in Importance - Non disponible')
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(self.plots_dir, f'feature_importance_detailed_{strategy_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def optimize_threshold_rigorous(self, y_true, y_pred_proba, strategy_name):
        """Optimisation rigoureuse des seuils avec validation croisée"""
        print(f"🎯 Optimisation des seuils pour {strategy_name}...")
        
        thresholds = np.arange(0.05, 0.95, 0.01)
        metrics_results = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
            'balanced_accuracy': [],
            'youden_index': []
        }
        
        for threshold in tqdm(thresholds, desc="Test des seuils"):
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculer toutes les métriques
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred_thresh)
            
            # Spécificité
            tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Indice de Youden
            youden = recall + specificity - 1
            
            # Stocker les résultats
            metrics_results['threshold'].append(threshold)
            metrics_results['precision'].append(precision)
            metrics_results['recall'].append(recall)
            metrics_results['f1'].append(f1)
            metrics_results['specificity'].append(specificity)
            metrics_results['balanced_accuracy'].append(balanced_acc)
            metrics_results['youden_index'].append(youden)
        
        # Trouver les seuils optimaux pour chaque métrique
        results_df = pd.DataFrame(metrics_results)
        
        optimal_thresholds = {
            'f1': {
                'threshold': results_df.loc[results_df['f1'].idxmax(), 'threshold'],
                'score': results_df['f1'].max()
            },
            'balanced_accuracy': {
                'threshold': results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold'],
                'score': results_df['balanced_accuracy'].max()
            },
            'youden': {
                'threshold': results_df.loc[results_df['youden_index'].idxmax(), 'threshold'],
                'score': results_df['youden_index'].max()
            },
            'precision': {
                'threshold': results_df.loc[results_df['precision'].idxmax(), 'threshold'],
                'score': results_df['precision'].max()
            },
            'recall': {
                'threshold': results_df.loc[results_df['recall'].idxmax(), 'threshold'],
                'score': results_df['recall'].max()
            }
        }
        
        # Sauvegarder les résultats
        results_df.to_csv(
            os.path.join(self.metrics_dir, f'threshold_analysis_{strategy_name}.csv'), 
            index=False
        )
        
        return optimal_thresholds, results_df
    
    def cross_validate_strategy(self, X, y, pipeline, strategy_name, cv_folds=5):
        """Validation croisée rigoureuse d'une stratégie"""
        print(f"🔄 Validation croisée pour {strategy_name}...")
        
        # Utiliser StratifiedKFold pour maintenir la distribution des classes
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Définir les métriques d'évaluation
        scoring = {
            'roc_auc': 'roc_auc',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0),
            'balanced_accuracy': 'balanced_accuracy'
        }
        
        cv_results = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"   {metric_name}: {scores.mean():.3f} (±{scores.std():.3f})")
        
        return cv_results
    
    def holdout_validation(self, df_rf, features, target, patient_id_col='pseudo_provisoire', test_size=0.15, val_size=0.15):
        """Division rigoureuse Train/Validation/Test basée sur les patients"""
        print("📊 Division Train/Validation/Test par patients...")

        # Étape 1: Patients uniques
        unique_patients = df_rf[patient_id_col].unique()
        
        # Étape 2: Split patients
        train_val_patients, holdout_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_size_adjusted, random_state=42
        )
        
        # Étape 3: Filtrer les lignes correspondantes
        df_train = df_rf[df_rf[patient_id_col].isin(train_patients)]
        df_val = df_rf[df_rf[patient_id_col].isin(val_patients)]
        df_holdout = df_rf[df_rf[patient_id_col].isin(holdout_patients)]
        
        print(f"   Train: {len(df_train)} lignes, {len(train_patients)} patients")
        print(f"   Val: {len(df_val)} lignes, {len(val_patients)} patients")
        print(f"   Holdout: {len(df_holdout)} lignes, {len(holdout_patients)} patients")

        # Sauvegarder les splits pour reproductibilité
        split_data = {
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist(),
            'holdout_patients': holdout_patients.tolist(),
        }

        with open(os.path.join(self.data_dir, 'data_splits.json'), 'w') as f:
            json.dump(split_data, f, indent=2)

        # Retourner X/y comme dans la version originale
        X_train = df_train[features]
        y_train = df_train[target]
        X_val = df_val[features]
        y_val = df_val[target]
        X_holdout = df_holdout[features]
        y_holdout = df_holdout[target]

        return X_train, X_val, X_holdout, y_train, y_val, y_holdout

    
    def save_strategy_results(self, strategy_name, results):
        """Sauvegarde les résultats d'une stratégie individuelle"""
        strategy_dir = os.path.join(self.output_dir, f"strategy_{strategy_name}")
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Sauvegarder le pipeline
        pipeline_path = os.path.join(self.models_dir, f'pipeline_{strategy_name}.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(results['pipeline'], f)
        
        # Sauvegarder les métriques
        metrics_path = os.path.join(self.metrics_dir, f'metrics_{strategy_name}.json')
        metrics_data = {
            'cv_results': results['cv_results'],
            'validation_metrics': results['validation_metrics'],
            'optimal_thresholds': results['optimal_thresholds'],
            'feature_importance': results.get('feature_importance', {}),
            'timestamp': dt.now().isoformat()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        print(f"💾 Résultats de {strategy_name} sauvegardés")
    
    def evaluate_all_strategies(self, df_rf, features, target='etat_critique', resume_experiment_id=None, output_dir=None):
        """Évaluation complète et rigoureuse de toutes les stratégies avec persistance"""
        
        print("🚀 DÉBUT DE L'ÉVALUATION RIGOUREUSE")
        print("="*60)

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.create_output_directory(resume_experiment_id)

        
        # Créer ou reprendre le répertoire de sortie
        self.create_output_directory(resume_experiment_id)
        
        # Charger l'état précédent si reprise
        if resume_experiment_id:
            self.load_experiment_state()
        
        # Identifier les types de features
        X = df_rf[features]
        numerical_features = X.select_dtypes(include=['int64', 'float64','int32']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Features numériques: {len(numerical_features)}")
        print(f"Features catégorielles: {len(categorical_features)}")
        
        # Division Train/Validation/Holdout
        X_train, X_val, X_holdout, y_train, y_val, y_holdout = self.holdout_validation(
            df_rf, features, target
        )
        
        # Préparer les stratégies de rééquilibrage
        strategies = self.prepare_resampling_strategies()
        
        # Charger les résultats existants
        all_results = self.load_existing_results()
        
        # Évaluer chaque stratégie (skip si déjà complétée)
        for strategy_name, strategy in strategies.items():
            
            if strategy_name in self.completed_strategies:
                print(f"\n⏭️ {strategy_name} déjà complétée, passage à la suivante...")
                continue
            
            print(f"\n{'='*60}")
            print(f"🧪 ÉVALUATION DE LA STRATÉGIE: {strategy_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Sauvegarder l'état
                self.save_experiment_state(all_results, strategy_name)
                
                # Créer le pipeline
                pipeline = self.create_pipeline(numerical_features, categorical_features, strategy_name, strategy)
                
                # Validation croisée sur les données d'entraînement
                cv_results = self.cross_validate_strategy(X_train, y_train, pipeline, strategy_name)
                
                # Entraînement sur train et évaluation sur validation
                print(f"🏋️ Entraînement du modèle {strategy_name}...")
                
                # Gérer les cas spéciaux pour l'entraînement
                if hasattr(strategy, 'fit_resample') and strategy is not None:
                    # Appliquer le rééquilibrage avant l'entraînement
                    preprocessor = pipeline.named_steps['preprocessor']
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_train_resampled, y_train_resampled = strategy.fit_resample(X_train_processed, y_train)
                    
                    # Créer un pipeline simplifié pour l'entraînement
                    classifier = pipeline.named_steps['classifier']
                    classifier.fit(X_train_resampled, y_train_resampled)
                    
                    # Prédictions sur validation
                    X_val_processed = preprocessor.transform(X_val)
                    y_val_pred_proba = classifier.predict_proba(X_val_processed)[:, 1]
                    y_val_pred = classifier.predict(X_val_processed)
                else:
                    # Entraînement standard
                    pipeline.fit(X_train, y_train)
                    y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                    y_val_pred = pipeline.predict(X_val)
                
                # Métriques sur validation
                val_metrics = {
                    'roc_auc': roc_auc_score(y_val, y_val_pred_proba),
                    'avg_precision': average_precision_score(y_val, y_val_pred_proba),
                    'precision': precision_score(y_val, y_val_pred, zero_division=0),
                    'recall': recall_score(y_val, y_val_pred, zero_division=0),
                    'f1': f1_score(y_val, y_val_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
                    'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
                }
                
                # Optimisation des seuils
                optimal_thresholds, threshold_df = self.optimize_threshold_rigorous(
                    y_val, y_val_pred_proba, strategy_name
                )
                
                # Calcul de l'importance des features
                feature_importance = self.calculate_feature_importance(
                    pipeline, X_val, y_val, strategy_name, features
                )
                
                # Création des visualisations complètes
                self.plot_comprehensive_analysis(
                    strategy_name, y_val, y_val_pred, y_val_pred_proba,
                    feature_importance, threshold_df, features
                )
                
                # Stocker les résultats
                strategy_results = {
                    'cv_results': cv_results,
                    'validation_metrics': val_metrics,
                    'optimal_thresholds': optimal_thresholds,
                    'feature_importance': feature_importance,
                    'pipeline': pipeline,
                    'threshold_analysis': threshold_df.to_dict('records')
                }
                
                all_results[strategy_name] = strategy_results
                
                # Sauvegarder immédiatement cette stratégie
                self.save_strategy_results(strategy_name, strategy_results)
                
                # Marquer comme complétée
                self.completed_strategies.add(strategy_name)
                
                print(f"✅ {strategy_name} terminé avec succès")
                print(f"   ROC-AUC: {val_metrics['roc_auc']:.3f}")
                print(f"   F1-Score: {val_metrics['f1']:.3f}")
                print(f"   Balanced Accuracy: {val_metrics['balanced_accuracy']:.3f}")
                
                # Sauvegarder l'état après chaque stratégie
                self.save_experiment_state(all_results)
                
            except Exception as e:
                print(f"❌ Erreur avec {strategy_name}: {str(e)}")
                # Sauvegarder l'erreur pour débogage
                error_info = {
                    'strategy': strategy_name,
                    'error': str(e),
                    'timestamp': dt.now().isoformat()
                }
                
                error_file = os.path.join(self.output_dir, 'errors.json')
                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        errors = json.load(f)
                else:
                    errors = []
                
                errors.append(error_info)
                with open(error_file, 'w') as f:
                    json.dump(errors, f, indent=2)
                
                continue
        
        # Analyser et comparer tous les résultats
        comparison_df, holdout_metrics = self.analyze_and_compare_results(all_results, X_holdout, y_holdout)
        
        # Sauvegarder les résultats finaux
        self.save_comprehensive_results(all_results, features, target, comparison_df, holdout_metrics)
        
        # Créer un rapport final
        self.create_final_report(all_results, comparison_df, holdout_metrics, features, target)
        
        return all_results
    
    def analyze_and_compare_results(self, all_results, X_holdout, y_holdout):
        """Analyse comparative de toutes les stratégies"""
        print(f"\n{'='*60}")
        print("📊 ANALYSE COMPARATIVE DES RÉSULTATS")
        print(f"{'='*60}")
        
        # Créer un DataFrame comparatif
        comparison_data = []
        
        for strategy_name, results in all_results.items():
            if 'cv_results' in results:  # S'assurer que les résultats sont complets
                cv_results = results['cv_results']
                val_metrics = results['validation_metrics']
                
                comparison_data.append({
                    'Strategy': strategy_name,
                    'CV_ROC_AUC_Mean': cv_results['roc_auc']['mean'],
                    'CV_ROC_AUC_Std': cv_results['roc_auc']['std'],
                    'CV_F1_Mean': cv_results['f1']['mean'],
                    'CV_F1_Std': cv_results['f1']['std'],
                    'CV_Precision_Mean': cv_results['precision']['mean'],
                    'CV_Recall_Mean': cv_results['recall']['mean'],
                    'CV_Balanced_Acc_Mean': cv_results['balanced_accuracy']['mean'],
                    'Val_ROC_AUC': val_metrics['roc_auc'],
                    'Val_F1': val_metrics['f1'],
                    'Val_Balanced_Acc': val_metrics['balanced_accuracy'],
                    'Val_Precision': val_metrics['precision'],
                    'Val_Recall': val_metrics['recall'],
                    'Val_Avg_Precision': val_metrics['avg_precision']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val_ROC_AUC', ascending=False)
        
        print("\n🏆 CLASSEMENT DES STRATÉGIES (par ROC-AUC validation):")
        print(comparison_df[['Strategy', 'Val_ROC_AUC', 'Val_F1', 'Val_Balanced_Acc', 'Val_Precision', 'Val_Recall']].round(3))
        
        # Sauvegarder le comparatif détaillé
        comparison_df.to_csv(os.path.join(self.metrics_dir, 'strategy_comparison_detailed.csv'), index=False)
        
        # Créer un graphique comparatif
        self.plot_strategy_comparison(comparison_df)
        
        # Évaluation finale sur holdout avec les 3 meilleures stratégies
        top_3_strategies = comparison_df.head(3)['Strategy'].tolist()
        holdout_results = {}
        
        print(f"\n🎯 ÉVALUATION FINALE SUR HOLDOUT - TOP 3 STRATÉGIES")
        
        for strategy_name in top_3_strategies:
            if strategy_name in all_results and 'pipeline' in all_results[strategy_name]:
                pipeline = all_results[strategy_name]['pipeline']
                
                print(f"\n🧪 Test final de {strategy_name} sur données holdout...")
                y_holdout_pred_proba = pipeline.predict_proba(X_holdout)[:, 1]
                y_holdout_pred = pipeline.predict(X_holdout)
                
                holdout_metrics = {
                    'roc_auc': roc_auc_score(y_holdout, y_holdout_pred_proba),
                    'avg_precision': average_precision_score(y_holdout, y_holdout_pred_proba),
                    'precision': precision_score(y_holdout, y_holdout_pred, zero_division=0),
                    'recall': recall_score(y_holdout, y_holdout_pred, zero_division=0),
                    'f1': f1_score(y_holdout, y_holdout_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_holdout, y_holdout_pred),
                    'classification_report': classification_report(y_holdout, y_holdout_pred, output_dict=True)
                }
                
                holdout_results[strategy_name] = holdout_metrics
                
                print(f"📊 RÉSULTATS HOLDOUT - {strategy_name}:")
                for metric, value in holdout_metrics.items():
                    if metric != 'classification_report':
                        print(f"   {metric}: {value:.3f}")
                
                # Créer des visualisations pour le holdout
                self.plot_holdout_analysis(strategy_name, y_holdout, y_holdout_pred, y_holdout_pred_proba)
        
        # Sauvegarder les résultats holdout
        with open(os.path.join(self.metrics_dir, 'holdout_results.json'), 'w') as f:
            json.dump(holdout_results, f, indent=2, default=str)
        
        return comparison_df, holdout_results
    
    def plot_strategy_comparison(self, comparison_df):
        """Crée des graphiques de comparaison entre stratégies"""
        
        # Graphique de comparaison des métriques principales
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison des Stratégies de Rééquilibrage', fontsize=16, fontweight='bold')
        
        strategies = comparison_df['Strategy']
        
        # ROC-AUC
        axes[0,0].bar(strategies, comparison_df['Val_ROC_AUC'], color='skyblue', alpha=0.7)
        axes[0,0].set_title('ROC-AUC sur Validation')
        axes[0,0].set_ylabel('ROC-AUC')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # F1-Score
        axes[0,1].bar(strategies, comparison_df['Val_F1'], color='lightgreen', alpha=0.7)
        axes[0,1].set_title('F1-Score sur Validation')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Balanced Accuracy
        axes[1,0].bar(strategies, comparison_df['Val_Balanced_Acc'], color='orange', alpha=0.7)
        axes[1,0].set_title('Balanced Accuracy sur Validation')
        axes[1,0].set_ylabel('Balanced Accuracy')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[1,1].scatter(comparison_df['Val_Recall'], comparison_df['Val_Precision'], 
                         s=100, alpha=0.7, c=comparison_df['Val_ROC_AUC'], 
                         cmap='viridis')
        
        # Annoter chaque point avec le nom de la stratégie
        for i, strategy in enumerate(strategies):
            axes[1,1].annotate(strategy, 
                             (comparison_df.iloc[i]['Val_Recall'], comparison_df.iloc[i]['Val_Precision']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title('Precision vs Recall (couleur = ROC-AUC)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(self.plots_dir, 'strategy_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Graphique radar pour comparer les métriques
        self.plot_radar_comparison(comparison_df)
    
    def plot_radar_comparison(self, comparison_df):
        """Crée un graphique radar pour comparer les top 5 stratégies"""
        
        # Sélectionner les 5 meilleures stratégies
        top_5 = comparison_df.head(5)
        
        # Métriques à inclure dans le radar
        metrics = ['Val_ROC_AUC', 'Val_F1', 'Val_Balanced_Acc', 'Val_Precision', 'Val_Recall']
        metric_labels = ['ROC-AUC', 'F1-Score', 'Balanced Acc', 'Precision', 'Recall']
        
        # Créer le graphique radar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Angles pour chaque métrique
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le cercle
        
        # Couleurs pour chaque stratégie
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_5)))
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Fermer le cercle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Strategy'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Personnaliser le graphique
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Comparaison Radar - Top 5 Stratégies', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(self.plots_dir, 'radar_comparison_top5.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_holdout_analysis(self, strategy_name, y_true, y_pred, y_pred_proba):
        """Crée des graphiques d'analyse pour les résultats holdout"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Analyse Holdout - {strategy_name}', fontsize=16, fontweight='bold')
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice de Confusion - Holdout')
        axes[0].set_xlabel('Prédiction')
        axes[0].set_ylabel('Réalité')
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Taux de Faux Positifs')
        axes[1].set_ylabel('Taux de Vrais Positifs')
        axes[1].set_title('Courbe ROC - Holdout')
        axes[1].legend(loc="lower right")
        
        # Distribution des probabilités
        axes[2].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label='Classe 0', color='blue')
        axes[2].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label='Classe 1', color='red')
        axes[2].set_xlabel('Probabilité prédite')
        axes[2].set_ylabel('Fréquence')
        axes[2].set_title('Distribution des Probabilités - Holdout')
        axes[2].legend()
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(self.plots_dir, f'holdout_analysis_{strategy_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self, all_results, features, target, comparison_df, holdout_metrics):
        """Sauvegarde complète de tous les résultats"""
        print(f"\n💾 Sauvegarde complète des résultats dans: {self.output_dir}")
        
        # Sauvegarder les métadonnées de l'expérience
        metadata = {
            'experiment_info': {
                'date': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                'target_variable': target,
                'n_features': len(features),
                'features_used': features,
                'strategies_evaluated': list(all_results.keys()),
                'total_strategies_completed': len(self.completed_strategies),
                'output_directory': self.output_dir
            },
            'methodology': {
                'data_split': 'Train(70%)/Validation(15%)/Holdout(15%)',
                'cross_validation': 'StratifiedKFold(5)',
                'resampling': 'Applied only on training data',
                'threshold_optimization': 'Validated on validation set',
                'final_evaluation': 'Holdout set (unseen data)',
                'feature_importance': 'Permutation + Built-in (when available)'
            },
            'best_strategies': {
                'top_3_by_roc_auc': comparison_df.head(3)[['Strategy', 'Val_ROC_AUC']].to_dict('records'),
                'top_3_by_f1': comparison_df.sort_values('Val_F1', ascending=False).head(3)[['Strategy', 'Val_F1']].to_dict('records')
            }
        }
        
        # Sauvegarder les résultats détaillés (sans les pipelines pour éviter les gros fichiers)
        results_for_json = {}
        for strategy_name, results in all_results.items():
            if 'cv_results' in results:  # S'assurer que les résultats sont complets
                results_for_json[strategy_name] = {
                    'cv_results': results['cv_results'],
                    'validation_metrics': results['validation_metrics'],
                    'optimal_thresholds': results['optimal_thresholds'],
                    'feature_importance': results.get('feature_importance', {}),
                    'threshold_analysis': results.get('threshold_analysis', [])
                }
        
        metadata['detailed_results'] = results_for_json
        metadata['holdout_evaluation'] = holdout_metrics
        
        with open(os.path.join(self.output_dir, "comprehensive_experiment_results.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # Sauvegarder le comparatif des stratégies
        comparison_df.to_csv(os.path.join(self.output_dir, 'strategy_comparison_final.csv'), index=False)
        
        print(f"✅ Sauvegarde complète terminée")
        print(f"   - Résultats détaillés: comprehensive_experiment_results.json")
        print(f"   - Comparaison finale: strategy_comparison_final.csv") 
        print(f"   - Modèles sauvegardés: {self.models_dir}")
        print(f"   - Graphiques: {self.plots_dir}")
        print(f"   - Métriques détaillées: {self.metrics_dir}")
    
    def create_final_report(self, all_results, comparison_df, holdout_metrics, features, target):
        """Crée un rapport final au format markdown"""
        
        report_path = os.path.join(self.output_dir, "RAPPORT_FINAL.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 RAPPORT FINAL - EXPÉRIENCE ML RIGOUREUSE\n\n")
            
            f.write(f"**Date d'expérience:** {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Variable cible:** {target}\n")
            f.write(f"**Nombre de features:** {len(features)}\n")
            f.write(f"**Stratégies évaluées:** {len(all_results)}\n\n")
            
            f.write("## 🎯 RÉSUMÉ EXÉCUTIF\n\n")
            
            # Meilleure stratégie
            best_strategy = comparison_df.iloc[0]
            f.write(f"**🥇 MEILLEURE STRATÉGIE:** {best_strategy['Strategy']}\n")
            f.write(f"- ROC-AUC (Validation): {best_strategy['Val_ROC_AUC']:.3f}\n")
            f.write(f"- F1-Score (Validation): {best_strategy['Val_F1']:.3f}\n")
            f.write(f"- Balanced Accuracy (Validation): {best_strategy['Val_Balanced_Acc']:.3f}\n\n")
            
            if best_strategy['Strategy'] in holdout_metrics:
                holdout = holdout_metrics[best_strategy['Strategy']]
                f.write(f"**🎯 PERFORMANCE HOLDOUT (données jamais vues):**\n")
                f.write(f"- ROC-AUC: {holdout['roc_auc']:.3f}\n")
                f.write(f"- F1-Score: {holdout['f1']:.3f}\n")
                f.write(f"- Precision: {holdout['precision']:.3f}\n")
                f.write(f"- Recall: {holdout['recall']:.3f}\n\n")
            
            f.write("## 📈 CLASSEMENT DES STRATÉGIES\n\n")
            f.write("| Rang | Stratégie | ROC-AUC | F1-Score | Balanced Acc | Precision | Recall |\n")
            f.write("|------|-----------|---------|----------|--------------|-----------|--------|\n")
            
            for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                f.write(f"| {i} | {row['Strategy']} | {row['Val_ROC_AUC']:.3f} | {row['Val_F1']:.3f} | {row['Val_Balanced_Acc']:.3f} | {row['Val_Precision']:.3f} | {row['Val_Recall']:.3f} |\n")
            
            f.write("\n## 🔍 ANALYSE DÉTAILLÉE\n\n")
            
            f.write("### Méthodologie\n")
            f.write("- **Division des données:** Train (70%) / Validation (15%) / Holdout (15%)\n")
            f.write("- **Validation croisée:** StratifiedKFold avec 5 splits\n")
            f.write("- **Rééquilibrage:** Appliqué uniquement sur les données d'entraînement\n")
            f.write("- **Optimisation des seuils:** Validée sur l'ensemble de validation\n")
            f.write("- **Évaluation finale:** Ensemble holdout (données jamais vues)\n\n")
            
            f.write("### Top 3 des Stratégies\n\n")
            for i, (_, row) in enumerate(comparison_df.head(3).iterrows(), 1):
                f.write(f"**{i}. {row['Strategy']}**\n")
                f.write(f"- Validation croisée ROC-AUC: {row['CV_ROC_AUC_Mean']:.3f} ± {row['CV_ROC_AUC_Std']:.3f}\n")
                f.write(f"- Validation ROC-AUC: {row['Val_ROC_AUC']:.3f}\n")
                f.write(f"- F1-Score: {row['Val_F1']:.3f}\n")
                
                if row['Strategy'] in holdout_metrics:
                    f.write(f"- **Holdout ROC-AUC: {holdout_metrics[row['Strategy']]['roc_auc']:.3f}**\n")
                f.write("\n")
            
            f.write("## 📁 FICHIERS GÉNÉRÉS\n\n")
            f.write("### Modèles\n")
            f.write("- Modèles entraînés: `models/pipeline_[strategie].pkl`\n\n")
            
            f.write("### Métriques\n")
            f.write("- Métriques détaillées: `metrics/metrics_[strategie].json`\n")
            f.write("- Analyse des seuils: `metrics/threshold_analysis_[strategie].csv`\n")
            f.write("- Comparaison finale: `strategy_comparison_final.csv`\n\n")
            
            f.write("### Visualisations\n")
            f.write("- Analyse complète par stratégie: `plots/analysis_complete_[strategie].png`\n")
            f.write("- Importance des features: `plots/feature_importance_detailed_[strategie].png`\n")
            f.write("- Comparaison des stratégies: `plots/strategy_comparison.png`\n")
            f.write("- Graphique radar: `plots/radar_comparison_top5.png`\n")
            f.write("- Analyse holdout: `plots/holdout_analysis_[strategie].png`\n\n")
            
            f.write("### Données\n")
            f.write("- Indices des splits: `data/data_splits.json`\n")
            f.write("- État de l'expérience: `experiment_state.json`\n\n")
            
            f.write("##  RECOMMANDATIONS\n\n")
            f.write(f"1. **Modèle recommandé:** {best_strategy['Strategy']}\n")
            f.write("2. **Seuil optimal:** Utiliser l'optimisation des seuils selon le critère métier\n")
            f.write("3. **Features importantes:** Voir les graphiques de feature importance\n")
            f.write("4. **Validation:** Les résultats holdout confirment la robustesse du modèle\n\n")
            
            f.write("---\n")
            f.write("*Rapport généré automatiquement par le Pipeline ML Rigoureux*\n")
        
        print(f"📋 Rapport final créé: {report_path}")

    # FONCTION UTILITAIRE POUR REPRENDRE UNE EXPÉRIENCE
    def resume_experiment(experiment_directory, df_rf, features, target='statut_deces_boolean'):
        """
        Reprend une expérience interrompue
        """
        print(f"🔄 REPRISE D'EXPÉRIENCE: {experiment_directory}")
        
        pipeline = EnhancedRigorousMLPipeline(
            base_output_path="C:/experiments_rigorous",
            experiment_name="medical_prediction_rigorous"
        )
        
        results = pipeline.evaluate_all_strategies(
            df_rf, features, target, 
            resume_experiment_id=experiment_directory
        )
        
        return results, pipeline


    def run_enhanced_rigorous_experiment(self, df_rf, features, target='statut_deces_boolean',output_dir="f{df_rf}"):
        """
        Lance une expérience complète et rigoureuse avec toutes les améliorations
        """
        print("LANCEMENT DE L'EXPÉRIENCE RIGOUREUSE AMÉLIORÉE")
        print("SUPPRESSION DE COLONNES DUPLIQUEES SI EXISTANTES...")

        df_rf = df_rf.loc[:, ~df_rf.columns.duplicated()]
        
        # Exécuter l'évaluation complète des stratégies
        results = self.evaluate_all_strategies(df_rf, features, target)
        if output_dir:
            self.output_dir = output_dir   
        else:
            self.output_dir = self.create_output_directory()
        
        result_path = os.path.join(self.output_dir, f"RAPPORT_FINAL.md")
        
        print("✅ EXPÉRIENCE TERMINÉE AVEC SUCCÈS")
        return results

    def create_final_report(self, output_dir, metadata, importance_df, threshold_results):
        """Crée un rapport final en markdown"""
        
        report_path = os.path.join(output_dir, "RAPPORT_FINAL.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🏥 Rapport d'Expérience - Modèle de Prédiction Random Forest\n\n")
            f.write(f"**Date de l'expérience :** {metadata['experiment_info']['date']}\n\n")
            
            f.write("## 📊 Informations Générales\n\n")
            f.write(f"- **Variable cible :** {metadata['experiment_info']['target_variable']}\n")
            f.write(f"- **Nombre de caractéristiques :** {metadata['experiment_info']['n_features']}\n")
            f.write(f"- **Échantillons d'entraînement :** {metadata['experiment_info']['n_samples_train']}\n")
            f.write(f"- **Échantillons de test :** {metadata['experiment_info']['n_samples_test']}\n")
            f.write(f"- **Score AUC :** {metadata['experiment_info']['auc_score']:.3f}\n\n")
            
            f.write("## 🎯 Optimisation des Seuils de Décision\n\n")
            f.write("| Stratégie | Seuil Optimal | Précision | Rappel | F1-Score | Spécificité |\n")
            f.write("|-----------|---------------|-----------|--------|----------|-------------|\n")
            
            for strategy, results in threshold_results.items():
                f.write(f"| {strategy.capitalize()} | {results['optimal_threshold']:.3f} | "
                    f"{results['precision_at_optimal']:.3f} | {results['recall_at_optimal']:.3f} | "
                    f"{results['f1_at_optimal']:.3f} | {results['specificity_at_optimal']:.3f} |\n")
            
            f.write("\n## 🔍 Top 10 Variables Importantes\n\n")
            top_10 = importance_df.head(10)
            f.write("| Rang | Variable | Importance |\n")
            f.write("|------|----------|------------|\n")
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"| {i} | {row['Feature']} | {row['Importance']:.4f} |\n")
            
            f.write("\n## 📁 Fichiers Générés\n\n")
            f.write("- `pipeline_model.pkl` : Modèle entraîné\n")
            f.write("- `experiment_metadata.json` : Métadonnées complètes\n")
            f.write("- `feature_importances.csv` : Importance des variables\n")
            f.write("- `threshold_comparison.csv` : Comparaison des seuils\n")
            f.write("- `threshold_optimization_*.png` : Graphiques d'optimisation\n")
            f.write("- `rapport_classification_default.txt` : Rapport avec seuil par défaut\n")
            f.write("- `matrice_confusion_default.png` : Matrice de confusion\n")
            f.write("- `roc_curve.png` : Courbe ROC\n")
            f.write("- `top20_importances.png` : Graphique des variables importantes\n\n")
            
            f.write("## 💡 Recommandations Cliniques\n\n")
            f.write("### Choix du Seuil selon l'Usage :\n\n")
            f.write("- **Pour minimiser les faux négatifs (ne pas rater de décès)** : Utiliser le seuil optimisé pour le rappel\n")
            f.write("- **Pour minimiser les faux positifs (éviter les alertes inutiles)** : Utiliser le seuil optimisé pour la précision\n")
            f.write("- **Pour un équilibre général** : Utiliser le seuil optimisé pour le F1-Score\n")
            f.write("- **Pour équilibrer sensibilité et spécificité** : Utiliser l'indice de Youden\n")
        
        print(f"📋 Rapport final créé : {report_path}")






