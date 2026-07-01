
import pandas as pd
from adjustText import adjust_text
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

class PCAProcessor:
    def __init__(self, df, variables, var_expliquee_min=0.99):
        """
        Initialise l'ACP Analyzer.
        Args:
            df (pd.DataFrame): Les données.
            variables (list): Liste des variables à inclure dans l'ACP.
            var_expliquee_min (float): Seuil de variance expliquée cumulée pour retenir les composantes.
        """
        self.df = df
        self.df = self.df.reset_index(drop=True)
        self.variables = variables
        self.var_expliquee_min = var_expliquee_min
        self.scaler = StandardScaler()
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.n_components = None
        self.df_pca = None
        self.df_clean_original_index = None # Pour garder la trace des index originaux des lignes non NaN

    def preparer_donnees(self):
        """
        Prépare les données en sélectionnant les variables, en supprimant les lignes avec des NaN,
        et en standardisant les données.
        """
        print("Préparation et normalisation des données... 🧹")
        
        # S'assurer que seules les variables spécifiées sont utilisées et gérer les NaN
        X = self.df[self.variables]
        
        # Garder une trace des index originaux avant de supprimer les NaN
        self.df_clean_original_index = X.dropna().index
        X_clean = X.dropna()

        if X_clean.empty:
            raise ValueError("Aucune donnée disponible après suppression des NaN dans les variables spécifiées. Veuillez vérifier vos données ou la liste des variables.")
        
        self.X_scaled = self.scaler.fit_transform(X_clean)
        self.df_clean = X_clean.reset_index(drop=True) 
        print(f"Données préparées. {len(self.df_clean)} lignes utilisées après suppression des NaN. ✅")
        return self.df_clean

    def verifier_kmo_bartlett(self):
        """
        Effectue le test de Kaiser-Meyer-Olkin (KMO) et le test de sphéricité de Bartlett
        pour évaluer l'adéquation des données à l'ACP.
        """
        if self.X_scaled is None:
            raise ValueError("Les données n'ont pas été préparées. Exécutez 'preparer_donnees()' d'abord.")

        print("Calcul du KMO et du test de Bartlett... 🧪")
        kmo_all, kmo_model = calculate_kmo(self.X_scaled)
        print(f"KMO Global = {kmo_model:.3f} (doit être > 0.5 pour une ACP appropriée)")
        
        # Test de Bartlett ne nécessite pas le même X_scaled que calculate_kmo
        # Il opère sur la matrice de corrélation qui peut être dérivée de X_scaled
        chi_square_value, p_value = calculate_bartlett_sphericity(self.X_scaled)
        print(f"Bartlett p-value = {p_value:.3e} (doit être < 0.05 pour rejeter l'hypothèse nulle et indiquer une adéquation)")
        
        if kmo_model < 0.5 or p_value >= 0.05:
            print("⚠️ Attention : Les tests KMO/Bartlett suggèrent que l'ACP pourrait ne pas être la méthode la plus appropriée pour ces données.")
        else:
            print("Tests KMO/Bartlett réussis. Les données sont adéquates pour l'ACP. 👍")


    def calculer_acp(self):
        """
        Calcule les composantes principales, détermine le nombre optimal de composantes
        basé sur la variance expliquée minimale, et construit le DataFrame des composantes.
        """
        
        if self.X_scaled is None:
            raise ValueError("Les données n'ont pas été standardisées. Exécutez 'preparer_donnees()' d'abord.")


        print("Calcul des composantes principales... ⚙️")
        self.pca = PCA()
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        explained_var = self.pca.explained_variance_ratio_
        cum_var = explained_var.cumsum()
        
        # Détermine le nombre de composantes nécessaires pour atteindre la variance expliquée minimale
        self.n_components = np.argmax(cum_var >= self.var_expliquee_min) + 1
        
        print(f"Nombre de composantes retenues pour {self.var_expliquee_min*100:.1f}% de variance expliquée : {self.n_components} 📈")
        
        # Créer le DataFrame des composantes principales uniquement
        df_pca_only = pd.DataFrame(self.X_pca[:, :self.n_components], 
                                   columns=[f'PC{i+1}' for i in range(self.n_components)])
        
        # Concaténer le DataFrame original (nettoyé) avec les composantes principales
        # Utiliser l'index original des lignes nettoyées pour l'alignement
        self.df_pca = pd.concat([self.df.loc[self.df_clean_original_index].reset_index(drop=True), 
                                 df_pca_only], axis=1)
        
        print("DataFrame PCA généré avec succès. ✨")


    def plot_variance_expliquee(self):
        """
        Affiche le graphique de la variance expliquée cumulée par le nombre de composantes principales.
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord. ❌")
            return

        print("Affichage de la variance expliquée cumulée... 📊")
        explained_var = self.pca.explained_variance_ratio_
        cum_var = explained_var.cumsum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-', color='indigo', linewidth=2)
        plt.axhline(self.var_expliquee_min, color='red', linestyle='--', label=f'Seuil {self.var_expliquee_min*100:.1f}%')
        plt.axvline(self.n_components, color='green', linestyle=':', label=f'{self.n_components} Composantes')
        
        plt.title("Variance Expliquée Cumulée par Composante Principale", fontsize=15)
        plt.xlabel("Nombre de Composantes Principales", fontsize=2, rotation=45)
        plt.ylabel("Variance Expliquée Cumulée", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(range(1, len(cum_var) + 1),  rotation=45)
        plt.tight_layout()
        plt.show()

    def get_top_variables_for_plot(self, n_top=20, method='loadings_abs_sum'):
        """
        Sélectionne les n_top variables les plus importantes pour la visualisation sur le cercle de corrélation
        basées sur différentes méthodes.
        Args:
            n_top (int): Nombre de variables à retourner.
            method (str): Méthode de sélection ('loadings_abs_sum', 'cos2_sum', 'contributions').
                          'loadings_abs_sum': Somme des valeurs absolues des loadings sur PC1 et PC2.
                          'cos2_sum': Somme des cos2 (carré des loadings) sur PC1 et PC2.
                          'contributions': Somme des contributions (cos2 / variance expliquée de la composante).
        Returns:
            list: Liste des noms des variables sélectionnées.
        """
        if self.pca is None or self.X_scaled is None:
            print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord. ❌")
            return []

        # Pour le cercle de corrélation, nous nous intéressons aux loadings sur PC1 et PC2
        # S'assurer que nous avons au moins 2 composantes pour le cercle de corrélation
        if self.pca.components_.shape[0] < 2:
            print("Moins de 2 composantes principales disponibles pour le cercle de corrélation. ⚠️")
            return []

        loadings_df = pd.DataFrame(self.pca.components_[:2].T, # Seulement PC1 et PC2 pour le cercle
                                   index=self.variables,
                                   columns=['PC1', 'PC2'])

        if method == 'loadings_abs_sum':
            scores = loadings_df.abs().sum(axis=1)
        elif method == 'cos2_sum':
            cos2_df = loadings_df**2
            scores = cos2_df.sum(axis=1)
        elif method == 'contributions':
            # Contributions = (cos2 * 100) / variance expliquée de la composante
            # Cette formule est généralement appliquée par variable sur une composante
            # Pour la somme sur PC1 et PC2, on peut sommer les contributions individuelles.
            # Attention: self.pca.explained_variance_ratio_ est un array
            explained_var_pc1 = self.pca.explained_variance_ratio_[0]
            explained_var_pc2 = self.pca.explained_variance_ratio_[1]
            
            contributions_pc1 = (loadings_df['PC1']**2 * 100) / explained_var_pc1
            contributions_pc2 = (loadings_df['PC2']**2 * 100) / explained_var_pc2
            scores = contributions_pc1 + contributions_pc2
        else:
            raise ValueError("Méthode de sélection invalide. Choisissez parmi 'loadings_abs_sum', 'cos2_sum', 'contributions'.")

        return scores.nlargest(n_top).index.tolist()

    def plot_cercle_correlation(self, n_top_vars=20):
        """
        Affiche le cercle des corrélations des variables avec les deux premières composantes principales.
        Args:
            n_top_vars (int, optional): Nombre de variables les plus importantes à afficher. 
                                        Si None, toutes les variables sont affichées (peut être encombrant).
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord. ❌")
            return
        if self.pca.components_.shape[0] < 2:
            print("Moins de 2 composantes principales disponibles pour le cercle de corrélation. ⚠️")
            return

        print("Affichage du cercle des corrélations... 🌐")
        
        # Utiliser la méthode pour sélectionner les variables à afficher
        vars_to_plot = self.variables
        if n_top_vars is not None and len(self.variables) > n_top_vars:
            vars_to_plot = self.get_top_variables_for_plot(n_top=n_top_vars)
            print(f"Affichage des {len(vars_to_plot)} variables les plus contributives. (Méthode par défaut: loadings_abs_sum) ")

        components = self.pca.components_
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Cercle unité
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(circle)
        
        texts = []
        for var_name in vars_to_plot:
            idx = self.variables.index(var_name)
            x = components[0, idx] # Coordonnée sur PC1
            y = components[1, idx] # Coordonnée sur PC2
            
            # Flèche du centre vers la variable
            ax.arrow(0, 0, x, y, color='black', alpha=0.7, head_width=0.03, length_includes_head=True)
            
            # Texte de la variable
            texts.append(ax.text(x * 1.05, y * 1.05, var_name, color='darkgreen', ha='center', va='center', fontsize=9))
        
        # Ajuster le texte pour éviter les chevauchements
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='lightgrey', lw=0.5))
        
        # Lignes des axes
        ax.axhline(0, color='grey', lw=0.8, linestyle='--')
        ax.axvline(0, color='grey', lw=0.8, linestyle='--')
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(f'Composante principale 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'Composante principale 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Cercle des corrélations (Variables et Composantes)', fontsize=15)
        ax.set_aspect('equal') # Assure que le cercle est bien rond
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_heatmap_contributions(self, n_top_pcs=5):
        """
        Affiche une heatmap des contributions des variables aux premières composantes principales.
        Args:
            n_top_pcs (int): Nombre de premières composantes principales à inclure dans la heatmap.
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord. ❌")
            return

        print("Affichage de la heatmap des contributions... 🌡️")

        # Vérifier si suffisamment de composantes sont disponibles
        num_pcs_to_plot = min(n_top_pcs, self.pca.components_.shape[0])
        if num_pcs_to_plot == 0:
            print("Aucune composante principale à afficher dans la heatmap. ⚠️")
            return

        # Calculer les contributions (carré des loadings)
        # contributions = (self.pca.components_**2) # Cette ligne est déjà correcte pour les loadings carrés
        
        # Pour des contributions réelles (pourcentage de variance expliquée par une variable sur une composante)
        # contribution_ij = (cor(Xi, Pj)^2 * 100) / (var_expliquée_par_Pj)
        # C'est souvent plus complexe, ici on utilise le carré des loadings comme proxy simple de "contribution" ou "importance"
        loadings_squared = self.pca.components_[:num_pcs_to_plot]**2
        loadings_df = pd.DataFrame(loadings_squared.T, 
                                   index=self.variables, 
                                   columns=[f'PC{i+1}' for i in range(num_pcs_to_plot)])
        
        # Optionnel: Trier les variables par la somme de leurs contributions (ou importance) pour une meilleure lisibilité
        # Trier par la somme des loadings carrés sur toutes les PC affichées
        loadings_df['Total_Contribution'] = loadings_df.sum(axis=1)
        loadings_df = loadings_df.sort_values(by='Total_Contribution', ascending=False).drop(columns='Total_Contribution')


        plt.figure(figsize=(min(num_pcs_to_plot * 2 + 4, 16), min(len(self.variables) * 0.3 + 2, 20))) # Taille auto-ajustée
        sns.heatmap(loadings_df, annot=True, cmap='viridis', fmt=".2f", # Viridis est une bonne palette par défaut
                            linewidths=.5, cbar_kws={'label': 'Loadings au Carré (Proxy de Contribution)'})
        
        plt.title(f'Importance des Variables sur les {num_pcs_to_plot} Premières Composantes Principales', fontsize=15)
        plt.xlabel('Composantes Principales', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.show()

    
    def plot_cercle_correlation_interactif(self):
        """
        Affiche le cercle des corrélations en version interactive avec Plotly.
        """
        if self.pca is None or self.pca.components_.shape[0] < 2:
            print("L'ACP n'a pas été calculée ou moins de 2 composantes principales sont disponibles. Exécutez 'calculer_acp()' d'abord. ❌")
            return
            
        print("Affichage interactif du cercle des corrélations... ⚡")

        components = self.pca.components_
        fig = go.Figure()

        # Cercle unité
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Cercle Unité'
        ))

        # Vecteurs des variables
        for i, var in enumerate(self.variables):
            x = components[0, i]
            y = components[1, i]
            fig.add_trace(go.Scatter(
                x=[0, x],
                y=[0, y],
                mode='lines+markers+text',
                text=[None, var],
                textposition='top center',
                name=var,
                line=dict(color='red', width=2),
                marker=dict(size=6, color='red')
            ))

        fig.update_layout(
            title='Cercle des corrélations (ACP) - Interactif',
            xaxis=dict(title=f'Composante Principale 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', 
                       range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
            yaxis=dict(title=f'Composante Principale 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', 
                       range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
            width=750, # Légèrement plus grand
            height=750,
            showlegend=False,
            hovermode='closest' # Pour faciliter l'interaction au survol
        )
        fig.show()

    def plot_cercle_correlation_panels_with_groups(self, n_panels=4, vars_per_panel=None):
        """
        Affiche le cercle des corrélations en plusieurs panneaux avec identification automatique
        des groupes de variables selon leur position sur le cercle.
        Utile pour les rapports avec beaucoup de variables.
        """
        
        if self.pca is None or self.pca.components_.shape[0] < 2:
            print("L'ACP n'a pas été calculée ou moins de 2 composantes principales sont disponibles.")
            return

        if vars_per_panel is None:
            vars_per_panel = max(8, len(self.variables) // n_panels)
        
        # Obtenir les variables triées par importance
        top_vars = self.get_top_variables_for_plot(n_top=len(self.variables), method='cos2_sum')
        
        # Diviser en panneaux
        panels = [top_vars[i:i + vars_per_panel] for i in range(0, len(top_vars), vars_per_panel)]
        
        fig, axes = plt.subplots(2, 2, figsize=(25, 16))
        axes = axes.flatten()
        
        # Définir les couleurs pour les groupes
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for idx, panel_vars in enumerate(panels[:n_panels]):
            ax = axes[idx]
            
            # Cercle unité
            circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
            ax.add_artist(circle)
            
            # Calculer les coordonnées des variables pour ce panneau
            var_coords = {var_name: (self.pca.components_[0, self.variables.index(var_name)],
                                      self.pca.components_[1, self.variables.index(var_name)])
                          for var_name in panel_vars}
            
            # Identifier les groupes de variables selon leur position
            groups = self._identify_variable_groups(var_coords)
            
            # Dessiner les variables par groupe
            for group_idx, (group_name, group_vars) in enumerate(groups.items()):
                color = colors[group_idx % len(colors)]
                
                # Trace le nuage de points pour ce groupe
                x_coords = [var_coords[var][0] for var in group_vars]
                y_coords = [var_coords[var][1] for var in group_vars]
                ax.scatter(x_coords, y_coords, color=color, label=group_name)
                
                # Dessiner les flèches (uniquement pour les points)
                for x, y in zip(x_coords, y_coords):
                    ax.arrow(0, 0, x, y, color=color, alpha=0.7, head_width=0.03, 
                             length_includes_head=True, linewidth=2)
            
            # Ajouter une légende des groupes sur le côté droit du graphique
            legend_text = self._create_groups_legend(groups)

            # Ajuster la position de la légende en fonction de la colonne
            # Les panneaux 0 et 2 sont dans la première colonne
            if idx in [0, 2]:
                x_pos = 1.15  # Rapprocher la légende de l'image
            # Les panneaux 1 et 3 sont dans la deuxième colonne
            else:
                x_pos = 1.05

            ax.text(x_pos, 0.8, legend_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor="lightgray", alpha=0.8))
            
            # Configuration de l'axe
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.axhline(0, color='grey', lw=0.8, linestyle='--')
            ax.axvline(0, color='grey', lw=0.8, linestyle='--')
            ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title(f'pane {idx+1} - Variables {idx*vars_per_panel+1} to {min((idx+1)*vars_per_panel, len(top_vars))}')
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
        
        # Masquer les axes non utilisés
        for idx in range(len(panels), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('cercle_correlation_panels_with_groups.png', dpi=400, bbox_inches='tight')
        plt.show()
        
        # Afficher un résumé textuel des groupes pour chaque panneau
        self._print_groups_summary(panels, top_vars, vars_per_panel)
        
    def _identify_variable_groups(self, var_coords, angle_threshold=45):
        """
        Identifie les groupes de variables selon leur position angulaire sur le cercle de corrélation.
        
        Args:
            var_coords (dict): Dictionnaire {nom_variable: (x, y)} des coordonnées des variables
            angle_threshold (float): Seuil d'angle en degrés pour grouper les variables
        
        Returns:
            dict: Dictionnaire des groupes {nom_groupe: [liste_variables]}
        """
        import math
        
        # Calculer l'angle de chaque variable par rapport à l'axe PC1
        var_angles = {}
        for var_name, (x, y) in var_coords.items():
            angle = math.degrees(math.atan2(y, x))
            if angle < 0:
                angle += 360  # Normaliser entre 0 et 360 degrés
            var_angles[var_name] = angle
        
        # Définir les secteurs principaux
        sectors = {
            'PC1+ (Droite)': (315, 45),  # -45° à +45°
            'PC2+ (Haut)': (45, 135),   # 45° à 135°
            'PC1- (Gauche)': (135, 225), # 135° à 225°
            'PC2- (Bas)': (225, 315)    # 225° à 315°
        }
        
        groups = {sector_name: [] for sector_name in sectors.keys()}
        
        for var_name, angle in var_angles.items():
            for sector_name, (min_angle, max_angle) in sectors.items():
                if sector_name == 'PC1+ (Droite)':
                    # Cas spécial pour le secteur qui traverse 0°
                    if angle >= min_angle or angle <= max_angle:
                        groups[sector_name].append(var_name)
                        break
                else:
                    if min_angle <= angle <= max_angle:
                        groups[sector_name].append(var_name)
                        break
        
        # Supprimer les groupes vides
        groups = {k: v for k, v in groups.items() if v}
        
        return groups

    def _create_groups_legend(self, groups):
            """
            Crée le texte de légende pour les groupes de variables.
            
            Args:
                groups (dict): Dictionnaire des groupes {nom_groupe: [liste_variables]}
            
            Returns:
                str: Texte formaté pour la légende
            """
            legend_lines = ["GROUPES DE VARIABLES:"]
            legend_lines.append("-" * 25)
            
            for group_name, variables in groups.items():
                if variables:  # Seulement afficher les groupes non vides
                    legend_lines.append(f"\n{group_name}:")
                    # Afficher chaque variable sur une nouvelle ligne
                    for var in variables:
                        legend_lines.append(f"  • {var}")
            
            return '\n'.join(legend_lines)

    def _print_groups_summary(self, panels, top_vars, vars_per_panel):
        """
        Affiche un résumé textuel détaillé des groupes pour chaque panneau.
        
        Args:
            panels (list): Liste des panneaux de variables
            top_vars (list): Liste de toutes les variables triées par importance
            vars_per_panel (int): Nombre de variables par panneau
        """
        print("\n" + "="*80)
        print("RÉSUMÉ DES GROUPES DE VARIABLES PAR PANNEAU")
        print("="*80)
        
        for idx, panel_vars in enumerate(panels[:4]):
            print(f"\n📊 PANNEAU {idx+1} - Variables {idx*vars_per_panel+1} à {min((idx+1)*vars_per_panel, len(top_vars))}")
            print("-" * 60)
            
            # Calculer les coordonnées pour ce panneau
            var_coords = {}
            for var_name in panel_vars:
                var_idx = self.variables.index(var_name)
                x = self.pca.components_[0, var_idx]
                y = self.pca.components_[1, var_idx]
                var_coords[var_name] = (x, y)
            
            # Identifier les groupes
            groups = self._identify_variable_groups(var_coords)
            
            for group_name, variables in groups.items():
                if variables:
                    print(f"\n🔍 {group_name}:")
                    print(f"   Nombre de variables: {len(variables)}")
                    print(f"   Variables: {', '.join(variables)}")
                    
                    # Calculer les statistiques du groupe
                    loadings_pc1 = [var_coords[var][0] for var in variables]
                    loadings_pc2 = [var_coords[var][1] for var in variables]
                    
                    print(f"   Moyenne PC1: {np.mean(loadings_pc1):.3f}")
                    print(f"   Moyenne PC2: {np.mean(loadings_pc2):.3f}")
                    print(f"   Distance moyenne à l'origine: {np.mean([np.sqrt(x**2 + y**2) for x, y in [var_coords[var] for var in variables]]):.3f}")
        
        print("\n" + "="*80)
        print("INTERPRÉTATION DES GROUPES:")
        print("="*80)
        print("• PC1+ (Droite): Variables positivement corrélées avec la première composante")
        print("• PC1- (Gauche): Variables négativement corrélées avec la première composante")
        print("• PC2+ (Haut): Variables positivement corrélées avec la deuxième composante")
        print("• PC2- (Bas): Variables négativement corrélées avec la deuxième composante")
        print("\nLes variables proches du centre (distance faible) sont moins bien représentées")
        print("par les deux premières composantes principales.")

    def export_groups_analysis(self, output_file='groups_analysis.csv'):
        """
        Exporte une analyse complète des groupes de variables dans un fichier CSV.
        
        Args:
            output_file (str): Nom du fichier de sortie
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée.")
            return None
        
        # Calculer les coordonnées pour toutes les variables
        var_coords = {}
        for i, var_name in enumerate(self.variables):
            x = self.pca.components_[0, i]
            y = self.pca.components_[1, i]
            var_coords[var_name] = (x, y)
        
        # Identifier les groupes pour toutes les variables
        groups = self._identify_variable_groups(var_coords)
        
        # Créer un DataFrame avec l'analyse
        analysis_data = []
        for var_name, (x, y) in var_coords.items():
            # Trouver le groupe de cette variable
            group_name = "Non classé"
            for gname, gvars in groups.items():
                if var_name in gvars:
                    group_name = gname
                    break
            
            # Calculer les métriques
            distance_origin = np.sqrt(x**2 + y**2)
            angle = np.degrees(np.arctan2(y, x))
            if angle < 0:
                angle += 360
            
            cos2_pc1 = x**2
            cos2_pc2 = y**2
            cos2_total = cos2_pc1 + cos2_pc2
            
            analysis_data.append({
                'Variable': var_name,
                'Groupe': group_name,
                'PC1_loading': x,
                'PC2_loading': y,
                'Distance_origine': distance_origin,
                'Angle_degres': angle,
                'Cos2_PC1': cos2_pc1,
                'Cos2_PC2': cos2_pc2,
                'Cos2_total': cos2_total,
                'Qualite_representation': 'Bonne' if cos2_total > 0.5 else 'Moyenne' if cos2_total > 0.3 else 'Faible'
            })
        
        df_analysis = pd.DataFrame(analysis_data)
        df_analysis = df_analysis.sort_values(['Groupe', 'Cos2_total'], ascending=[True, False])
        
        # Sauvegarder
        df_analysis.to_csv(output_file, index=False)
        print(f"Analyse des groupes sauvegardée dans '{output_file}'")
        
        return df_analysis
    
    def export_pca_summary_table(self, n_top=20):
        """
        Exporte un tableau récapitulatif des variables les plus importantes pour le rapport.
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée.")
            return None
        
        # Calculer les métriques pour toutes les variables
        loadings_df = pd.DataFrame(self.pca.components_[:2].T, 
                                index=self.variables,
                                columns=['PC1_loading', 'PC2_loading'])
        
        # Calculer cos2, contributions, etc.
        loadings_df['cos2_PC1'] = loadings_df['PC1_loading']**2
        loadings_df['cos2_PC2'] = loadings_df['PC2_loading']**2
        loadings_df['cos2_total'] = loadings_df['cos2_PC1'] + loadings_df['cos2_PC2']
        
        loadings_df['contrib_PC1'] = (loadings_df['cos2_PC1'] * 100) / self.pca.explained_variance_ratio_[0]
        loadings_df['contrib_PC2'] = (loadings_df['cos2_PC2'] * 100) / self.pca.explained_variance_ratio_[1]
        loadings_df['contrib_total'] = loadings_df['contrib_PC1'] + loadings_df['contrib_PC2']
        
        # Trier par contribution totale
        summary_df = loadings_df.sort_values('contrib_total', ascending=False).head(n_top)
        
        # Arrondir pour la présentation
        summary_df = summary_df.round(3)
        
        # Sauvegarder
        summary_df.to_csv('pca_variables_summary.csv')
        print(f"Tableau récapitulatif sauvegardé dans 'pca_variables_summary.csv'")
        
        return summary_df

    def generate_report_assets_with_groups(self):
        """
        Génère tous les éléments nécessaires pour le rapport avec l'analyse des groupes.
        """
        print("Génération des éléments pour le rapport avec analyse des groupes...")
        
        # 1. Cercle standard avec moins de variables
        self.plot_cercle_correlation(n_top_vars=12)
        
        # 2. Panneaux détaillés avec groupes identifiés
        self.plot_cercle_correlation_panels_with_groups()
        
        # 3. Tableau récapitulatif
        summary_df = self.export_pca_summary_table()
        
        # 4. Analyse des groupes
        groups_df = self.export_groups_analysis()
        
        # 5. Heatmap des contributions
        self.plot_heatmap_contributions()
        
        print("Tous les éléments ont été générés et sauvegardés.")
        return summary_df, groups_df
    


    def plot_cercle_correlation_unified(self, pc1_color='red', pc2_color='green', mixed_color='orange'):
            """
            Affiche un seul cercle des corrélations avec toutes les variables sous forme de flèches colorées.
            - Rouge : variables principalement corrélées avec PC1
            - Vert : variables principalement corrélées avec PC2
            - Orange : variables avec corrélations mixtes (PC1 et PC2 similaires)
            
            Args:
                pc1_color (str): Couleur pour les variables principalement corrélées avec PC1
                pc2_color (str): Couleur pour les variables principalement corrélées avec PC2
                mixed_color (str): Couleur pour les variables avec corrélations mixtes
            """
            if self.pca is None:
                print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord.")
                return
            if self.pca.components_.shape[0] < 2:
                print("Moins de 2 composantes principales disponibles pour le cercle de corrélation.")
                return

            print("Affichage du cercle des corrélations unifié...")
            
            components = self.pca.components_
            
            fig, ax = plt.subplots(figsize=(14, 14))
            
            # Cercle unité
            circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=2)
            ax.add_artist(circle)
            
            # Listes pour les groupes de variables
            pc1_vars = []
            pc2_vars = []
            mixed_vars = []
            
            # Analyser chaque variable et l'assigner à un groupe
            for i, var_name in enumerate(self.variables):
                x = components[0, i]  # Coordonnée sur PC1
                y = components[1, i]  # Coordonnée sur PC2
                
                # Calculer les corrélations au carré (cos2)
                cos2_pc1 = x**2
                cos2_pc2 = y**2
                
                # Déterminer la couleur basée sur la corrélation dominante
                # Seuil pour considérer une corrélation comme "mixte"
                ratio_threshold = 0.7  # Si cos2_max / cos2_min < 0.7, c'est mixte
                
                if cos2_pc1 > cos2_pc2:
                    if cos2_pc2 / cos2_pc1 > ratio_threshold:
                        color = mixed_color
                        mixed_vars.append(var_name)
                    else:
                        color = pc1_color
                        pc1_vars.append(var_name)
                else:
                    if cos2_pc1 / cos2_pc2 > ratio_threshold:
                        color = mixed_color
                        mixed_vars.append(var_name)
                    else:
                        color = pc2_color
                        pc2_vars.append(var_name)
                
                # Dessiner la flèche
                ax.arrow(0, 0, x, y, color=color, alpha=0.7, head_width=0.02, 
                        length_includes_head=True, linewidth=2, zorder=3)
            
            # Lignes des axes
            ax.axhline(0, color='grey', lw=1, linestyle='--', alpha=0.8)
            ax.axvline(0, color='grey', lw=1, linestyle='--', alpha=0.8)
            
            # Configuration des axes
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlabel(f'Principal Component 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
            ax.set_ylabel(f'Principal Component 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
            ax.set_title('Correlations circle - Classification on PC1/PC2', fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.4)
            
            # Légende personnalisée
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=pc1_color, label=f'Correlated to PC1 ({len(pc1_vars)} vars)'),
                Patch(facecolor=pc2_color, label=f'Correlated to PC2 ({len(pc2_vars)} vars)'),
                Patch(facecolor=mixed_color, label=f'mixed correlation ({len(mixed_vars)} vars)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.tight_layout()
            plt.savefig('cercle_correlation_unified.png', dpi=400, bbox_inches='tight')
            plt.show()
            
            # Afficher un résumé des groupes
            self._print_unified_groups_summary(pc1_vars, pc2_vars, mixed_vars)

    def _print_unified_groups_summary(self, pc1_vars, pc2_vars, mixed_vars):
        """
        Affiche un résumé des groupes de variables pour le cercle unifié.
        
        Args:
            pc1_vars (list): Variables principalement corrélées avec PC1
            pc2_vars (list): Variables principalement corrélées avec PC2  
            mixed_vars (list): Variables avec corrélations mixtes
        """
        print("\n" + "="*80)
        print("RÉSUMÉ DES GROUPES - CERCLE DE CORRÉLATION UNIFIÉ")
        print("="*80)
        
        # Groupe PC1
        if pc1_vars:
            print(f"\n🔴 VARIABLES PRINCIPALEMENT CORRÉLÉES AVEC PC1 ({len(pc1_vars)} variables):")
            print("-" * 60)
            for i, var in enumerate(pc1_vars, 1):
                var_idx = self.variables.index(var)
                pc1_loading = self.pca.components_[0, var_idx]
                pc2_loading = self.pca.components_[1, var_idx]
                cos2_pc1 = pc1_loading**2
                cos2_pc2 = pc2_loading**2
                print(f"  {i:2d}. {var:<20} | PC1: {pc1_loading:6.3f} | PC2: {pc2_loading:6.3f} | Cos²PC1: {cos2_pc1:.3f}")
        
        # Groupe PC2
        if pc2_vars:
            print(f"\n🟢 VARIABLES PRINCIPALEMENT CORRÉLÉES AVEC PC2 ({len(pc2_vars)} variables):")
            print("-" * 60)
            for i, var in enumerate(pc2_vars, 1):
                var_idx = self.variables.index(var)
                pc1_loading = self.pca.components_[0, var_idx]
                pc2_loading = self.pca.components_[1, var_idx]
                cos2_pc1 = pc1_loading**2
                cos2_pc2 = pc2_loading**2
                print(f"  {i:2d}. {var:<20} | PC1: {pc1_loading:6.3f} | PC2: {pc2_loading:6.3f} | Cos²PC2: {cos2_pc2:.3f}")
        
        # Groupe mixte
        if mixed_vars:
            print(f"\n🟠 VARIABLES AVEC CORRÉLATIONS MIXTES ({len(mixed_vars)} variables):")
            print("-" * 60)
            for i, var in enumerate(mixed_vars, 1):
                var_idx = self.variables.index(var)
                pc1_loading = self.pca.components_[0, var_idx]
                pc2_loading = self.pca.components_[1, var_idx]
                cos2_pc1 = pc1_loading**2
                cos2_pc2 = pc2_loading**2
                ratio = min(cos2_pc1, cos2_pc2) / max(cos2_pc1, cos2_pc2)
                print(f"  {i:2d}. {var:<20} | PC1: {pc1_loading:6.3f} | PC2: {pc2_loading:6.3f} | Ratio: {ratio:.3f}")
        
        print("\n" + "="*80)
        print("INTERPRÉTATION:")
        print("="*80)
        print("• Variables ROUGES : Contribuent principalement à la première dimension (PC1)")
        print("• Variables VERTES : Contribuent principalement à la deuxième dimension (PC2)")
        print("• Variables ORANGE : Contribuent de manière équilibrée aux deux dimensions")
        print("\nLa longueur des flèches indique la qualité de représentation de chaque variable")
        print("dans l'espace PC1-PC2 (plus la flèche est longue, mieux la variable est représentée).")

    def plot_cercle_correlation_sans_labels(self, pc1_color='red', pc2_color='green', point_size=60, alpha=0.7):
        """
        Affiche un cercle des corrélations avec toutes les variables représentées par des points colorés
        sans étiquettes de noms. Les variables sont colorées selon leur axe de corrélation principal :
        - Rouge : variables principalement corrélées avec PC1
        - Vert : variables principalement corrélées avec PC2
        
        Args:
            pc1_color (str): Couleur pour les variables principalement corrélées avec PC1
            pc2_color (str): Couleur pour les variables principalement corrélées avec PC2
            point_size (int): Taille des points représentant les variables
            alpha (float): Transparence des points (0-1)
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée. Exécutez 'calculer_acp()' d'abord. ❌")
            return
        if self.pca.components_.shape[0] < 2:
            print("Moins de 2 composantes principales disponibles pour le cercle de corrélation. ⚠️")
            return

        print("Affichage du cercle des corrélations sans labels...")
        
        components = self.pca.components_
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Cercle unité
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=2)
        ax.add_artist(circle)
        
        # Listes pour stocker les coordonnées par groupe
        pc1_x, pc1_y = [], []
        pc2_x, pc2_y = [], []
        
        # Analyser chaque variable et l'assigner à un groupe
        for i, var_name in enumerate(self.variables):
            x = components[0, i]  # Coordonnée sur PC1
            y = components[1, i]  # Coordonnée sur PC2
            
            # Calculer les corrélations au carré (cos2)
            cos2_pc1 = x**2
            cos2_pc2 = y**2
            
            # Déterminer le groupe basé sur la corrélation dominante
            if cos2_pc1 > cos2_pc2:
                pc1_x.append(x)
                pc1_y.append(y)
            else:
                pc2_x.append(x)
                pc2_y.append(y)
        
        # Dessiner les points par groupe
        if pc1_x:  # Variables corrélées à PC1
            ax.scatter(pc1_x, pc1_y, color=pc1_color, s=point_size, alpha=alpha, 
                    label=f'Correlated to PC1 ({len(pc1_x)} variables)', zorder=3)
        
        if pc2_x:  # Variables corrélées à PC2
            ax.scatter(pc2_x, pc2_y, color=pc2_color, s=point_size, alpha=alpha, 
                    label=f'Correlated to PC2 ({len(pc2_x)} variables)', zorder=3)
        
        # Lignes des axes
        ax.axhline(0, color='grey', lw=1, linestyle='--', alpha=0.8)
        ax.axvline(0, color='grey', lw=1, linestyle='--', alpha=0.8)
        
        # Configuration des axes
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(f'Principal Component 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax.set_ylabel(f'Principal Component 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax.set_title('Correlation Circle - Points colored by principal axis', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Légende
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)
        
        plt.tight_layout()
        plt.savefig('cercle_correlation_sans_labels.png', dpi=400, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Cercle affiché avec {len(pc1_x)} variables corrélées à PC1 et {len(pc2_x)} variables corrélées à PC2")

    def export_unified_groups_analysis(self, output_file='unified_groups_analysis.csv'):
        """
        Exporte l'analyse des groupes du cercle unifié dans un fichier CSV.
        
        Args:
            output_file (str): Nom du fichier de sortie
        """
        if self.pca is None:
            print("L'ACP n'a pas été calculée.")
            return None
        
        components = self.pca.components_
        analysis_data = []
        
        for i, var_name in enumerate(self.variables):
            x = components[0, i]  # PC1
            y = components[1, i]  # PC2
            
            cos2_pc1 = x**2
            cos2_pc2 = y**2
            cos2_total = cos2_pc1 + cos2_pc2
            
            # Déterminer le groupe
            ratio_threshold = 0.7
            if cos2_pc1 > cos2_pc2:
                if cos2_pc2 / cos2_pc1 > ratio_threshold:
                    group = "Mixte"
                else:
                    group = "PC1"
            else:
                if cos2_pc1 / cos2_pc2 > ratio_threshold:
                    group = "Mixte"
                else:
                    group = "PC2"
            
            # Calculer les métriques
            distance_origin = np.sqrt(x**2 + y**2)
            angle = np.degrees(np.arctan2(y, x))
            if angle < 0:
                angle += 360
            
            dominance_ratio = max(cos2_pc1, cos2_pc2) / min(cos2_pc1, cos2_pc2) if min(cos2_pc1, cos2_pc2) > 0 else np.inf
            
            analysis_data.append({
                'Variable': var_name,
                'Groupe': group,
                'PC1_loading': x,
                'PC2_loading': y,
                'Cos2_PC1': cos2_pc1,
                'Cos2_PC2': cos2_pc2,
                'Cos2_total': cos2_total,
                'Distance_origine': distance_origin,
                'Angle_degres': angle,
                'Ratio_dominance': dominance_ratio,
                'Qualite_representation': 'Bonne' if cos2_total > 0.5 else 'Moyenne' if cos2_total > 0.3 else 'Faible'
            })
        
        df_analysis = pd.DataFrame(analysis_data)
        df_analysis = df_analysis.sort_values(['Groupe', 'Cos2_total'], ascending=[True, False])
        
        # Sauvegarder
        df_analysis.to_csv(output_file, index=False)
        print(f"Analyse des groupes unifiés sauvegardée dans '{output_file}'")
        
        return df_analysis
    
    


    def run_acp_complet(self):
        """
        Exécute l'ensemble du processus ACP en appelant toutes les méthodes nécessaires
        dans le bon ordre.
        """
        print("\n--- Début du processus d'Analyse en Composantes Principales (ACP) ---")
        try:
            self.preparer_donnees()
            self.verifier_kmo_bartlett()
            self.calculer_acp()
            self.plot_variance_expliquee()
            self.plot_cercle_correlation(n_top_vars=15) # Affiche les 15 variables les plus importantes
            self.plot_heatmap_contributions(n_top_pcs=5) # Affiche les contributions pour les 5 premières PC
            # self.plot_cercle_correlation_interactif() 
            print("Retour de PCA :", type(self.df_pca), type(self.pca))
            print("\n--- Processus ACP terminé avec succès ! 🎉 ---")
            print("Retour de PCA :", type(self.df_pca), type(self.pca))

            return self.df_pca, self.pca
        except ValueError as e:
            print(f"\n--- Erreur lors de l'exécution de l'ACP : {e} ❌ ---")
            return None, None
        except Exception as e:
            print(f"\n--- Une erreur inattendue est survenue lors de l'ACP : {e} 🐛 ---")
            return None, None




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator # Assurez-vous d'avoir installé 'kneed' (pip install kneed)

class ClusteringProcessor:
    def __init__(self, df_pca, k_range=range(2, 13)):
        """
        Initialise le ClusteringProcessor.

        Args:
            df_pca (pd.DataFrame): DataFrame contenant les composantes principales (au moins 'PC1' et 'PC2').
            k_range (range): Plage de k à tester pour le K-means (nombre de clusters).
        """
        self.df_pca = df_pca.copy()
        self.k_range = k_range
        self.best_k = None
        self.kmeans_model = None
        self.df_clusters = None

    def _nettoyer_pcs(self):
        """
        Corrige les colonnes 'PC' qui contiendraient des listes ou arrays,
        en extrayant le premier élément s'il existe, ou np.nan sinon.
        """
        pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
        if not pcs:
            print("⚠️ Aucune colonne 'PC' trouvée pour le nettoyage.")
            return

        for pc in pcs:
            # Vérifie si le premier élément de la colonne est une liste ou un array NumPy
            # Utilise .iloc[0] pour éviter l'évaluation paresseuse ou les Series vides
            if not self.df_pca[pc].empty and isinstance(self.df_pca[pc].iloc[0], (np.ndarray, list)):
                print(f"Correction de la colonne '{pc}' qui contient des arrays/listes...")
                self.df_pca[pc] = self.df_pca[pc].apply(lambda x: x[0] if len(x) > 0 else np.nan)
        print("Nettoyage des colonnes 'PC' terminé.")


    def _calculer_inertie(self):
        """
        Calcule l'inertie pour chaque k dans la plage définie et détecte le "coude"
        pour suggérer le meilleur nombre de clusters.
        """
        print("Calcul de l'inertie pour chaque k...")
        pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
        if not pcs:
            raise ValueError("Aucune colonne 'PC' trouvée dans self.df_pca. Assurez-vous que la PCA a été effectuée et que le DataFrame est correctement défini.")

        # Supprimer les lignes avec des valeurs NaN dans les colonnes PCA avant le clustering
        X = self.df_pca[pcs].dropna().values
        if X.size == 0:
            raise ValueError("Le DataFrame PCA est vide après avoir supprimé les NaN, impossible d'effectuer le clustering.")

        inertias = []
        for k in self.k_range:
            # Assurer un nombre d'initialisations suffisant pour des résultats stables
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Détecter le coude (point d'inflexion) de la courbe d'inertie
        kneedle = KneeLocator(list(self.k_range), inertias, curve='convex', direction='decreasing')

        # Si aucun coude n'est détecté, une valeur par défaut de 3 est utilisée.
        self.best_k = kneedle.knee if kneedle.knee is not None else 3
        print(f"Meilleur k détecté par la méthode du coude : {self.best_k} 🎉")

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.k_range), inertias, marker='o', linestyle='-', color='skyblue')

        # Affiche une ligne verticale au niveau du meilleur k détecté
        plt.vlines(self.best_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='red', label=f'best k = {self.best_k}')

        plt.title("Elbow Method (intra-cluster Inertia)", fontsize=14)
        plt.xlabel("Number of cluster(k)", fontsize=12)
        plt.ylabel("Inertia", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(list(self.k_range)) # Assure que tous les k de la plage sont affichés sur l'axe x
        plt.tight_layout()
        plt.show()

    #------------------------------------------------------------------------------------------------------------------------

    def _clustering_final(self, n_clusters_override=None):
        """
        Applique l'algorithme K-means sur les données PCA avec le nombre de clusters
        spécifié ou le meilleur k détecté précédemment.
        Réassigne aussi les labels de clusters pour cohérence entre visualisation et stats.
        """
        pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
        if not pcs:
            raise ValueError("Aucune colonne 'PC' trouvée dans self.df_pca. Assurez-vous que la PCA a été effectuée.")

        # Supprimer les lignes NaN
        X = self.df_pca[pcs].dropna().values
        if X.size == 0:
            raise ValueError("Le DataFrame PCA est vide après avoir supprimé les NaN, impossible d'effectuer le clustering.")

        # Déterminer le nombre de clusters
        if n_clusters_override is not None:
            n_clusters_to_use = n_clusters_override
            print(f"Clustering final avec k = {n_clusters_to_use} (manuel)...")
        elif self.best_k is not None:
            n_clusters_to_use = self.best_k
            print(f"Clustering final avec k = {n_clusters_to_use} (meilleur k détecté automatiquement)...")
        else:
            raise ValueError("Aucun nombre de clusters n'a été déterminé. Exécutez '_calculer_inertie()' ou spécifiez 'n_clusters_override'.")

        # KMeans
        self.kmeans_model = KMeans(n_clusters=n_clusters_to_use, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(X)

        # Ajouter les clusters au DataFrame
        self.df_clusters = self.df_pca.copy()
        original_indices = self.df_pca[pcs].dropna().index
        self.df_clusters.loc[original_indices, 'cluster'] = labels + 1  # Label brut (1..k)

        # === Réassignation cohérente des clusters ===
        centers = self.kmeans_model.cluster_centers_[:, :2]  # sur PC1 et PC2
        df_centers = pd.DataFrame(centers, columns=['PC1', 'PC2'])
        df_centers['old_label'] = np.arange(1, len(centers) + 1)

        new_labels_map = {}
        # Exemple de règles arbitraires pour fixer les labels (tu peux ajuster selon ton projet) :
        new_labels_map[df_centers.loc[df_centers['PC1'] + df_centers['PC2'] == (df_centers['PC1'] + df_centers['PC2']).min(), 'old_label'].values[0]] = 3  # cluster sain
        new_labels_map[df_centers.loc[df_centers['PC1'] == df_centers['PC1'].max(), 'old_label'].values[0]] = 4  # extrême
        new_labels_map[df_centers.loc[df_centers['PC2'] == df_centers['PC2'].max(), 'old_label'].values[0]] = 2  # haut risque
        remaining_labels = set(df_centers['old_label']) - set(new_labels_map.keys())
        for l in remaining_labels:
            new_labels_map[l] = 1  # autres

        # Appliquer la réassignation
        self.df_clusters['cluster_relabel'] = self.df_clusters['cluster'].map(new_labels_map)

        print("✅ Clustering effectué et labels réassignés pour cohérence entre stats et visualisation.")


    #------------------------------------------------------------------------------------------------------------------------

    def _visualiser_clusters(self, PC1, PC2):
        """
        Visualise les clusters sur les deux premières composantes principales (PC1 et PC2).
        Utilise les labels réassignés ('cluster_relabel') pour cohérence avec les stats.
        """

        if self.df_clusters is None or self.kmeans_model is None:
            print("Impossible de visualiser : Le clustering n'a pas été effectué. Exécutez '_clustering_final()' d'abord. ❌")
            return

        if PC1 not in self.df_clusters.columns or PC2 not in self.df_clusters.columns:
            print(f"Impossible de visualiser : Les colonnes '{PC1}' et/ou '{PC2}' sont manquantes dans le DataFrame des clusters. ⚠️")
            return

        print("Visualisation des clusters... 📊")

        # Palette de couleurs fixe
        palette_custom = {
            1: '#1f77b4',   # Bleu
            2: '#ff7f0e',   # Orange
            3: '#2ca02c',   # Vert
            4: '#d62728',   # Rouge
            5: '#9467bd',   # Violet
            6: '#8c564b',   # Marron
            7: '#e377c2',   # Rose
            8: '#7f7f7f',   # Gris
            9: '#bcbd22',   # Jaune-vert
            10: '#17becf',  # Cyan
            11: '#aec7e8',  # Bleu clair
            12: '#ffbb78',  # Orange clair
            13: '#98df8a'   # Vert clair
        }

        # Détection des clusters présents
        clusters_detectes = sorted(self.df_clusters['cluster_relabel'].dropna().unique())
        palette_utilisee = {k: palette_custom.get(k, '#333333') for k in clusters_detectes}

        # Scatterplot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=self.df_clusters.dropna(subset=['cluster_relabel']),
            x=PC1,
            y=PC2,
            hue='cluster_relabel',
            palette=palette_utilisee,
            s=30,
            alpha=0.7,
            legend='full'
        )

        # Annoter les centres (déjà dans le modèle KMeans)
        centers = self.kmeans_model.cluster_centers_[:, :2]
        for i, center in enumerate(centers):
            plt.text(center[0], center[1], f'Cluster {i+1}', fontsize=11, fontweight='bold',
                    ha='center', va='center', color='darkblue',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'))

        plt.title('Clusters visualisation on the first 2 principal components', fontsize=12)
        plt.xlabel(PC1, fontsize=13)
        plt.ylabel(PC2, fontsize=13)
        plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.98, 1])
        plt.savefig(f"projection_{PC1}_{PC2}.png", dpi=600)
        print(f"Graphique des clusters enregistré sous 'projection_{PC1}_{PC2}.png'")
        plt.show()




    #------------------------------------------------------------------------------------------------------------------------

    def run_clustering(self, PC1, PC2, n_clusters=None):
        """
        Exécute le processus complet de clustering :
        1. Nettoyage des colonnes PCA.
        2. Calcul de l'inertie (si n_clusters n'est pas spécifié).
        3. Application du clustering final.
        4. Visualisation des clusters.

        Args:
            PC1 (str): Nom de la colonne pour la première composante principale.
            PC2 (str): Nom de la colonne pour la deuxième composante principale.
            n_clusters (int, optional): Le nombre de clusters à utiliser.
                                        Si spécifié, la méthode du coude est ignorée.
                                        Si None, la méthode du coude est utilisée pour déterminer le meilleur k.
        Returns:
            tuple: Un tuple contenant le DataFrame avec les labels de cluster et le modèle KMeans final.
        """
        print("\n--- Début du processus de clustering ---")

        self._nettoyer_pcs()

        # Nouvelle logique : Appeler _calculer_inertie UNIQUEMENT si n_clusters n'est pas spécifié
        if n_clusters is None:
            self._calculer_inertie() # La méthode du coude est appelée et son graphique s'affiche.
        else:
            print(f"Un nombre de clusters ({n_clusters}) a été spécifié, la méthode du coude ne sera pas exécutée pour la détection automatique. ℹ️")
            self.best_k = None # Assure que best_k est None si non calculé, clarifiant qu'il n'est pas le choix automatique.

        self._clustering_final(n_clusters_override=n_clusters)
        self._visualiser_clusters(PC1,PC2)

        print("--- Processus de clustering terminé --- ✅")
        return self.df_clusters, self.kmeans_model
    


    


# import pandas as pd
# from adjustText import adjust_text
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import plotly.graph_objects as go
# import io
# import base64
# from kneed import KneeLocator
# from sklearn.cluster import KMeans

# class PCAProcessor:
#     def __init__(self, df, variables, var_expliquee_min=0.99):
#         """
#         Initialise l'ACP Analyzer.
#         Args:
#             df (pd.DataFrame): Les données.
#             variables (list): Liste des variables à inclure dans l'ACP.
#             var_expliquee_min (float): Seuil de variance expliquée cumulée pour retenir les composantes.
#         """
#         self.df = df
#         self.df = self.df.reset_index(drop=True)
#         self.variables = variables
#         self.var_expliquee_min = var_expliquee_min
#         self.scaler = StandardScaler()
#         self.pca = None
#         self.X_scaled = None
#         self.X_pca = None
#         self.n_components = None
#         self.df_pca = None
#         self.df_clean_original_index = None # Pour garder la trace des index originaux des lignes non NaN

#     def preparer_donnees(self):
#         """
#         Prépare les données en sélectionnant les variables, en supprimant les lignes avec des NaN,
#         et en standardisant les données.
#         """
#         message = "Préparation et normalisation des données... 🧹\n"
        
#         X = self.df[self.variables]
#         self.df_clean_original_index = X.dropna().index
#         X_clean = X.dropna()

#         if X_clean.empty:
#             raise ValueError("Aucune donnée disponible après suppression des NaN dans les variables spécifiées. Veuillez vérifier vos données ou la liste des variables.")
        
#         self.X_scaled = self.scaler.fit_transform(X_clean)
#         self.df_clean = X_clean.reset_index(drop=True) 
#         message += f"Données préparées. {len(self.df_clean)} lignes utilisées après suppression des NaN. ✅"
#         return self.df_clean, message

#     def verifier_kmo_bartlett(self):
#         """
#         Effectue le test de Kaiser-Meyer-Olkin (KMO) et le test de sphéricité de Bartlett
#         pour évaluer l'adéquation des données à l'ACP. Retourne un message HTML.
#         """
#         if self.X_scaled is None:
#             raise ValueError("Les données n'ont pas été préparées. Exécutez 'preparer_donnees()' d'abord.")

#         html_message = "<h3>Tests d'adéquation pour l'ACP</h3>"
#         kmo_all, kmo_model = calculate_kmo(self.X_scaled)
#         html_message += f"<p><b>KMO Global</b> = {kmo_model:.3f} (doit être > 0.5 pour une ACP appropriée)</p>"
        
#         chi_square_value, p_value = calculate_bartlett_sphericity(self.X_scaled)
#         html_message += f"<p><b>Bartlett p-value</b> = {p_value:.3e} (doit être < 0.05 pour rejeter l'hypothèse nulle)</p>"
        
#         if kmo_model < 0.5 or p_value >= 0.05:
#             html_message += "<p>⚠️ <b>Attention</b> : Les tests KMO/Bartlett suggèrent que l'ACP pourrait ne pas être la méthode la plus appropriée pour ces données.</p>"
#         else:
#             html_message += "<p>Tests KMO/Bartlett réussis. Les données sont adéquates pour l'ACP. 👍</p>"
#         return html_message

#     def calculer_acp(self):
#         """
#         Calcule les composantes principales, détermine le nombre optimal de composantes
#         basé sur la variance expliquée minimale, et construit le DataFrame des composantes.
#         """
#         if self.X_scaled is None:
#             raise ValueError("Les données n'ont pas été standardisées. Exécutez 'preparer_donnees()' d'abord.")

#         self.pca = PCA()
#         self.X_pca = self.pca.fit_transform(self.X_scaled)
        
#         explained_var = self.pca.explained_variance_ratio_
#         cum_var = explained_var.cumsum()
        
#         self.n_components = np.argmax(cum_var >= self.var_expliquee_min) + 1
        
#         message = f"Nombre de composantes retenues pour {self.var_expliquee_min*100:.1f}% de variance expliquée : {self.n_components} 📈"
        
#         df_pca_only = pd.DataFrame(self.X_pca[:, :self.n_components], 
#                                     columns=[f'PC{i+1}' for i in range(self.n_components)])
        
#         self.df_pca = pd.concat([self.df.loc[self.df_clean_original_index].reset_index(drop=True), 
#                                   df_pca_only], axis=1)
        
#         message += "\nDataFrame PCA généré avec succès. ✨"
#         return self.df_pca, message

#     def plot_to_html(self, fig):
#         """
#         Convertit une figure matplotlib en une balise HTML <img> avec l'image encodée en base64.
#         """
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         img_str = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close(fig)
#         return f'<img src="data:image/png;base64,{img_str}" alt="Matplotlib Plot">'

#     def get_top_variables_for_plot(self, n_top=20, method='loadings_abs_sum'):
#         #... (méthode non modifiée, elle est interne)
#         if self.pca is None or self.X_scaled is None:
#             return []

#         if self.pca.components_.shape[0] < 2:
#             return []

#         loadings_df = pd.DataFrame(self.pca.components_[:2].T, 
#                                    index=self.variables,
#                                    columns=['PC1', 'PC2'])

#         if method == 'loadings_abs_sum':
#             scores = loadings_df.abs().sum(axis=1)
#         elif method == 'cos2_sum':
#             cos2_df = loadings_df**2
#             scores = cos2_df.sum(axis=1)
#         elif method == 'contributions':
#             explained_var_pc1 = self.pca.explained_variance_ratio_[0]
#             explained_var_pc2 = self.pca.explained_variance_ratio_[1]
            
#             contributions_pc1 = (loadings_df['PC1']**2 * 100) / explained_var_pc1
#             contributions_pc2 = (loadings_df['PC2']**2 * 100) / explained_var_pc2
#             scores = contributions_pc1 + contributions_pc2
#         else:
#             raise ValueError("Méthode de sélection invalide. Choisissez parmi 'loadings_abs_sum', 'cos2_sum', 'contributions'.")

#         return scores.nlargest(n_top).index.tolist()

#     def plot_variance_expliquee(self):
#         """
#         Affiche le graphique de la variance expliquée cumulée. Retourne une balise HTML.
#         """
#         if self.pca is None:
#             return "L'ACP n'a pas été calculée. ❌"

#         explained_var = self.pca.explained_variance_ratio_
#         cum_var = explained_var.cumsum()
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-', color='indigo', linewidth=2)
#         ax.axhline(self.var_expliquee_min, color='red', linestyle='--', label=f'Seuil {self.var_expliquee_min*100:.1f}%')
#         ax.axvline(self.n_components, color='green', linestyle=':', label=f'{self.n_components} Composantes')
        
#         ax.set_title("Variance Expliquée Cumulée par Composante Principale", fontsize=15)
#         ax.set_xlabel("Nombre de Composantes Principales", fontsize=12)
#         ax.set_ylabel("Variance Expliquée Cumulée", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend(fontsize=10)
#         ax.set_xticks(range(1, len(cum_var) + 1))
        
#         return self.plot_to_html(fig)

#     def plot_cercle_correlation(self, n_top_vars=20):
#         """
#         Affiche le cercle des corrélations. Retourne une balise HTML.
#         """
#         if self.pca is None or self.pca.components_.shape[0] < 2:
#             return "Moins de 2 composantes principales disponibles pour le cercle de corrélation. ⚠️"

#         vars_to_plot = self.variables
#         if n_top_vars is not None and len(self.variables) > n_top_vars:
#             vars_to_plot = self.get_top_variables_for_plot(n_top=n_top_vars)

#         components = self.pca.components_
        
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
#         ax.add_artist(circle)
        
#         texts = []
#         for var_name in vars_to_plot:
#             idx = self.variables.index(var_name)
#             x = components[0, idx]
#             y = components[1, idx]
#             ax.arrow(0, 0, x, y, color='black', alpha=0.7, head_width=0.03, length_includes_head=True)
#             texts.append(ax.text(x * 1.05, y * 1.05, var_name, color='darkgreen', ha='center', va='center', fontsize=9))
        
#         adjust_text(texts, arrowprops=dict(arrowstyle="-", color='lightgrey', lw=0.5))
        
#         ax.axhline(0, color='grey', lw=0.8, linestyle='--')
#         ax.axvline(0, color='grey', lw=0.8, linestyle='--')
        
#         ax.set_xlim(-1.1, 1.1)
#         ax.set_ylim(-1.1, 1.1)
#         ax.set_xlabel(f'Composante principale 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
#         ax.set_ylabel(f'Composante principale 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
#         ax.set_title('Cercle des corrélations (Variables et Composantes)', fontsize=15)
#         ax.set_aspect('equal')
#         ax.grid(True, linestyle=':', alpha=0.5)
        
#         return self.plot_to_html(fig)

#     def plot_heatmap_contributions(self, n_top_pcs=5):
#         """
#         Affiche une heatmap des contributions des variables. Retourne une balise HTML.
#         """
#         if self.pca is None:
#             return "L'ACP n'a pas été calculée. ❌"

#         num_pcs_to_plot = min(n_top_pcs, self.pca.components_.shape[0])
#         if num_pcs_to_plot == 0:
#             return "Aucune composante principale à afficher dans la heatmap. ⚠️"

#         loadings_squared = self.pca.components_[:num_pcs_to_plot]**2
#         loadings_df = pd.DataFrame(loadings_squared.T, 
#                                    index=self.variables, 
#                                    columns=[f'PC{i+1}' for i in range(num_pcs_to_plot)])
        
#         loadings_df['Total_Contribution'] = loadings_df.sum(axis=1)
#         loadings_df = loadings_df.sort_values(by='Total_Contribution', ascending=False).drop(columns='Total_Contribution')

#         fig, ax = plt.subplots(figsize=(min(num_pcs_to_plot * 2 + 4, 16), min(len(self.variables) * 0.3 + 2, 20)))
#         sns.heatmap(loadings_df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Loadings au Carré (Proxy de Contribution)'}, ax=ax)
        
#         ax.set_title(f'Importance des Variables sur les {num_pcs_to_plot} Premières Composantes Principales', fontsize=15)
#         ax.set_xlabel('Composantes Principales', fontsize=12)
#         ax.set_ylabel('Variables', fontsize=12)
        
#         return self.plot_to_html(fig)

#     def plot_cercle_correlation_interactif(self):
#         """
#         Affiche le cercle des corrélations en version interactive avec Plotly. Retourne une chaîne HTML.
#         """
#         if self.pca is None or self.pca.components_.shape[0] < 2:
#             return "L'ACP n'a pas été calculée ou moins de 2 composantes principales sont disponibles. ❌"
            
#         components = self.pca.components_
#         fig = go.Figure()

#         theta = np.linspace(0, 2 * np.pi, 100)
#         fig.add_trace(go.Scatter(
#             x=np.cos(theta), y=np.sin(theta), mode='lines',
#             line=dict(color='blue', dash='dash'), name='Cercle Unité'
#         ))

#         for i, var in enumerate(self.variables):
#             x = components[0, i]
#             y = components[1, i]
#             fig.add_trace(go.Scatter(
#                 x=[0, x], y=[0, y], mode='lines+markers+text',
#                 text=[None, var], textposition='top center', name=var,
#                 line=dict(color='red', width=2), marker=dict(size=6, color='red')
#             ))

#         fig.update_layout(
#             title='Cercle des corrélations (ACP) - Interactif',
#             xaxis=dict(title=f'Composante Principale 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', 
#                         range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
#             yaxis=dict(title=f'Composante Principale 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', 
#                         range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
#             width=750, height=750, showlegend=False, hovermode='closest'
#         )
#         # Retourne le HTML pour l'intégration, pas la figure complète
#         return fig.to_html(full_html=False, include_plotlyjs='cdn')


#     def run_acp_complet(self):
#         """
#         Exécute l'ensemble du processus ACP en appelant toutes les méthodes nécessaires
#         dans le bon ordre. Retourne un dictionnaire de résultats HTML et le DataFrame PCA.
#         """
#         results_html = {}
#         try:
#             _, results_html['preparer_donnees'] = self.preparer_donnees()
#             results_html['verifier_kmo_bartlett'] = self.verifier_kmo_bartlett()
#             df_pca, results_html['calculer_acp'] = self.calculer_acp()
#             results_html['plot_variance_expliquee'] = self.plot_variance_expliquee()
#             results_html['plot_cercle_correlation'] = self.plot_cercle_correlation(n_top_vars=15)
#             results_html['plot_heatmap_contributions'] = self.plot_heatmap_contributions(n_top_pcs=5)
#             # results_html['plot_cercle_correlation_interactif'] = self.plot_cercle_correlation_interactif()
#             return df_pca, results_html
#         except ValueError as e:
#             return None, {'error': f"--- Erreur lors de l'exécution de l'ACP : {e} ❌ ---"}

# class ClusteringProcessor:
#     def __init__(self, df_pca, k_range=range(2, 13)):
#         self.df_pca = df_pca.copy()
#         self.k_range = k_range
#         self.best_k = None
#         self.kmeans_model = None
#         self.df_clusters = None
#         self.pca = None
    
#     def plot_to_html(self, fig):
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         img_str = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close(fig)
#         return f'<img src="data:image/png;base64,{img_str}" alt="Matplotlib Plot">'

#     def _nettoyer_pcs(self):
#         #... (méthode non modifiée, elle est interne)
#         pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
#         if not pcs:
#             return "⚠️ Aucune colonne 'PC' trouvée pour le nettoyage."
#         for pc in pcs:
#             if not self.df_pca[pc].empty and isinstance(self.df_pca[pc].iloc[0], (np.ndarray, list)):
#                 self.df_pca[pc] = self.df_pca[pc].apply(lambda x: x[0] if len(x) > 0 else np.nan)
#         return "Nettoyage des colonnes 'PC' terminé."


#     def _calculer_inertie(self):
#         """
#         Calcule l'inertie et détecte le "coude". Retourne le graphique en HTML.
#         """
#         pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
#         if not pcs:
#             raise ValueError("Aucune colonne 'PC' trouvée.")
#         X = self.df_pca[pcs].dropna().values
#         if X.size == 0:
#             raise ValueError("Le DataFrame PCA est vide après avoir supprimé les NaN.")

#         inertias = []
#         for k in self.k_range:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             kmeans.fit(X)
#             inertias.append(kmeans.inertia_)

#         kneedle = KneeLocator(list(self.k_range), inertias, curve='convex', direction='decreasing')
#         self.best_k = kneedle.knee if kneedle.knee is not None else 3
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(list(self.k_range), inertias, marker='o', linestyle='-', color='skyblue')
#         ax.vlines(self.best_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='red', label=f'Meilleur k = {self.best_k}')
#         ax.set_title("Méthode du coude (Inertie intra-cluster)", fontsize=14)
#         ax.set_xlabel("Nombre de clusters (k)", fontsize=12)
#         ax.set_ylabel("Inertie", fontsize=12)
#         ax.grid(True, linestyle=':', alpha=0.7)
#         ax.legend(fontsize=10)
#         ax.set_xticks(list(self.k_range))
#         return self.plot_to_html(fig), f"Meilleur k détecté par la méthode du coude : {self.best_k} 🎉"

#     def _clustering_final(self, n_clusters_override=None):
#         #... (méthode non modifiée, elle est interne)
#         pcs = [col for col in self.df_pca.columns if col.startswith('PC')]
#         if not pcs:
#             raise ValueError("Aucune colonne 'PC' trouvée dans self.df_pca.")

#         X = self.df_pca[pcs].dropna().values
#         if X.size == 0:
#             raise ValueError("Le DataFrame PCA est vide après avoir supprimé les NaN.")

#         if n_clusters_override is not None:
#             n_clusters_to_use = n_clusters_override
#             message = f"Clustering final avec k = {n_clusters_to_use} (nombre de clusters spécifié manuellement)..."
#         elif self.best_k is not None:
#             n_clusters_to_use = self.best_k
#             message = f"Clustering final avec k = {n_clusters_to_use} (meilleur k détecté automatiquement)..."
#         else:
#             raise ValueError("Aucun nombre de clusters n'a été déterminé.")

#         self.kmeans_model = KMeans(n_clusters=n_clusters_to_use, random_state=42, n_init=10)
#         labels = self.kmeans_model.fit_predict(X)

#         self.df_clusters = self.df_pca.copy()
#         original_indices = self.df_pca[pcs].dropna().index
#         self.df_clusters.loc[original_indices, 'cluster'] = labels + 1

#         return self.df_clusters, message

#     def _visualiser_clusters(self, pc_x, pc_y):
#         """
#         Visualise les clusters sur les composantes principales pc_x et pc_y. Retourne une balise HTML.
#         """
#         if self.df_clusters is None or self.kmeans_model is None:
#             return "Le clustering n'a pas été effectué. ❌"

#         if pc_x not in self.df_clusters.columns or pc_y not in self.df_clusters.columns:
#             return f"Impossible de visualiser : Les colonnes '{pc_x}' et/ou '{pc_y}' sont manquantes. ⚠️"

#         palette_custom = {
#             1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd', 
#             6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#bcbd22', 10: '#17becf',
#             11: '#aec7e8', 12: '#ffbb78', 13: '#98df8a'
#         }
#         clusters_detectes = sorted(self.df_clusters['cluster'].dropna().unique())
#         palette_utilisee = {k: palette_custom.get(k, '#333333') for k in clusters_detectes}

#         fig, ax = plt.subplots(figsize=(12, 8))
#         sns.scatterplot(data=self.df_clusters.dropna(subset=['cluster']), x=pc_x, y=pc_y, 
#                         hue='cluster', palette=palette_utilisee, s=30, alpha=0.7, legend='full', ax=ax)

#         centers = self.kmeans_model.cluster_centers_
#         center_pc_x_idx = int(pc_x.replace('PC', '')) - 1
#         center_pc_y_idx = int(pc_y.replace('PC', '')) - 1
        
#         for i, center in enumerate(centers):
#             ax.text(center[center_pc_x_idx], center[center_pc_y_idx], f'Cluster {i+1}', fontsize=11, fontweight='bold',
#                     ha='center', va='center', color='darkblue', 
#                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'))

#         # Check if pca is set
#         if self.pca is not None:
#             pc_x_explained_variance = self.pca.explained_variance_ratio_[center_pc_x_idx] * 100
#             pc_y_explained_variance = self.pca.explained_variance_ratio_[center_pc_y_idx] * 100
#             ax.set_xlabel(f'{pc_x} ({pc_x_explained_variance:.1f}%)', fontsize=13)
#             ax.set_ylabel(f'{pc_y} ({pc_y_explained_variance:.1f}%)', fontsize=13)
#         else:
#             ax.set_xlabel(f'{pc_x}', fontsize=13)
#             ax.set_ylabel(f'{pc_y}', fontsize=13)

#         ax.set_title(f'Clusters sur {pc_x} et {pc_y}', fontsize=16)
#         ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
#         ax.grid(True, linestyle='--', alpha=0.6)
        
#         return self.plot_to_html(fig)

#     def run_clustering(self, pca_model, pc_x, pc_y, n_clusters=None):
#         """
#         Exécute le processus complet de clustering et retourne un dictionnaire de résultats HTML.
#         Args:
#             pca_model (PCAProcessor): L'objet PCAProcessor déjà exécuté.
#             pc_x (str): La première composante principale à visualiser (ex: 'PC1').
#             pc_y (str): La deuxième composante principale à visualiser (ex: 'PC2').
#             n_clusters (int, optional): Le nombre de clusters à utiliser.
#         """
#         self.pca = pca_model.pca
#         results_html = {}
#         try:
#             results_html['nettoyage'] = self._nettoyer_pcs()
#             if n_clusters is None:
#                 results_html['elbow_plot'], results_html['elbow_message'] = self._calculer_inertie()
#             else:
#                 self.best_k = n_clusters
#                 results_html['elbow_message'] = f"Un nombre de clusters ({n_clusters}) a été spécifié, la méthode du coude ne sera pas exécutée."
            
#             self.df_clusters, results_html['clustering_message'] = self._clustering_final(n_clusters_override=n_clusters)
#             results_html['cluster_plot'] = self._visualiser_clusters(pc_x, pc_y)
            
#             return self.df_clusters, results_html
#         except ValueError as e:
#             return None, {'error': f"--- Erreur lors de l'exécution du clustering : {e} ❌ ---"}

# def generate_html_report(pca_processor, clustering_processor, pc_x='PC1', pc_y='PC2', n_clusters=None, output_filename='pca_clustering_report.html'):
#     """
#     Exécute le pipeline ACP et Clustering et génère un rapport HTML complet.
#     Args:
#         pca_processor (PCAProcessor): L'objet PCAProcessor initialisé.
#         clustering_processor (ClusteringProcessor): L'objet ClusteringProcessor initialisé.
#         pc_x (str): La première composante principale pour le graphique des clusters.
#         pc_y (str): La deuxième composante principale pour le graphique des clusters.
#         n_clusters (int, optional): Le nombre de clusters à utiliser.
#         output_filename (str): Le nom du fichier HTML à sauvegarder.
#     """
    
#     df_pca, pca_results_html = pca_processor.run_acp_complet()
    
#     if df_pca is None:
#         print(pca_results_html.get('error', 'Une erreur inconnue est survenue lors de l\'ACP.'))
#         return

#     clustering_processor.df_pca = df_pca
#     df_clusters, cluster_results_html = clustering_processor.run_clustering(pca_processor, pc_x=pc_x, pc_y=pc_y, n_clusters=n_clusters)

#     html_content = f"""

#         <!DOCTYPE html>
#         <html lang="fr">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Visualisation Interactive ACP</title>
#             <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
#             <style>
#                 body {
#                     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                     margin: 0;
#                     padding: 20px;
#                     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                     color: #333;
#                     min-height: 100vh;
#                 }
                
#                 .container {
#                     width: 100%;
#                     max-width: 1400px;
#                     margin: 0 auto;
#                     background: white;
#                     border-radius: 20px;
#                     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#                     overflow: hidden;
#                 }
                
#                 .header {
#                     background: linear-gradient(135deg, #667eea, #764ba2);
#                     color: white;
#                     padding: 30px;
#                     text-align: center;
#                 }
                
#                 .header h1 {
#                     margin: 0;
#                     font-size: 2.5em;
#                     font-weight: 300;
#                 }
                
#                 .controls {
#                     padding: 30px;
#                     background: #f8f9fa;
#                     border-bottom: 1px solid #dee2e6;
#                 }
                
#                 .control-group {
#                     display: grid;
#                     grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#                     gap: 20px;
#                     margin-bottom: 20px;
#                 }
                
#                 .control-item {
#                     background: white;
#                     padding: 20px;
#                     border-radius: 10px;
#                     box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#                 }
                
#                 .control-item label {
#                     display: block;
#                     font-weight: 600;
#                     margin-bottom: 10px;
#                     color: #495057;
#                 }
                
#                 .control-item select, .control-item input {
#                     width: 100%;
#                     padding: 10px;
#                     border: 2px solid #e9ecef;
#                     border-radius: 8px;
#                     font-size: 14px;
#                     transition: border-color 0.3s ease;
#                 }
                
#                 .control-item select:focus, .control-item input:focus {
#                     outline: none;
#                     border-color: #667eea;
#                 }
                
#                 button {
#                     background: linear-gradient(135deg, #667eea, #764ba2);
#                     color: white;
#                     border: none;
#                     padding: 12px 30px;
#                     border-radius: 25px;
#                     font-size: 16px;
#                     font-weight: 600;
#                     cursor: pointer;
#                     transition: all 0.3s ease;
#                     margin: 10px;
#                 }
                
#                 button:hover {
#                     transform: translateY(-2px);
#                     box-shadow: 0 10px 20px rgba(0,0,0,0.2);
#                 }
                
#                 .visualization-container {
#                     padding: 30px;
#                 }
                
#                 .plot-container {
#                     background: white;
#                     border-radius: 15px;
#                     box-shadow: 0 5px 15px rgba(0,0,0,0.08);
#                     margin-bottom: 30px;
#                 }
                
#                 .stats-grid {
#                     display: grid;
#                     grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#                     gap: 20px;
#                     margin-bottom: 30px;
#                 }
                
#                 .stat-card {
#                     background: linear-gradient(135deg, #74b9ff, #0984e3);
#                     color: white;
#                     padding: 20px;
#                     border-radius: 15px;
#                     text-align: center;
#                     box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#                 }
                
#                 .stat-value {
#                     font-size: 2.5em;
#                     font-weight: bold;
#                     margin-bottom: 5px;
#                 }
                
#                 .stat-label {
#                     font-size: 14px;
#                     opacity: 0.9;
#                 }
                
#                 .variable-list {
#                     max-height: 200px;
#                     overflow-y: auto;
#                     background: #f8f9fa;
#                     border-radius: 8px;
#                     padding: 15px;
#                     margin-top: 10px;
#                 }
                
#                 .variable-item {
#                     display: flex;
#                     align-items: center;
#                     padding: 5px 0;
#                     border-bottom: 1px solid #e9ecef;
#                 }
                
#                 .variable-item:last-child {
#                     border-bottom: none;
#                 }
                
#                 .variable-checkbox {
#                     margin-right: 10px;
#                 }
                
#                 .toggle-section {
#                     background: white;
#                     padding: 20px;
#                     border-radius: 10px;
#                     margin-bottom: 20px;
#                     box-shadow: 0 2px 10px rgba(0,0,0,0.05);
#                 }
                
#                 .section-title {
#                     font-size: 1.2em;
#                     font-weight: 600;
#                     margin-bottom: 15px;
#                     color: #495057;
#                 }
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <div class="header">
#                     <h1>🎯 Visualisation Interactive ACP</h1>
#                     <p>Explorez vos composantes principales de manière dynamique</p>
#                 </div>
                
#                 <div class="controls">
#                     <div class="control-group">
#                         <div class="control-item">
#                             <label>🎚️ Nombre de variables à afficher</label>
#                             <input type="range" id="numVarsSlider" min="5" max="50" value="15" 
#                                 oninput="updateNumVars(this.value)">
#                             <div style="text-align: center; margin-top: 5px;">
#                                 <span id="numVarsValue">15</span> variables
#                             </div>
#                         </div>
                        
#                         <div class="control-item">
#                             <label>📊 Méthode de sélection</label>
#                             <select id="selectionMethod" onchange="updateVisualization()">
#                                 <option value="contribution">Contribution (recommandé)</option>
#                                 <option value="cos2">Qualité de représentation (cos²)</option>
#                                 <option value="loading">Corrélation absolue</option>
#                             </select>
#                         </div>
                        
#                         <div class="control-item">
#                             <label>🎨 Taille des points</label>
#                             <input type="range" id="pointSize" min="5" max="20" value="8" 
#                                 oninput="updatePointSize(this.value)">
#                             <div style="text-align: center; margin-top: 5px;">
#                                 <span id="pointSizeValue">8</span>px
#                             </div>
#                         </div>
                        
#                         <div class="control-item">
#                             <label>📝 Taille des étiquettes</label>
#                             <input type="range" id="labelSize" min="8" max="16" value="10" 
#                                 oninput="updateLabelSize(this.value)">
#                             <div style="text-align: center; margin-top: 5px;">
#                                 <span id="labelSizeValue">10</span>px
#                             </div>
#                         </div>
#                     </div>
                    
#                     <div class="toggle-section">
#                         <div class="section-title">🎛️ Options d'affichage</div>
#                         <label><input type="checkbox" id="showCircle" checked onchange="updateVisualization()"> Afficher le cercle unité</label><br>
#                         <label><input type="checkbox" id="showGrid" checked onchange="updateVisualization()"> Afficher la grille</label><br>
#                         <label><input type="checkbox" id="showArrows" checked onchange="updateVisualization()"> Afficher les flèches</label><br>
#                         <label><input type="checkbox" id="colorByContrib" onchange="updateVisualization()"> Colorier par contribution</label>
#                     </div>
                    
#                     <div class="control-group">
#                         <button onclick="downloadPlot('png')">📸 Télécharger PNG</button>
#                         <button onclick="downloadPlot('svg')">🎨 Télécharger SVG</button>
#                         <button onclick="downloadPlot('html')">💾 Télécharger HTML</button>
#                         <button onclick="resetView()">🔄 Reset Vue</button>
#                     </div>
#                 </div>
                
#                 <div class="visualization-container">
#                     <div class="stats-grid">
#                         <div class="stat-card">
#                             <div class="stat-value" id="pc1Variance">65.4%</div>
#                             <div class="stat-label">Variance PC1</div>
#                         </div>
#                         <div class="stat-card">
#                             <div class="stat-value" id="pc2Variance">23.1%</div>
#                             <div class="stat-label">Variance PC2</div>
#                         </div>
#                         <div class="stat-card">
#                             <div class="stat-value" id="totalVariance">88.5%</div>
#                             <div class="stat-label">Variance Totale</div>
#                         </div>
#                         <div class="stat-card">
#                             <div class="stat-value" id="numVariablesShown">15</div>
#                             <div class="stat-label">Variables Affichées</div>
#                         </div>
#                     </div>
                    
#                     <div class="plot-container">
#                         <div id="correlationCircle" style="width: 100%; height: 700px;"></div>
#                     </div>
                    
#                     <div class="toggle-section">
#                         <div class="section-title">🔍 Sélection manuelle des variables</div>
#                         <div id="variableSelector" class="variable-list">
#                             <!-- Les variables seront ajoutées dynamiquement -->
#                         </div>
#                     </div>
#                 </div>
#             </div>
            
#             <script>
#                 // Données simulées - remplacez par vos vraies données
#                 const pcaData = {
#                     variables: [
#                         'Temperature', 'Humidity', 'PM25', 'PM10', 'NO2', 'O3', 'CO', 'SO2',
#                         'WindSpeed', 'Pressure', 'Rainfall', 'UVIndex', 'Visibility', 'AQI',
#                         'NO', 'NOx', 'NH3', 'NMHC', 'CH4', 'CO2', 'Benzene', 'Toluene',
#                         'Xylene', 'AsthmaRate', 'COPDRate', 'HeartDisease', 'StrokeRate',
#                         'RespiratoryInfection', 'Age_0_14', 'Age_15_64', 'Age_65plus',
#                         'Population', 'PopulationDensity', 'UrbanRural', 'SES_Index'
#                     ],
#                     pc1_loadings: null,
#                     pc2_loadings: null,
#                     explained_variance: [0.654, 0.231, 0.087, 0.028]
#                 };
                
#                 // Génération de loadings simulés
#                 function generateSimulatedLoadings() {
#                     const n_vars = pcaData.variables.length;
#                     pcaData.pc1_loadings = Array.from({length: n_vars}, () => 
#                         (Math.random() - 0.5) * 2 * Math.random());
#                     pcaData.pc2_loadings = Array.from({length: n_vars}, () => 
#                         (Math.random() - 0.5) * 2 * Math.random());
#                 }
                
#                 function calculateContributions() {
#                     return pcaData.variables.map((_, i) => {
#                         const pc1_contrib = Math.pow(pcaData.pc1_loadings[i], 2) / pcaData.explained_variance[0];
#                         const pc2_contrib = Math.pow(pcaData.pc2_loadings[i], 2) / pcaData.explained_variance[1];
#                         return {
#                             index: i,
#                             variable: pcaData.variables[i],
#                             pc1_loading: pcaData.pc1_loadings[i],
#                             pc2_loading: pcaData.pc2_loadings[i],
#                             contribution: pc1_contrib + pc2_contrib,
#                             cos2: Math.pow(pcaData.pc1_loadings[i], 2) + Math.pow(pcaData.pc2_loadings[i], 2),
#                             loading_abs: Math.abs(pcaData.pc1_loadings[i]) + Math.abs(pcaData.pc2_loadings[i])
#                         };
#                     });
#                 }
                
#                 function getTopVariables(method, n_vars) {
#                     const contributions = calculateContributions();
#                     let sortKey;
                    
#                     switch(method) {
#                         case 'contribution':
#                             sortKey = 'contribution';
#                             break;
#                         case 'cos2':
#                             sortKey = 'cos2';
#                             break;
#                         case 'loading':
#                             sortKey = 'loading_abs';
#                             break;
#                         default:
#                             sortKey = 'contribution';
#                     }
                    
#                     return contributions
#                         .sort((a, b) => b[sortKey] - a[sortKey])
#                         .slice(0, n_vars);
#                 }
                
#                 function updateVisualization() {
#                     const numVars = parseInt(document.getElementById('numVarsSlider').value);
#                     const method = document.getElementById('selectionMethod').value;
#                     const showCircle = document.getElementById('showCircle').checked;
#                     const showGrid = document.getElementById('showGrid').checked;
#                     const showArrows = document.getElementById('showArrows').checked;
#                     const colorByContrib = document.getElementById('colorByContrib').checked;
                    
#                     const topVars = getTopVariables(method, numVars);
                    
#                     // Préparer les données pour le graphique
#                     const traces = [];
                    
#                     // Cercle unité
#                     if (showCircle) {
#                         const theta = Array.from({length: 100}, (_, i) => i * 2 * Math.PI / 99);
#                         traces.push({
#                             x: theta.map(t => Math.cos(t)),
#                             y: theta.map(t => Math.sin(t)),
#                             mode: 'lines',
#                             line: {color: 'rgba(0,0,255,0.3)', width: 2, dash: 'dash'},
#                             name: 'Cercle Unité',
#                             showlegend: false,
#                             hoverinfo: 'skip'
#                         });
#                     }
                    
#                     // Variables
#                     const colors = colorByContrib ? 
#                         topVars.map(v => v.contribution) : 
#                         Array(topVars.length).fill('#e74c3c');
                    
#                     const pointSize = parseInt(document.getElementById('pointSize').value);
#                     const labelSize = parseInt(document.getElementById('labelSize').value);
                    
#                     if (showArrows) {
#                         // Flèches
#                         topVars.forEach(v => {
#                             traces.push({
#                                 x: [0, v.pc1_loading],
#                                 y: [0, v.pc2_loading],
#                                 mode: 'lines',
#                                 line: {color: 'rgba(0,0,0,0.6)', width: 2},
#                                 showlegend: false,
#                                 hoverinfo: 'skip'
#                             });
#                         });
#                     }
                    
#                     // Points et étiquettes
#                     traces.push({
#                         x: topVars.map(v => v.pc1_loading),
#                         y: topVars.map(v => v.pc2_loading),
#                         mode: 'markers+text',
#                         marker: {
#                             size: pointSize,
#                             color: colors,
#                             colorscale: 'Viridis',
#                             showscale: colorByContrib,
#                             colorbar: {
#                                 title: 'Contribution',
#                                 titleside: 'right'
#                             },
#                             line: {color: 'white', width: 1}
#                         },
#                         text: topVars.map(v => v.variable),
#                         textposition: 'top center',
#                         textfont: {size: labelSize, color: '#2c3e50'},
#                         name: 'Variables',
#                         showlegend: false,
#                         hovertemplate: 
#                             '<b>%{text}</b><br>' +
#                             'PC1: %{x:.3f}<br>' +
#                             'PC2: %{y:.3f}<br>' +
#                             'Contribution: %{marker.color:.3f}<br>' +
#                             '<extra></extra>'
#                     });
                    
#                     const layout = {
#                         title: {
#                             text: `Cercle des Corrélations - ${method.charAt(0).toUpperCase() + method.slice(1)}`,
#                             font: {size: 20, color: '#2c3e50'}
#                         },
#                         xaxis: {
#                             title: `PC1 (${(pcaData.explained_variance[0] * 100).toFixed(1)}%)`,
#                             range: [-1.2, 1.2],
#                             zeroline: true,
#                             zerolinewidth: 2,
#                             zerolinecolor: '#bdc3c7',
#                             showgrid: showGrid,
#                             gridcolor: 'rgba(189, 195, 199, 0.3)'
#                         },
#                         yaxis: {
#                             title: `PC2 (${(pcaData.explained_variance[1] * 100).toFixed(1)}%)`,
#                             range: [-1.2, 1.2],
#                             zeroline: true,
#                             zerolinewidth: 2,
#                             zerolinecolor: '#bdc3c7',
#                             showgrid: showGrid,
#                             gridcolor: 'rgba(189, 195, 199, 0.3)'
#                         },
#                         plot_bgcolor: 'white',
#                         paper_bgcolor: 'white',
#                         font: {family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'},
#                         hovermode: 'closest'
#                     };
                    
#                     const config = {
#                         displayModeBar: true,
#                         displaylogo: false,
#                         modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
#                         toImageButtonOptions: {
#                             format: 'png',
#                             filename: 'cercle_correlation_acp',
#                             height: 700,
#                             width: 1000,
#                             scale: 2
#                         }
#                     };
                    
#                     Plotly.newPlot('correlationCircle', traces, layout, config);
                    
#                     // Mettre à jour les statistiques
#                     document.getElementById('numVariablesShown').textContent = numVars;
                    
#                     // Mettre à jour la liste des variables sélectionnées
#                     updateVariableSelector(topVars);
#                 }
                
#                 function updateVariableSelector(selectedVars) {
#                     const container = document.getElementById('variableSelector');
#                     container.innerHTML = '';
                    
#                     pcaData.variables.forEach((variable, index) => {
#                         const div = document.createElement('div');
#                         div.className = 'variable-item';
                        
#                         const checkbox = document.createElement('input');
#                         checkbox.type = 'checkbox';
#                         checkbox.className = 'variable-checkbox';
#                         checkbox.checked = selectedVars.some(v => v.variable === variable);
#                         checkbox.onchange = () => updateCustomSelection();
                        
#                         const label = document.createElement('span');
#                         label.textContent = variable;
                        
#                         div.appendChild(checkbox);
#                         div.appendChild(label);
#                         container.appendChild(div);
#                     });
#                 }
                
#                 function updateCustomSelection() {
#                     // Logique pour sélection manuelle
#                     const checkboxes = document.querySelectorAll('.variable-checkbox');
#                     const selectedIndices = [];
                    
#                     checkboxes.forEach((checkbox, index) => {
#                         if (checkbox.checked) {
#                             selectedIndices.push(index);
#                         }
#                     });
                    
#                     // Mettre à jour le slider
#                     document.getElementById('numVarsSlider').value = selectedIndices.length;
#                     updateNumVars(selectedIndices.length);
#                 }
                
#                 function updateNumVars(value) {
#                     document.getElementById('numVarsValue').textContent = value;
#                     updateVisualization();
#                 }
                
#                 function updatePointSize(value) {
#                     document.getElementById('pointSizeValue').textContent = value;
#                     updateVisualization();
#                 }
                
#                 function updateLabelSize(value) {
#                     document.getElementById('labelSizeValue').textContent = value;
#                     updateVisualization();
#                 }
                
#                 function downloadPlot(format) {
#                     const filename = `cercle_correlation_acp.${format}`;
                    
#                     if (format === 'html') {
#                         Plotly.downloadImage('correlationCircle', {
#                             format: 'html',
#                             filename: filename
#                         });
#                     } else {
#                         Plotly.downloadImage('correlationCircle', {
#                             format: format,
#                             filename: filename,
#                             width: 1200,
#                             height: 800,
#                             scale: 2
#                         });
#                     }
#                 }
                
#                 function resetView() {
#                     document.getElementById('numVarsSlider').value = 15;
#                     document.getElementById('selectionMethod').value = 'contribution';
#                     document.getElementById('pointSize').value = 8;
#                     document.getElementById('labelSize').value = 10;
#                     document.getElementById('showCircle').checked = true;
#                     document.getElementById('showGrid').checked = true;
#                     document.getElementById('showArrows').checked = true;
#                     document.getElementById('colorByContrib').checked = false;
                    
#                     updateNumVars(15);
#                     updatePointSize(8);
#                     updateLabelSize(10);
#                     updateVisualization();
#                 }
                
#                 // Initialisation
#                 document.addEventListener('DOMContentLoaded', function() {
#                     generateSimulatedLoadings();
#                     updateVisualization();
#                 });
#             </script>
#         </body>
#         </html>
# """

#     with open(output_filename, 'w', encoding='utf-8') as f:
#         f.write(html_content)
#     print(f"Rapport HTML sauvegardé dans '{output_filename}' ✅")

