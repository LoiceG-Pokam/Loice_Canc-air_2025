import json
import os
from pathlib import Path

def generate_interactive_pca_html(pca_processor, output_file="pca_interactive.html"):
    """
    Génère un fichier HTML interactif avec les vraies données de PCA.
    
    Args:
        pca_processor: Instance de votre classe PCAProcessor avec PCA calculée
        output_file: Nom du fichier HTML à générer
    """
    
    if pca_processor.pca is None:
        raise ValueError("PCA non calculée. Exécutez d'abord run_acp_complet()")
    
    # Extraire les données
    variables = pca_processor.variables
    loadings = pca_processor.pca.components_.T.tolist()  # Transpose pour avoir [variable][composante]
    explained_variance = pca_processor.pca.explained_variance_ratio_.tolist()
    
    # Template HTML (version simplifiée du fichier interactif)
    html_template = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse ACP Interactive - Données Réelles</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f8f9fa;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }}
        
        .sidebar {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: fit-content;
            position: sticky;
            top: 20px;
        }}
        
        .main-content {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .control-section {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .control-section:last-child {{
            border-bottom: none;
        }}
        
        .control-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        
        .control-item {{
            margin-bottom: 15px;
        }}
        
        .control-item label {{
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #495057;
        }}
        
        .control-item input, .control-item select {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .range-display {{
            text-align: center;
            font-weight: bold;
            color: #3498db;
            margin-top: 5px;
        }}
        
        .checkbox-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .checkbox-item input[type="checkbox"] {{
            width: auto;
        }}
        
        .plot-container {{
            padding: 20px;
        }}
        
        .stats-bar {{
            background: #2c3e50;
            color: white;
            padding: 15px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            text-align: center;
        }}
        
        .stat-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        
        .btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
            transition: background 0.3s;
        }}
        
        .btn:hover {{
            background: #2980b9;
        }}
        
        .component-selector {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .info-panel {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 0 4px 4px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analyse en Composantes Principales - Vos Données</h1>
        <p>Interface interactive pour l'exploration des résultats</p>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="control-section">
                <div class="control-title">Composantes à visualiser</div>
                <div class="component-selector">
                    <div class="control-item">
                        <label>Axe X</label>
                        <select id="pcX" onchange="updateDisplay()">
                            {generate_pc_options(len(explained_variance))}
                        </select>
                    </div>
                    <div class="control-item">
                        <label>Axe Y</label>
                        <select id="pcY" onchange="updateDisplay()">
                            {generate_pc_options(len(explained_variance), selected_index=1)}
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="control-section">
                <div class="control-title">Sélection des variables</div>
                <div class="control-item">
                    <label>Nombre de variables</label>
                    <input type="range" id="numVars" min="5" max="{len(variables)}" value="15" oninput="updateDisplay()">
                    <div class="range-display" id="numVarsDisplay">15 variables</div>
                </div>
                <div class="control-item">
                    <label>Méthode de sélection</label>
                    <select id="selectionMethod" onchange="updateDisplay()">
                        <option value="contribution">Contribution</option>
                        <option value="cos2">Qualité (cos²)</option>
                        <option value="loading">Corrélation abs</option>
                    </select>
                </div>
            </div>
            
            <div class="control-section">
                <div class="control-title">Apparence</div>
                <div class="control-item">
                    <label>Taille des points</label>
                    <input type="range" id="pointSize" min="6" max="20" value="10" oninput="updateDisplay()">
                    <div class="range-display" id="pointSizeDisplay">10px</div>
                </div>
                <div class="control-item">
                    <label>Taille des étiquettes</label>
                    <input type="range" id="labelSize" min="8" max="16" value="11" oninput="updateDisplay()">
                    <div class="range-display" id="labelSizeDisplay">11px</div>
                </div>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="showCircle" checked onchange="updateDisplay()">
                        <label for="showCircle">Cercle unité</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="showArrows" checked onchange="updateDisplay()">
                        <label for="showArrows">Flèches</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="colorByContrib" onchange="updateDisplay()">
                        <label for="colorByContrib">Colorier par contribution</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="showGrid" checked onchange="updateDisplay()">
                        <label for="showGrid">Grille</label>
                    </div>
                </div>
            </div>
            
            <div class="control-section">
                <div class="control-title">Actions</div>
                <button class="btn" onclick="resetSettings()">Reset</button>
                <button class="btn" onclick="downloadPlot('png')">PNG</button>
                <button class="btn" onclick="downloadPlot('svg')">SVG</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value" id="pcXVariance">{explained_variance[0]*100:.1f}%</div>
                    <div class="stat-label">Variance Axe X</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="pcYVariance">{explained_variance[1]*100:.1f}%</div>
                    <div class="stat-label">Variance Axe Y</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="totalVariance">{(explained_variance[0]+explained_variance[1])*100:.1f}%</div>
                    <div class="stat-label">Variance Totale</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="numVarsShown">15</div>
                    <div class="stat-label">Variables</div>
                </div>
            </div>
            
            <div class="plot-container">
                <div class="info-panel">
                    <strong>Interface interactive pour vos résultats PCA</strong><br>
                    Nombre total de variables : {len(variables)}<br>
                    Nombre de composantes : {len(explained_variance)}<br>
                    Variance expliquée cumulée (5 premières PC) : {sum(explained_variance[:5])*100:.1f}%
                </div>
                <div id="pcaPlot" style="width: 100%; height: 600px;"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Données réelles de votre PCA
        const pcaData = {{
            variables: {json.dumps(variables)},
            loadings: {json.dumps(loadings)},
            explained_variance: {json.dumps(explained_variance)}
        }};
        
        function calculateMetrics() {{
            return pcaData.variables.map((variable, i) => {{
                const loadings = pcaData.loadings[i];
                const cos2 = loadings.map(l => l * l);
                const contributions = cos2.map((c, j) => c / pcaData.explained_variance[j]);
                
                return {{
                    index: i,
                    variable: variable,
                    loadings: loadings,
                    cos2: cos2,
                    contributions: contributions,
                    cos2_sum: cos2.reduce((a, b) => a + b, 0),
                    contribution_sum: contributions.reduce((a, b) => a + b, 0),
                    loading_abs_sum: loadings.reduce((a, b) => a + Math.abs(b), 0)
                }};
            }});
        }}
        
        function getTopVariables(method, nVars, pcX, pcY) {{
            const metrics = calculateMetrics();
            
            let sortKey;
            switch(method) {{
                case 'contribution':
                    sortKey = (m) => m.contributions[pcX] + m.contributions[pcY];
                    break;
                case 'cos2':
                    sortKey = (m) => m.cos2[pcX] + m.cos2[pcY];
                    break;
                case 'loading':
                    sortKey = (m) => Math.abs(m.loadings[pcX]) + Math.abs(m.loadings[pcY]);
                    break;
                default:
                    sortKey = (m) => m.contribution_sum;
            }}
            
            return metrics
                .sort((a, b) => sortKey(b) - sortKey(a))
                .slice(0, nVars);
        }}
        
        function updateDisplay() {{
            const numVars = parseInt(document.getElementById('numVars').value);
            const method = document.getElementById('selectionMethod').value;
            const pcX = parseInt(document.getElementById('pcX').value);
            const pcY = parseInt(document.getElementById('pcY').value);
            const pointSize = parseInt(document.getElementById('pointSize').value);
            const labelSize = parseInt(document.getElementById('labelSize').value);
            const showCircle = document.getElementById('showCircle').checked;
            const showArrows = document.getElementById('showArrows').checked;
            const showGrid = document.getElementById('showGrid').checked;
            const colorByContrib = document.getElementById('colorByContrib').checked;
            
            // Mettre à jour les affichages
            document.getElementById('numVarsDisplay').textContent = `${{numVars}} variables`;
            document.getElementById('pointSizeDisplay').textContent = `${{pointSize}}px`;
            document.getElementById('labelSizeDisplay').textContent = `${{labelSize}}px`;
            
            const topVars = getTopVariables(method, numVars, pcX, pcY);
            
            // Créer le graphique
            const traces = [];
            
            // Cercle unité
            if (showCircle) {{
                const theta = Array.from({{length: 100}}, (_, i) => i * 2 * Math.PI / 99);
                traces.push({{
                    x: theta.map(t => Math.cos(t)),
                    y: theta.map(t => Math.sin(t)),
                    mode: 'lines',
                    line: {{color: 'rgba(0,100,200,0.4)', width: 2, dash: 'dash'}},
                    name: 'Cercle Unité',
                    showlegend: false,
                    hoverinfo: 'skip'
                }});
            }}
            
            // Flèches
            if (showArrows) {{
                topVars.forEach(v => {{
                    traces.push({{
                        x: [0, v.loadings[pcX]],
                        y: [0, v.loadings[pcY]],
                        mode: 'lines',
                        line: {{color: 'rgba(0,0,0,0.6)', width: 2}},
                        showlegend: false,
                        hoverinfo: 'skip'
                    }});
                }});
            }}
            
            // Points et étiquettes
            const colors = colorByContrib ? 
                topVars.map(v => v.contributions[pcX] + v.contributions[pcY]) : 
                Array(topVars.length).fill('#e74c3c');
            
            traces.push({{
                x: topVars.map(v => v.loadings[pcX]),
                y: topVars.map(v => v.loadings[pcY]),
                mode: 'markers+text',
                marker: {{
                    size: pointSize,
                    color: colors,
                    colorscale: 'Viridis',
                    showscale: colorByContrib,
                    colorbar: colorByContrib ? {{
                        title: 'Contribution',
                        titleside: 'right',
                        thickness: 15,
                        len: 0.7
                    }} : undefined,
                    line: {{color: 'white', width: 1}}
                }},
                text: topVars.map(v => v.variable),
                textposition: 'top center',
                textfont: {{size: labelSize, color: '#2c3e50', family: 'Segoe UI'}},
                name: 'Variables',
                showlegend: false,
                hovertemplate: 
                    '<b>%{{text}}</b><br>' +
                    `PC${{pcX + 1}}: %{{x:.3f}}<br>` +
                    `PC${{pcY + 1}}: %{{y:.3f}}<br>` +
                    'Contribution: %{{marker.color:.3f}}<br>' +
                    '<extra></extra>'
            }});
            
            const layout = {{
                title: {{
                    text: `Cercle des Corrélations - PC${{pcX + 1}} vs PC${{pcY + 1}}`,
                    font: {{size: 18, color: '#2c3e50', family: 'Segoe UI'}}
                }},
                xaxis: {{
                    title: `PC${{pcX + 1}} (${{(pcaData.explained_variance[pcX] * 100).toFixed(1)}}%)`,
                    range: [-1.2, 1.2],
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: '#95a5a6',
                    showgrid: showGrid,
                    gridcolor: 'rgba(149, 165, 166, 0.3)'
                }},
                yaxis: {{
                    title: `PC${{pcY + 1}} (${{(pcaData.explained_variance[pcY] * 100).toFixed(1)}}%)`,
                    range: [-1.2, 1.2],
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: '#95a5a6',
                    showgrid: showGrid,
                    gridcolor: 'rgba(149, 165, 166, 0.3)'
                }},
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: {{family: 'Segoe UI'}},
                hovermode: 'closest',
                margin: {{l: 60, r: 60, t: 80, b: 60}}
            }};
            
            const config = {{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                toImageButtonOptions: {{
                    format: 'png',
                    filename: `pca_PC${{pcX+1}}_PC${{pcY+1}}`,
                    height: 600,
                    width: 900,
                    scale: 2
                }}
            }};
            
            Plotly.newPlot('pcaPlot', traces, layout, config);
            
            // Mettre à jour les statistiques
            document.getElementById('pcXVariance').textContent = 
                `${{(pcaData.explained_variance[pcX] * 100).toFixed(1)}}%`;
            document.getElementById('pcYVariance').textContent = 
                `${{(pcaData.explained_variance[pcY] * 100).toFixed(1)}}%`;
            document.getElementById('totalVariance').textContent = 
                `${{((pcaData.explained_variance[pcX] + pcaData.explained_variance[pcY]) * 100).toFixed(1)}}%`;
            document.getElementById('numVarsShown').textContent = numVars;
        }}
        
        function resetSettings() {{
            document.getElementById('numVars').value = 15;
            document.getElementById('selectionMethod').value = 'contribution';
            document.getElementById('pcX').value = 0;
            document.getElementById('pcY').value = 1;
            document.getElementById('pointSize').value = 10;
            document.getElementById('labelSize').value = 11;
            document.getElementById('showCircle').checked = true;
            document.getElementById('showArrows').checked = true;
            document.getElementById('showGrid').checked = true;
            document.getElementById('colorByContrib').checked = false;
            
            updateDisplay();
        }}
        
        function downloadPlot(format) {{
            const pcX = parseInt(document.getElementById('pcX').value);
            const pcY = parseInt(document.getElementById('pcY').value);
            const filename = `pca_PC${{pcX+1}}_PC${{pcY+1}}.${{format}}`;
            
            Plotly.downloadImage('pcaPlot', {{
                format: format,
                filename: filename,
                width: 1200,
                height: 800,
                scale: 2
            }});
        }}
        
        // Initialisation
        document.addEventListener('DOMContentLoaded', function() {{
            updateDisplay();
        }});
    </script>
</body>
</html>"""
    
    # Sauvegarder le fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Fichier HTML interactif généré : {output_file}")
    print(f"Ouvrez le fichier dans votre navigateur pour utiliser l'interface interactive.")
    
    return output_file

def generate_pc_options(n_components, selected_index=0):
    """Génère les options HTML pour les sélecteurs de composantes principales."""
    options = []
    for i in range(n_components):
        selected = "selected" if i == selected_index else ""
        options.append(f'<option value="{i}" {selected}>PC{i+1}</option>')
    return "\n".join(options)

# Fonction d'utilisation pour votre classe PCAProcessor
def PCAProcessor_add_html_generation(self):
    """
    Ajoute la méthode de génération HTML à votre classe PCAProcessor.
    À ajouter à votre classe existante.
    """
    def generate_interactive_html(self, filename="pca_interactive.html"):
        """
        Génère un fichier HTML interactif avec les données de PCA.
        """
        return generate_interactive_pca_html(self, filename)
    
    # Ajouter la méthode à la classe
    self.generate_interactive_html = generate_interactive_html.__get__(self, type(self))

# Exemple d'utilisation
def example_usage():
    """
    Exemple d'utilisation avec votre workflow existant
    """
    # Votre code existant...
    # pca_processor = PCAProcessor(df, variables)
    # df_pca, pca_model = pca_processor.run_acp_complet()
    
    # Générer les assets pour le rapport
    # pca_processor.generate_report_assets()
    
    # Générer l'HTML interactif
    # generate_interactive_pca_html(pca_processor, "mon_analyse_pca.html")
    
    print("""
    Pour utiliser ces nouvelles fonctionnalités :
    
    1. Pour le rapport (visualisations statiques) :
       pca_processor.generate_report_assets()
    
    2. Pour l'interface interactive :
       generate_interactive_pca_html(pca_processor, "analyse_interactive.html")
    
    3. Intégration dans votre classe :
       # Ajoutez les nouvelles méthodes à votre classe PCAProcessor
       # puis utilisez : pca_processor.generate_interactive_html()
    """)
