import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Appariement Cas-Témoins 3D", layout="wide")

# ---------------------------
# Chargement des données
# ---------------------------
st.title("🔬 Visualisation 3D de l'appariement cas-témoins")

uploaded_file = st.file_uploader("📂 Charger le fichier contenant les appariements", type=["csv", "xlsx", "parquet"])

if uploaded_file is not None:
    # Choix du format
    if uploaded_file.name.endswith(".csv"):
        df_matched = pd.read_csv(uploaded_file, sep=';')
    elif uploaded_file.name.endswith(".xlsx"):
        df_matched = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".parquet"):
        df_matched = pd.read_parquet(uploaded_file)
    else:
        st.error("Format non supporté. Merci de charger un CSV, Excel ou Parquet.")
        st.stop()

    # Affichage des colonnes disponibles pour debug
    st.write("Colonnes disponibles:", df_matched.columns.tolist())
    
    # Vérification des colonnes nécessaires (les 3 facteurs confondants + colonnes de structure)
    required_cols = ['pseudo_provisoire', 'is_case', 'age_a_letude', 'patient_sexe', 'patho', 'cas_apparie']
    missing_cols = [col for col in required_cols if col not in df_matched.columns]
    
    if missing_cols:
        st.error(f"Colonnes manquantes: {missing_cols}")
        st.stop()

    # ---------------------------
    # Préparation des données pour la 3D
    # ---------------------------
    cases = df_matched[df_matched["is_case"] == True]
    controls = df_matched[df_matched["is_case"] == False]

    # Vérification qu'on a bien des cas et des témoins
    if len(cases) == 0:
        st.error("Aucun cas trouvé dans les données (is_case == True)")
        st.stop()
    
    if len(controls) == 0:
        st.error("Aucun témoin trouvé dans les données (is_case == False)")
        st.stop()

    # Vérification que la colonne 'patho' existe
    if 'patho' not in df_matched.columns:
        st.error("La colonne 'patho' est manquante dans les données")
        st.stop()
    
    # Normalisation pour les coordonnées 3D (les 3 facteurs confondants)
    
    # X = Âge (normalisé)
    age_min = df_matched["age_a_letude"].min()
    age_max = df_matched["age_a_letude"].max()
    
    if age_max > age_min:
        df_matched["ageNorm"] = (df_matched["age_a_letude"] - age_min) / (age_max - age_min)
    else:
        df_matched["ageNorm"] = 0.5  # valeur par défaut si tous les âges sont identiques
    
    df_matched["x"] = df_matched["ageNorm"] * 200 - 100
    
    # Y = Pathologie (encodage automatique des valeurs textuelles avec espacement de 5)
    unique_patho = df_matched["patho"].dropna().unique()
    patho_mapping = {patho: (i+1)*5 for i, patho in enumerate(sorted(unique_patho))}
    df_matched["y"] = df_matched["patho"].map(patho_mapping)
    
    # Z = Sexe (patient_sexe: M/F transformé en numérique)
    df_matched["z"] = df_matched["patient_sexe"].apply(
        lambda s: 50 if s == 'M' else -50 if s == 'F' else 0
    )

    # Recalculer après modification
    cases = df_matched[df_matched["is_case"] == True].copy()
    controls = df_matched[df_matched["is_case"] == False].copy()

    # ---------------------------
    # Interface Streamlit
    # ---------------------------
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("⚙️ Options")
        show_all = st.checkbox("Afficher tous les liens", value=False)
        
        # Gérer le cas où cas_apparie peut être NaN pour les témoins
        available_cases = cases["pseudo_provisoire"].dropna().unique()
        selected_case = st.selectbox("Sélectionner un cas", ["Aucun"] + list(available_cases))

        st.subheader("📊 Statistiques")
        
        # Compter les cas qui ont des témoins appariés
        matched_cases = controls["cas_apparie"].nunique() if not controls.empty else 0
        st.metric("Cas appariés", f"{matched_cases}/{len(cases)}")
        st.metric("Total témoins", len(controls))
        st.metric("Appariements", len(controls))

        if not controls.empty and not controls["cas_apparie"].isna().all():
            avg_controls = controls.groupby("cas_apparie").size().mean()
            st.metric("Témoins/cas (moy.)", f"{avg_controls:.1f}")
        else:
            st.metric("Témoins/cas (moy.)", "N/A")
            
        # Afficher l'encodage des pathologies
        st.subheader("🔢 Encodage Pathologies")
        patho_df = pd.DataFrame(list(patho_mapping.items()), columns=["Pathologie", "Code"])
        st.dataframe(patho_df, use_container_width=True)

    with col1:
        fig = go.Figure()

        # Cas (sphères bleues)
        fig.add_trace(go.Scatter3d(
            x=cases["x"], 
            y=cases["y"], 
            z=cases["z"],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="circle"),
            text=cases["pseudo_provisoire"].astype(str),
            hovertemplate="Cas: %{text}<br>Âge: %{customdata[0]}<br>Sexe: %{customdata[1]}<br>Patho: %{customdata[2]}<extra></extra>",
            customdata=list(zip(cases["age_a_letude"], cases["patient_sexe"], cases["patho"])),
            name="Cas"
        ))

        # Témoins (cubes roses)
        fig.add_trace(go.Scatter3d(
            x=controls["x"], 
            y=controls["y"], 
            z=controls["z"],
            mode="markers",
            marker=dict(size=6, color="pink", symbol="square"),
            text=controls["pseudo_provisoire"].astype(str),
            hovertemplate="Témoin: %{text}<br>Âge: %{customdata[0]}<br>Sexe: %{customdata[1]}<br>Patho: %{customdata[2]}<br>Cas apparié: %{customdata[3]}<extra></extra>",
            customdata=list(zip(controls["age_a_letude"], controls["patient_sexe"], controls["patho"], controls["cas_apparie"].astype(str))),
            name="Témoins"
        ))

        # Connexions
        if show_all or selected_case != "Aucun":
            # Filtrer les témoins selon la sélection
            if show_all:
                relevant_controls = controls[controls["cas_apparie"].notna()]
            else:
                relevant_controls = controls[controls["cas_apparie"] == selected_case]
            
            for _, control_row in relevant_controls.iterrows():
                case_pseudo = control_row["cas_apparie"]
                case_matches = cases[cases["pseudo_provisoire"] == case_pseudo]
                
                if not case_matches.empty:
                    case_pt = case_matches.iloc[0]
                    color = "red" if case_pseudo == selected_case else "orange"
                    width = 3 if case_pseudo == selected_case else 1
                    
                    fig.add_trace(go.Scatter3d(
                        x=[case_pt["x"], control_row["x"]],
                        y=[case_pt["y"], control_row["y"]],
                        z=[case_pt["z"], control_row["z"]],
                        mode="lines",
                        line=dict(color=color, width=width),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Âge (normalisé)",
                yaxis_title="Pathologie (encodée)",
                zaxis_title="Sexe (M=50, F=-50)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            title="Visualisation 3D des appariements cas-témoins<br><sub>Axes: Âge (X), Pathologie encodée (Y), Sexe (Z)</sub>"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Cas sélectionné : détails
    # ---------------------------
    if selected_case != "Aucun":
        st.subheader(f"🔎 Détails pour le cas {selected_case}")
        
        case_matches = cases[cases["pseudo_provisoire"] == selected_case]
        if not case_matches.empty:
            case_data = case_matches.iloc[0]
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.write(f"**Âge à l'étude :** {case_data['age_a_letude']:.0f} ans")
            with col_b:
                st.write(f"**Sexe :** {case_data['patient_sexe']}")
            with col_c:
                st.write(f"**Pathologie :** {case_data['patho']}")
            with col_d:
                if "date_reference" in case_data and pd.notna(case_data["date_reference"]):
                    st.write(f"**Date réf. :** {case_data['date_reference']}")

            # Témoins appariés
            matched_controls = controls[controls["cas_apparie"] == selected_case]
            if not matched_controls.empty:
                st.write(f"**Témoins appariés :** {len(matched_controls)}")
                
                # Tableau des témoins avec les 3 facteurs confondants
                controls_display = matched_controls[["pseudo_provisoire", "age_a_letude", "patient_sexe", "patho"]].copy()
                controls_display.columns = ["Pseudo", "Âge", "Sexe", "Patho"]
                st.dataframe(controls_display, use_container_width=True)
            else:
                st.write("**Aucun témoin apparié trouvé**")

    # ---------------------------
    # Aperçu des données
    # ---------------------------
    if st.checkbox("Afficher un aperçu des données"):
        st.subheader("📋 Aperçu des données")
        st.write("Premières lignes du fichier:")
        st.dataframe(df_matched.head(10))

else:
    st.info("""
    ➡️ Merci de charger un fichier CSV/XLSX/Parquet contenant les colonnes nécessaires :
    
    **Colonnes requises :**
    - `pseudo_provisoire` : identifiant unique
    - `is_case` : True pour les cas, False pour les témoins  
    - `age_a_letude` : âge du patient (facteur confondant 1)
    - `patient_sexe` : sexe du patient M/F (facteur confondant 2)
    - `patho` : pathologie (facteur confondant 3)
    - `cas_apparie` : pseudo du cas apparié (pour les témoins)
    
    **Visualisation 3D :**
    - Axe X : Âge (normalisé)
    - Axe Y : Pathologie (encodée automatiquement : première=5, deuxième=10, etc.)
    - Axe Z : Sexe (M=+50, F=-50)
    
    **Colonnes optionnelles :**
    - `date_reference` : date de référence
    """)