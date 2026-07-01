import numpy as np
import re
import pandas as pd
import re
from unidecode import unidecode


# replace cudf to pd
def NWmatrix2strings(S,T,D):
    """
    Needleman-Wunch matrix to aligned strings

    Args:
        S (str): first string
        T (str): second string
        D (np.array or list of lists): Decision matrix

    Returns:
        S_aligned (str): first string aligned
        T_aligned (str): second string aligned
    """
    S_WITH_GAP = -1
    T_WITH_GAP = 1
    S_T_ALIGN = 0
    i = D.shape[0]-1
    j = D.shape[1]-1
    S_aligned = ""
    T_aligned = ""
    
    while i > 0 or j > 0:
        if D[i,j] == S_WITH_GAP:
            S_aligned = S[i-1]+S_aligned
            T_aligned = "-"+T_aligned
            i -= 1
        elif D[i,j] == T_WITH_GAP:
            T_aligned = T[j-1]+ T_aligned
            S_aligned = "-"+S_aligned
            j -= 1
        elif D[i,j] == S_T_ALIGN:
            S_aligned = S[i-1] + S_aligned
            T_aligned = T[j-1] + T_aligned
            j -= 1
            i -= 1
    return(S_aligned,T_aligned)
           
    
def Needleman_Wunch_update(S,T,return_all=False, return_unmatched=True):
    """
    Algorithme de Needleman-Wunsch pour calculer la matrice de score et de décision.

    Args:
        S (str): premier string
        T (str): second string
        return_matrix (bool): indique si la fonction doit retourner les matrices F et D.

    Returns:
        F (np.array): Matrice de score
        D (np.array): Matrice de décision
        alignment_score (int): Score global d'alignement (coin inférieur droit de F)
    """
    S = S.upper()
    T = T.upper()
    
    # Définition des score
    T_WITH_GAP = 1
    S_WITH_GAP = -1
    S_T_ALIGN = 0
    decision = [T_WITH_GAP,S_WITH_GAP,S_T_ALIGN]
    
    gap_score = -1
    substitution_score = -10
    
    # Initialisation des matrices F (score) et D (décision)
    F = np.zeros((len(S)+1,len(T)+1))
    F[0,:] = range(len(T)+1)
    F[:,0] = range(len(S)+1)
    F *= gap_score ## gap score 
    D = np.zeros((len(S)+1,len(T)+1))
    D[0,1:] = T_WITH_GAP 
    D[1:,0] = S_WITH_GAP 
    
    for i in range(1,len(S)+1):
        for j in range(1,len(T)+1):
            comparison = 1 if (S[i-1]==T[j-1]) else substitution_score # = int(S[i-1]==T[j-1])*2-1 # * 10 
            options = [F[i,j-1] + gap_score , F[i-1,j] + gap_score , F[i-1,j-1] + comparison]
            F[i,j] = options[0]
            D[i,j] = T_WITH_GAP
            for o,val in enumerate(options):
                if val > F[i,j]:
                    F[i,j] = val
                    D[i,j] = decision[o]
                    
    ### Récupération des châines alignées 
                    
    S_aligned, T_aligned = NWmatrix2strings(S, T, D)

    unmatched_S, unmatched_T = get_unmatched_segments(S,T,S_aligned, T_aligned)


    # Calcul du score d'alignement par caractères
    matched_chars = sum(1 for s_char, t_char in zip(S_aligned, T_aligned) if s_char == t_char and s_char != '-')
    total_chars = max(len(S), len(T))
    char_alignment_score = matched_chars / total_chars * 100  # Pourcentage de caractères alignés

    # Calcul du score d'alignement par tokens (mots)

    matched_tokens = [token for token in S_aligned.split(' ') if token not in unmatched_S]
    if '' in matched_tokens : 
        matched_tokens.remove('')
    total_tokens = max(len(S_aligned.split(' ')), len(T_aligned.split(' ')))
    token_alignment_score = len(matched_tokens) / total_tokens * 100  # Pourcentage de tokens alignés
    
    if return_all : 
        return char_alignment_score, token_alignment_score, S_aligned, T_aligned, unmatched_S, unmatched_T
    
    if return_unmatched :
        return char_alignment_score, token_alignment_score, unmatched_S, unmatched_T

    
def extract_words_from_chars(chars):
    words = []
    current_word = []

    for char in chars:
        if char != ' ':  # Ignore les espaces
            current_word.append(char)
        else:
            # Si on rencontre un espace, ajoute le mot en cours (s'il y en a un) et réinitialise
            if current_word:
                words.append(''.join(current_word))
                current_word = []

    # Ajoute le dernier mot si la liste ne se termine pas par un espace
    if current_word:
        words.append(''.join(current_word))

    return words


def get_word_at_position(text, position):
    
    if position == -1 or pd.isna(text):
        return ""
    
    words = text.split()  # Divise la chaîne en une liste de mots
    current_pos = 0  # Position courante pour suivre l'index des lettres

    for word in words:
        word_length = len(word)

        # Vérifie si la position donnée se trouve dans le mot actuel
        if current_pos <= position < (current_pos + word_length):
            return word
        current_pos += word_length + 1  # Passe au mot suivant en tenant compte de l'espace

    return ""

def split_to_words(segment):
# Utiliser une expression régulière pour séparer en mots tout en gardant la ponctuation
    return [word.strip() for word in re.findall(r'\b\w+\b', segment)]


def get_unmatched_segments(S,T,S_aligned, T_aligned):
    """
    Retourne les segments non appariés entre deux strings alignés.

    Args:
        S_aligned (str): premier string aligné
        T_aligned (str): second string aligné

    Returns:
        unmatched_S (list): segments non appariés de S
        unmatched_T (list): segments non appariés de T
    """
    
    S = S.upper()
    T = T.upper()
    
    S_words = S.split(' ')
    T_words = T.split(' ')  
    

    
    unmatched_S = []
    unmatched_T = []
    
    matched_S = []
    
    current_unmatched_S = ""
    current_unmatched_T = ""
    
    filtered_S_words = []

    for s_char, t_char in zip(S_aligned, T_aligned):
        if s_char == '-' and t_char != '-':
            # Gap dans S, donc caractère non apparié dans T
            current_unmatched_T += t_char
            if current_unmatched_S:
                unmatched_S.append(current_unmatched_S)
                current_unmatched_S = ""
        elif t_char == '-' and s_char != '-':
            # Gap dans T, donc caractère non apparié dans S
            current_unmatched_S += s_char
            if current_unmatched_T:
                unmatched_T.append(current_unmatched_T)
                current_unmatched_T = ""
        else:
            matched_S.append(s_char)

            # Les deux caractères sont appariés
            if current_unmatched_S:
                unmatched_S.append(current_unmatched_S)
                current_unmatched_S = ""
            if current_unmatched_T:
                unmatched_T.append(current_unmatched_T)
                current_unmatched_T = ""

    # Ajouter les segments non appariés restants
    if current_unmatched_S:
        unmatched_S.append(current_unmatched_S)
    if current_unmatched_T:
        unmatched_T.append(current_unmatched_T)


    unmatched_S_words = []
    unmatched_T_words = []
    
    for segment in unmatched_S:
        unmatched_S_words.extend(split_to_words(segment))

    for segment in unmatched_T:
        unmatched_T_words.extend(split_to_words(segment))
        
    matched_S_cleaned = extract_words_from_chars(matched_S)

    filtered_matched = [word for word in matched_S_cleaned if (word in S_words) or (word in T_words)]
    filtered_matched_uncleaned = [word for word in matched_S_cleaned if word not in filtered_matched]
    
    filtered_S_words = [get_word_at_position(S, S.find(word)) for word in unmatched_S_words if S.find(word) != -1]
    filtered_T_words = [get_word_at_position(T, T.find(word)) for word in unmatched_T_words if T.find(word) != -1]
    for i in filtered_matched_uncleaned : 
        position = S.find(i)
        if position != -1:
            cleaned_word = get_word_at_position(S, position)
            if cleaned_word not in filtered_S_words : 
                filtered_S_words.append(cleaned_word)


    return list(dict.fromkeys(filtered_S_words)), list(dict.fromkeys(filtered_T_words))

def calculate_alignment_metrics(S_aligned, T_aligned):
    """
    Calcule les pourcentages et le score global d'alignement en fonction des identités,
    substitutions, et gaps dans l'alignement final.

    Args:
        S_aligned (str): premier string aligné
        T_aligned (str): second string aligné

    Returns:
        dict: Contient % identité, % gaps, % substitutions, et le score d'alignement
    """

    identity_score = 3
    substitution_score = -1
    gap_score = -2
    
    identities = substitutions = gaps = 0
    alignment_length = len(S_aligned)

    for s_char, t_char in zip(S_aligned, T_aligned):
        if s_char == t_char and s_char != "-":
            identities += 1
        elif s_char == "-" or t_char == "-":
            gaps += 1
        else:
            substitutions += 1

    # Calcul des pourcentages
    percent_identity = (identities / alignment_length) * 100
    percent_gaps = (gaps / alignment_length) * 100
    percent_substitutions = (substitutions / alignment_length) * 100

    # Calcul du score d'alignement
    alignment_score = (identities * identity_score) + (substitutions * substitution_score) + (gaps * gap_score)

    return {
        "percent_identity": percent_identity,
        "percent_gaps": percent_gaps,
        "percent_substitutions": percent_substitutions,
        "alignment_score": alignment_score
    }

def remplacer_types_de_voies(df, colonne, df_odonyme):
    """
    Remplace les types de voies dans une colonne donnée d'un DataFrame, en utilisant 
    les synonymes et termes correspondants d'un autre DataFrame (df_odonyme).

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne à modifier.
        colonne (str): Nom de la colonne où effectuer les remplacements.
        df_odonyme (pd.DataFrame): DataFrame contenant les colonnes 'synonym' et 'termn'.

    Returns:
        pd.DataFrame: DataFrame modifié avec les remplacements effectués.
    """
    dataframe = df.copy()

    # Trier df_odonyme par la longueur décroissante des synonymes
    df_odonyme = df_odonyme.assign(synonym_length=df_odonyme['synonym'].str.len())
    df_odonyme = df_odonyme.sort_values(by='synonym_length', ascending=False)

    # Effectuer les remplacements pour chaque synonyme dans l'ordre trié
    for _, row in df_odonyme.iterrows():
        synonym = row['synonym'].upper()
        replacement = row['terme'].upper() + ' '

        # Utiliser str.replace pour les remplacements avec correspondance stricte
        dataframe[colonne] = dataframe[colonne].str.replace(
            rf'\b{synonym}\b', replacement, regex=True
        )
        
    return dataframe

def normalisation_adresse(dataframe,column): 

    df = dataframe.copy()

    df[column] = df[column].str.upper()
    df[column] = df[column].apply(unidecode)
    df[column] = df[column].str.replace('.',' ')
    df[column] = df[column].str.replace(r'[\n\r]+', ' ', regex=True)
    
    
    
    df[column] = df[column].str.replace(rf'\bBOULEVARDDE\b','BOULEVARD DE')
    df[column] = df[column].str.replace(rf'\bBOULEVARDDU\b','BOULEVARD DU')
    df[column] = df[column].str.replace(rf'\bBOULEVARDDES\b','BOULEVARD DES')
    
    

    df[column] = df[column].str.replace(rf'\bDITALIE\b','D ITALIE')
    df[column] = df[column].str.replace(rf'\bDALESIA\b','D ALESIA')
    df[column] = df[column].str.replace(rf'\bLASSOMPTION\b','L ASSOMPTION')
    df[column] = df[column].str.replace(rf'\bDALLERAY\b','D ALLERAY')
    df[column] = df[column].str.replace(rf'\bLINGENIEUR\b','L INGENIEUR')
    df[column] = df[column].str.replace(rf'\bDAUBERVILLIERS\b','D AUBERVILLIERS')
    
    df[column] = df[column].str.replace(rf'\bCHAMPS\b','CHAMP')
    df[column] = df[column].str.replace(rf'\bGUTEMBERG\b','GUTENBERG')
    df[column] = df[column].str.replace(rf'\bLONGCHAMPS\b','LONGCHAMP')
    df[column] = df[column].str.replace(rf'\bCOLONNEL\b','COLONEL')
    df[column] = df[column].str.replace(rf'\bFLANDRES\b','FLANDRE')
    df[column] = df[column].str.replace(rf'\bACCACIAS\b','ACACIAS')
    df[column] = df[column].str.replace(rf'\bMOCQUET\b','MOQUET')
    df[column] = df[column].str.replace(rf'\bGAMMA\b','GAMA')
    df[column] = df[column].str.replace(rf'\bBRIANT\b','BRIAND')
    df[column] = df[column].str.replace(rf'\bALFONSE\b','ALPHONSE')
    df[column] = df[column].str.replace(rf'\bDASSAS\b','D ASSAS')
    df[column] = df[column].str.replace(rf'\bESTIENNES\b','ESTIENNE')
    df[column] = df[column].str.replace(rf'\bALLIENDE\b','ALLENDE')
    df[column] = df[column].str.replace(rf'\bHAL\b','HOPITAL')
    df[column] = df[column].str.replace(rf'\bSERRURIER\b','SERURIER')
    df[column] = df[column].str.replace(rf'\bRUIE\b','RUE')
    df[column] = df[column].str.replace(rf'\bRUR\b','RUR')
    df[column] = df[column].str.replace(rf'\bPAS\b','PASSAGE')
    df[column] = df[column].str.replace(rf'\bBLS\b','BOULEVARD')
    df[column] = df[column].str.replace(rf'\bSCE\b','SERVICE')
    df[column] = df[column].str.replace(rf'\bCOL\b','COLONEL')
    df[column] = df[column].str.replace(rf'\bGL\b','GENERAL')
    df[column] = df[column].str.replace(rf'\bGLE\b','GENERAL')
    df[column] = df[column].str.replace(rf'\bDR\b','DOCTEUR')
    df[column] = df[column].str.replace(rf'\bSTE\b','SAINTE')
    df[column] = df[column].str.replace(rf'\bST\b','SAINT')
    df[column] = df[column].str.replace(rf'\bPTE\b','PORTE')
    df[column] = df[column].str.replace(rf'\bHOP\b','HOPITAL')

    ###Bruits cas particuliers 
    df[column] = df[column].str.replace(rf'\bME\b','MME')
    df[column] = df[column].str.replace(rf'\bAPP\b','APPT')
    df[column] = df[column].str.replace(rf'\bAPT\b','APPT')
    df[column] = df[column].str.replace(rf'\bAPPART\b','APPT')
    df[column] = df[column].str.replace(rf'\bINCONNUE\b','INCONNU')
    df[column] = df[column].str.replace(rf'\bCZ\b','CHEZ')
    df[column] = df[column].str.replace(rf'\bBT\b','BATIMENT')
    df[column] = df[column].str.replace(rf'\bHAL\b','HOPITAL')
    

    
    # df[column] = df[column].str.replace(rf'\bRESIDENCE\b',' ', regex=True)
    # df[column] = df[column].str.replace(rf'\bLA FORET\b',' ', regex=True)
    # df[column] = df[column].str.replace(rf'\bFORET\b',' ', regex=True)
    # df[column] = df[column].str.replace(rf'\bBD\b','BOULEVARD')
    # df[column].str.replace(r'\b\d+\s+RESIDENCE\b', ' ', regex=True)
    # df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
    

# 2. NETTOYAGE DES COMPLÉMENTS OPTIMISÉ
    
    NUM_VOIE = r'\d+(?:\s*(?:BIS|TER|QUATER)|\s*[A-D])?'
    VOIES_TYPES = r'(RUE|AVENUE|BOULEVARD|BD|ALLEE|PASSAGE|IMPASSE|PLACE|CHEMIN|ROUTE|QUAI|COUR|COURS|CITE|PARVIS|VILLA|PROMENADE)'
    BRUITS_COMPLEMENTS = (
    r'(RESIDENCE|FORET|LA FORET|BATIMENT|BAT|BT|'
    r'APPT|APPART|APPARTEMENT|'
    r'PORTE|ESCALIER|ESC|ETAGE|ETG|'
    r'HALL|LOGEMENT|BUREAU|'
    r'INCONNU|CHEZ|CZ|MME|MR|MADAME|MONSIEUR|MLLE)'
)

   
    df[column] = df[column].str.replace(rf'^.*?\b({NUM_VOIE}\s+{VOIES_TYPES})',r'\1',regex=True)
    df[column] = df[column].str.replace(r'\b(PORTE|ESCALIER|ESC|ETAGE|ETG|HALL)\b.*$', '', regex=True)

    df[column] = df[column].str.replace(r'\b([A-Z])\s*[.\-]\s*([A-Z])\b',r'\1\2',regex=True)
    df[column] = df[column].str.replace(r'\b(PORTE|ESCALIER|ESC|ETAGE|ETG|HALL)\b.*$', '',  regex=True)


    pattern_initial_complement_no_num = r'^\s*\b' + BRUITS_COMPLEMENTS + r'.*?(\s*' + NUM_VOIE + r'\s+' + VOIES_TYPES + r')'
    df[column] = df[column].str.replace(pattern_initial_complement_no_num, r'\1', regex=True)
    pattern_initial_complement = r'^\s*' + NUM_VOIE + r'\s+' + BRUITS_COMPLEMENTS + r'.*?(\s*' + NUM_VOIE + r'\s+' + VOIES_TYPES + r')'
    df[column] = df[column].str.replace(pattern_initial_complement, r'\1', regex=True)

    pattern_final_complement = r'\b(' + BRUITS_COMPLEMENTS + r').*$'
    df[column] = df[column].str.replace(pattern_final_complement, ' ', regex=True)
    df[column] = df[column].str.replace(r'\b' + NUM_VOIE + r'\s+' + BRUITS_COMPLEMENTS + r'\b', ' ', regex=True)


    df[column] = df[column].str.replace(rf'\bBD\b','BOULEVARD')
    df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
    df[column] = df[column].str.strip()
    return df


def normalisation_commune(df, column):
    df = df.copy()

    df[column] = (
        df[column]
        .fillna('')
        .astype(str)
        .str.upper()
        .apply(unidecode)
    )

    # Apostrophes & tirets
    df[column] = df[column].str.replace(r"'", ' ', regex=True)
    df[column] = df[column].str.replace(r'-', ' ', regex=True)

    # Abréviations SAINT / SAINTE (standard français)
    df[column] = df[column].str.replace(r'\bST\b', 'SAINT', regex=True)
    df[column] = df[column].str.replace(r'\bSTE\b', 'SAINTE', regex=True)
    df[column] = df[column].str.replace(r'\bSTS\b', 'SAINTS', regex=True)
    df[column] = df[column].str.replace(r'\bSTES\b', 'SAINTES', regex=True)

    # Articles élidés
    df[column] = df[column].str.replace(r'\bD\b', 'DE', regex=True)
    df[column] = df[column].str.replace(r'\bL\b', 'LE', regex=True)

    # Cas fréquents raccourcis
    df[column] = df[column].str.replace(r'\bSUR\b', ' SUR ', regex=True)
    df[column] = df[column].str.replace(r'\bSOUS\b', ' SOUS ', regex=True)
    df[column] = df[column].str.replace(r'\bEN\b', ' EN ', regex=True)
    df[column] = df[column].str.replace(r'\bLES\b', ' LES ', regex=True)

    # Nettoyage final
    df[column] = (
        df[column]
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    return df

def find_most_common_biaises(df,to_clean): 
    
    """
    3. Récupération des éléments non alignés de l'adresse brute et de l'adresse géocodée 
    4. Nettoyage des éléments : 
        - suppression des éléments nuls 
        - suppression des redondances 
        - séparation des numéros et mots 
        - suppression des numéros (-> idée : adresse avec num non alignée = err geocodage ?) 
    """
    not_matched_brute = pd.Series(df['not_matched_brute'].dropna())
    not_matched_geo =  pd.Series(df['not_matched_geo'].dropna())

    not_matched_brute_as_str = [x if isinstance(x, str) else ' '.join(x) for x in not_matched_brute]
    not_matched_geo_as_str = [x if isinstance(x, str) else ' '.join(x) for x in not_matched_geo]

    # not_matched_brute_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_brute]
    # not_matched_geo_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_geo]

    not_matched_brute_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_brute_as_str]
    not_matched_geo_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_geo_as_str]
    
    # list_not_matched_brute_noNum = pd.Series([x.replace('[\d ]+', '') for x in not_matched_brute_noNum if x != ''])
    list_not_matched_brute_noNum = pd.Series([re.sub(r'[\d ]+', '', x) for x in not_matched_brute_noNum if x != ''])
    # list_not_matched_geo_noNum = pd.Series([x.replace('[\d ]+', '') for x in not_matched_geo_noNum if x != ''])
    list_not_matched_geo_noNum = pd.Series([re.sub(r'[\d ]+', '', x) for x in not_matched_geo_noNum if x != ''])
    
    """5. Transformation de la liste des éléments alignées (string) vers des éléments individuels  
    - ensemble des éléments supplémentaires de l'adresse brute : {bruits}  
    - ensemble des éléments supplémentaires de l'adresse géocodée : {complement}
   """ 
    list_bruit_split = list_not_matched_brute_noNum.str.split(' ') 
    list_bruit_exploded = list_bruit_split.explode()
    bruit_count = pd.DataFrame(list_bruit_exploded.value_counts(), columns=['count'],index=pd.Index(list_bruit_exploded).unique())
    
    list_compl_split = list_not_matched_geo_noNum.str.split(' ') 
    list_compl_exploded = list_compl_split.explode()
    compl_count = pd.DataFrame(list_compl_exploded.value_counts(), columns=['count'],index=pd.Index(list_compl_exploded).unique())
    compl_count = compl_count[compl_count['count']>=2]


    """ 6. Filtre des éléments non alignées de l'adresse brutes avec les compléments  
        {bruits filtres} = {bruits} \ {complement}

        ##Probleme - filtre des éléments pertinents tel que "CHEZ" ou autre pouvant être considérés comme biais - retrait manuel 
    """
    common = bruit_count.index.intersection(compl_count.index)


    unwanted_elements = ['HOPITAL', 'CHEZ', 'BATIMENT', 'LOTISSEMENT', 'APPT', 'PREMIER', 'ESC', 'GROUPE' , 'SECOURS', 'HOTEL' , 
                         'CO' ,'DOMICILE','APPARTEMENT','MAISON','ESCALIER' ]
    
    common = common.difference(unwanted_elements)
    bruits_filtres = bruit_count.drop(common)

    
    """ 7. Filtre des éléments non alignées de l'adresse brutes avec les noms propres   
        {bruits vrais} = {bruits filtres} \ ( {prénoms} U {voies} ) """
    
    common = bruits_filtres.index.intersection(to_clean)
    bruits_vrais = bruits_filtres.drop(common)

    
    """8. Filtre des éléments non alignés de l'adresses brutes comportant des éléments de voiries, issus d'erreur de typographie"""
    pattern = r"\b(AVENUE|RUE|PLACE|IMPASSE|BOULEVARD|ALLEE|SQUARE|ROUTE|ESPLANADE|CHEMIN|GRANDE\sRUE|ROND\sPOINT|FAUBOURG|JARDIN|VIA|GALERIE|VOIE|QUAI|PASSAGE|COUR|COURS|CITE|PARVIS|HAMEAU|VILLA|VILLAGE|VILLE|TERRASSE|PROMENADE|SENTIER|CARREFOUR|CHAUSSEE|DOMAINE|CLOS|MOULIN|CENTRE|MAIL|BOIS|PROMENEE|VALLEE|RESIDENCE|QUARTIER|LOTISSEMENT|TRAVERSE|LIEU\sDIT|FERME|LE\sBOURG|PARC|COTE)[A-Za-z]+"

    bruits_fin = bruits_vrais[~bruits_vrais.index.str.contains(pattern, regex=True)]
    
   
    return bruits_fin.sort_values('count',ascending=False)


def find_pos_elem(df, column,elem,col_elem,col_contains_elem,col_pos_elem,drop_col_elem=True):
    """
    dataframe containing a column in which to search the position of certain elements 
    
    column : column in which to find position of elements 
    elem : Series of element to search 
    col_elem : column in which to stock the corresponding element found
    col_contains_elem : boolean True if one of the element of elem is present in the column 
    col_pos_elem : position of the element in the string 

    """
    dataframe = df.copy()

    dataframe[column] = dataframe[column].str.strip()
    pattern_voies = r'\b(?:' + '|'.join(re.escape(word) for word in elem) + r')\b'
    dataframe[col_contains_elem] = dataframe[column].str.contains(pattern_voies)
    
    dataframe[col_elem]=dataframe.loc[dataframe[col_contains_elem], column].str.extract(f'({pattern_voies})', expand=False)

    dataframe[f'{elem}_tokens'] = dataframe[column].str.strip().str.split(r'\s+')

    ### retrieve position
    df_exploded = dataframe.explode(f'{elem}_tokens')

    # Step 2: Find the index positions where `tokens` matches `col_elem`
    df_exploded['position'] = df_exploded.groupby(level=0).cumcount()  # Generate index per word
    df_exploded['match'] = df_exploded[f'{elem}_tokens'] == df_exploded[col_elem]

    # Step 3: Retrieve position where `match` is True
    df_positions = df_exploded[df_exploded['match']].groupby(level=0)['position'].first()

    # Step 4: Merge back into the original DataFrame
    dataframe[col_pos_elem] = df_positions.reindex(dataframe.index, fill_value=-1)

    if drop_col_elem :
        dataframe = dataframe.drop([col_elem,f'{elem}_tokens'],axis=1)
    if drop_col_elem==False : 
        dataframe = dataframe.drop([f'{elem}_tokens'],axis=1)

    return dataframe 

def freq_couverture(dataframe,bruit,colonne):
    
    df = dataframe[(~dataframe[colonne].isna())&(dataframe[colonne]!="")]

    ##Serie contenant les bruits vrais contenus dans le dataframe : bruit_vrais_w_street 
    pattern_biaises = r'\b(?:' + '|'.join(re.escape(word) for word in bruit) + r')\b'

    ### Recherche des termes dans le dataframe
    df['biaises_found'] = df[colonne].str.findall(pattern_biaises)

    bruit_trouves = df[['id', 'biaises_found']].explode('biaises_found').reset_index()
    
    bruit_trouves.rename(columns={'id':'row_id','biaises_found': 'biais'}, inplace=True)
    bruit_trouves = bruit_trouves.dropna(subset="biais")
    bruit_trouves = bruit_trouves.drop('index',axis=1)

    df_groupby = bruit_trouves.groupby('biais').agg(list)

    df_groupby['list_size'] = df_groupby['row_id'].apply(len)#.list.len()

    df_sorted_gp = df_groupby.sort_values(by='list_size', ascending=True).drop(columns=['list_size'])

    seen_biaises = set()
    cumsum_values = []

    for list_of_row_id in df_sorted_gp['row_id']:
        # Add unique values to the seen set
        seen_biaises.update(list_of_row_id)
        # The cumulative count is the size of the seen set
        cumsum_values.append(len(seen_biaises))

    # Assign the corrected cumsum values to the DataFrame
    df_sorted_gp['cumsum'] = cumsum_values
    
    df['biaises_found'] = df['biaises_found'].astype(str)
    nb_row_w_biais = len(df[df['biaises_found']!="[]"])

    df_sorted_gp['freq_globale_cum'] = df_sorted_gp['cumsum']/ len(dataframe)
    df_sorted_gp['freq_biais_cum'] = df_sorted_gp['cumsum']/ nb_row_w_biais
    
    return df_sorted_gp


def filter_by_word_in_column(df, column_name, word):
    
    df_filtre = df[df[column_name].str.contains(rf'\b{word}\b', regex=True)]
    
    return df_filtre


   
def calculer_distance_euclidienne(x1, y1, x2, y2):
    try:
        # Compute Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    except Exception as e:
        print(f"Erreur lors du calcul de la distance : {e}")
        return np.nan