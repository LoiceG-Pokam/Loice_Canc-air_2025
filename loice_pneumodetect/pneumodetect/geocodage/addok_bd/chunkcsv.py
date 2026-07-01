import os
import math
import shutil
import requests

import requests
import json
import pandas as pd
 

# Use http://localhost:7878 if you run a local instance.

# ADDOK_URL = """CHANGE WITH YOUR OWN SERVER ADRESS"""

ADDOK_URL = "http://127.0.0.1:7878/search"
 
# def geocode(filepath_in, requests_options, filepath_out='geocoded.csv'):
#     with open(filepath_in, 'rb') as f:
#         filename, response = post_to_addok(filepath_in, f.read(), requests_options)
#         # print(response)
#         write_response_to_disk(filepath_out, response)


# -------------------------------------- Nouvelle Fonction Loice ----------------------------------------------------------------------------
def geocode(filepath_in, columns, filepath_out='geocoded.csv'):
    """
    Lit un CSV, géocode chaque ligne via l'API Search (GET), et enregistre les résultats.
    """
    # 1. Lire le chunk CSV
    df_chunk = pd.read_csv(filepath_in, sep=';')
    
    # 2. Préparer les colonnes pour les résultats
    results = []
    
    # 3. Définir les noms de colonnes du fichier d'entrée
    adresse_col = columns['columns'][0] # 'adresse'
    codepostal_col = columns['columns'][1] # 'code_postal'
    commune_col = columns['columns'][2] # 'commune'

    # 4. Géocodage ligne par ligne
    print(f" Début du géocodage ligne par ligne pour {filepath_in}...")
    
    for index, row in df_chunk.iterrows():
        
        # Récupération des valeurs (gestion des valeurs manquantes/NaN)
        address = str(row[adresse_col]) if pd.notna(row[adresse_col]) else ""
        postcode = str(row[codepostal_col]) if pd.notna(row[codepostal_col]) else None
        city = str(row[commune_col]) if pd.notna(row[commune_col]) else None
        
        # Appel à la fonction de recherche ligne par ligne
        geo_result = search_single_address(address, postcode, city)
        
        # Stockage du résultat
        results.append(geo_result)
        
        if index % 1000 == 0 and index > 0:
            print(f"   -> {index} adresses traitées.")


    # 5. Fusionner les résultats avec le DataFrame original
    df_results = pd.DataFrame(results)
    df_geocoded = pd.concat([df_chunk, df_results], axis=1)

    # 6. Écrire le fichier de sortie
    df_geocoded.to_csv(filepath_out, sep= ';', index=False)
    print(f"✅ Géocodage de {filepath_in} terminé. Résultats enregistrés dans {filepath_out}.")
        
# -------------------------------------- Nouvelle Fonction Loice ----------------------------------------------------------------------------



def geocode_chunked(filepath_in, filename_pattern, chunk_by_approximate_lines, requests_options):
    b = os.path.getsize(filepath_in)
    output_files = []
    with open(filepath_in, 'r') as bigfile:
        row_count = sum(1 for row in bigfile)
    with open(filepath_in, 'r') as bigfile:
        headers = bigfile.readline()
        chunk_by = math.ceil(b / row_count * chunk_by_approximate_lines)
        current_lines = bigfile.readlines(chunk_by)
        i = 1
        # import ipdb;ipdb.set_trace()
        while current_lines:
            current_filename = filename_pattern.format(i)
            current_csv = ''.join([headers] + current_lines)
            # import ipdb;ipdb.set_trace()
            filename, response = post_to_addok(current_filename, current_csv, requests_options)
            write_response_to_disk(current_filename, response)
            current_lines = bigfile.readlines(chunk_by)
            i += 1
            output_files.append(current_filename)
    return output_files



## ------------------------------------------ NOUVELLE FONCTION LOICE --------------------------------------------------------------------------------
def search_single_address(address, postcode, city):
    """
    Envoie une seule requête GET à l'API Addok.
    """
    # Construction de la chaîne de recherche (q)
    query_parts = [address]
    if postcode:
        query_parts.append(str(postcode))
    if city:
        query_parts.append(city)
        
    search_query = " ".join(query_parts)
    
    # Paramètres de la requête
    params = {'q': search_query}
    
    try:
        # Envoi de la requête GET
        response = requests.get(ADDOK_URL, params=params, verify=False, timeout=10)
        response.raise_for_status()
        
        # Le serveur doit renvoyer un JSON
        data = response.json()
        
        # Extraction du premier résultat (le plus pertinent)
        if data and data.get('features'):
            feature = data['features'][0]
            geometry = feature['geometry']
            properties = feature['properties']
            
            return {
                'match': True,
                'lon': geometry['coordinates'][0],
                'lat': geometry['coordinates'][1],
                'result_label': properties.get('label'),
                'score': properties.get('score'),
                'type': properties.get('type')
            }
        else:
            return {'match': False, 'lon': None, 'lat': None, 'score': 0, 'result_label': None, 'type': None}

    except requests.exceptions.HTTPError as e:
        print(f"❌ Erreur HTTP lors de la recherche : {e}")
        # En cas d'erreur, ne pas planter, retourner une ligne vide
        return {'match': False, 'lon': None, 'lat': None, 'score': 0, 'result_label': 'HTTP Error', 'type': None}
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion lors de la recherche : {e}")
        return {'match': False, 'lon': None, 'lat': None, 'score': 0, 'result_label': 'Connection Error', 'type': None}
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        return {'match': False, 'lon': None, 'lat': None, 'score': 0, 'result_label': 'Unknown Error', 'type': None}
 
## ------------------------------------------ NOUVELLE FONCTION LOICE --------------------------------------------------------------------------------



def write_response_to_disk(filename, response, chunk_size=1024):
    with open(filename, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

# def post_to_addok(filename, filelike_object, requests_options):
#     files = {'data': (filename, filelike_object)}
#     response = requests.post(ADDOK_URL, files=files, data=requests_options)
#    # You might want to use https://github.com/g2p/rfc6266
#     content_disposition = response.headers['content-disposition']
#     filename = content_disposition[len('attachment; filename="'):-len('"')]
#     return filename, response


def post_to_addok(filename, filelike_object, requests_options):


    # ----------------------------------------- CODE JOSEPHINE -----------------------------------------------------------------------

    # 1. Préparation du fichier à uploader
    # files = {'data': (filename, filelike_object)}

    # # 2. Préparation des options : S'assurer que 'columns' est une chaîne JSON
    # # Créer une copie pour ne pas modifier l'objet original
    # payload = requests_options.copy() 
    # if 'columns' in payload and isinstance(payload['columns'], list):
    #     payload['columns'] = json.dumps(payload['columns'])

    # # 3. Envoi de la requête POST
    # # L'URL doit être "http://localhost:7878/batch" (comme corrigé précédemment)
    # response = requests.post(ADDOK_URL, files=files, data=payload)

    # ----------------------------------------- CODE JOSEPHINE -----------------------------------------------------------------------
    

    files = {
        'data': (filename, filelike_object, 'text/csv'), # Envoi explicite du fichier CSV
    }
    
    # Utilisez 'data' pour les options et 'files' pour le fichier.
    response = requests.post(
        ADDOK_URL,
        data=requests_options, # Options comme {'columns': [...]}
        files=files,           # Le fichier CSV
    )


    #### ------------------------------------------------   CODE DE JOSEPHINE COMMENTE -----------------------------------------------------------------

    # **NOUVEAU : Vérification du statut HTTP**
    # Si le statut est 4xx ou 5xx, ceci lèvera l'exception HTTPError et affichera le message d'erreur d'Addok
    # try:
    #     response.raise_for_status()
    # except requests.exceptions.HTTPError as e:
    #     print(f"❌ Erreur HTTP lors de l'envoi à Addok : {e}")
    #     print(f"Réponse du serveur (détails de l'erreur) : {response.text}")
    #     # Si nous avons une erreur, nous devons renvoyer quelque chose ou lever l'exception
    #     # Pour éviter la KeyError qui suit, levons l'exception après affichage de l'erreur serveur :
    #     raise

     #### ------------------------------------------------   CODE DE JOSEPHINE COMMENTE -----------------------------------------------------------------

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # ❌ Affichage du message d'erreur détaillé du serveur Addok ❌
        print(f"❌ Erreur HTTP lors de l'envoi à Addok : {e}")
        print("Réponse détaillée du serveur (pour diagnostic) :")
        print(response.text) # <-- C'est cette ligne qui va nous donner l'info
        raise # Lève l'exception pour arrêter le processus

    # 4. Lecture de l'en-tête (seulement si la requête a réussi, grâce à raise_for_status)
    content_disposition = response.headers['content-disposition']
    filename = content_disposition[len('attachment; filename="'):-len('"')]
    return filename, response


def process_dataframe_in_chunks2(df, label ,chunk_size, chunks_dir_path, chunks_geo_dir_path, columns): 
    
    for i, chunk in enumerate(range(0, len(df), chunk_size)): 
    
        df_chunk = df.iloc[chunk:chunk + chunk_size] 
        chunk_path_in = os.path.join(chunks_dir_path, f"{label}_chunk_{i}.csv") 
        chunk_path_out = os.path.join(chunks_geo_dir_path, f"{label}_chunk_{i}_geocoded.csv") 
        df_chunk.to_csv(chunk_path_in, sep=';', index=False)
        
        geocode(chunk_path_in, columns, chunk_path_out)

 

 
def consolidate_multiple_csv(files, output_name):
    with open(output_name, 'wb') as outfile:
        for i, fname in enumerate(files):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)


