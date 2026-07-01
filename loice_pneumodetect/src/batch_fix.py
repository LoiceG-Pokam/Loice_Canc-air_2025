import gzip
import json
import io 
import sys
import redis 

# Importation pour Addok 2.x (doit être stable)
from addok.helpers.index import index_document 

# Importation du core config
from addok.config import config


# --- CONFIGURATION REDIS ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# --- CONFIGURATION FICHIER ---
FILE_PATH = r"H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\Notebooks\addok_bd\adresses-addok-france.ndjson.gz"


if __name__ == '__main__':
    
    # 🌟 CORRECTION CRITIQUE (Configuration Manuelle Minimale) 🌟
    # Le 'NoneType' object is not iterable est causé par l'absence de cette ligne
    config.LANGUAGE = 'fr' 

    # Définir les filtres/processors essentiels pour être sûr qu'ils ne sont pas None
    config.FILTERS = [
        'addok.helpers.text.tokenize',
        'addok.helpers.text.normalize',
    ]
    config.PROCESSORS = [
        'addok.helpers.text.synonymize',
        'addok.helpers.text.slugify',
    ]
    config.INTERSECT_FILTERS = [
        'addok.helpers.text.skip_roman_numeral',
    ]
    # -----------------------------------------------------------------

    print("🚀 Lancement de l'indexation SÉQUENTIELLE (Configuration Minimale et Stable)...")
    
    total_indexed = 0
    
    try:
        # 1. Initialiser la connexion Redis et créer le pipeline
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        pipe = r.pipeline()
        print(f"Connexion établie à Redis sur {REDIS_HOST}:{REDIS_PORT}")
        
        # 2. Ouverture GZIP en binaire, puis enveloppe en UTF-8
        with gzip.open(FILE_PATH, 'rb') as f_binary:
            f_text = io.TextIOWrapper(f_binary, encoding='utf-8')
            
            for i, line in enumerate(f_text):
                try:
                    data = json.loads(line)
                    
                    # CORRECTION DE LA CLEF MANQUANTE (KeyError: '_id')
                    if '_id' not in data and 'id' in data:
                        data['_id'] = data['id']
                    
                    # 3. Appel de la fonction avec les deux arguments requis: pipe et doc=data
                    index_document(pipe, doc=data) 
                    
                    total_indexed += 1
                    
                    if total_indexed % 50000 == 0:
                        # Exécuter les commandes en attente dans le pipeline (flush)
                        pipe.execute()
                        print(f"Indexé : {total_indexed:,} documents...", end='\r')
                        
                except json.JSONDecodeError as e:
                    print(f"Erreur de décodage JSON ligne {i+1} : {e}", file=sys.stderr)
                
        # Exécuter les commandes restantes
        pipe.execute()
        
        print(f"\n✅ Importation des adresses terminée. Total indexé : {total_indexed:,} documents.")

    except redis.exceptions.ConnectionError as e:
        print(f"\n❌ Erreur de Connexion Redis : {e}. Assurez-vous que le serveur Redis est démarré et fonctionne.", file=sys.stderr)
    except Exception as e:
        # Affichage sécurisé de l'erreur
        print(f"\n❌ Erreur critique : {type(e).__name__}: {e}", file=sys.stderr)

    print("Prochaine étape: ngrams (addok ngrams).")