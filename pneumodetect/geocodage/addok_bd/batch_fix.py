import gzip
import json
import io 
import sys
import os
import redis 
from addok.batch import batch

# --- CONFIGURATION REDIS ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# --- CONFIGURATION FICHIER (DYNAMIQUE) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "adresses-addok-france.ndjson.gz")

if not os.path.exists(FILE_PATH):
    print(f"❌ Erreur : Fichier introuvable : {FILE_PATH}")
    sys.exit(1)


if __name__ == '__main__':
    
    print("🚀 Lancement de l'indexation...")
    print(f"📂 Fichier : {FILE_PATH}")
    
    try:
        # Vérifier la connexion Redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        print(f"✅ Connexion établie à Redis sur {REDIS_HOST}:{REDIS_PORT}")
        
        # Ouvrir le fichier gzip et passer au stdin d'Addok batch
        with gzip.open(FILE_PATH, 'rb') as f:
            # Remplacer stdin par le fichier gzip
            original_stdin = sys.stdin
            sys.stdin = io.TextIOWrapper(f, encoding='utf-8')
            
            try:
                # Lancer la commande batch d'Addok
                batch(sys.stdin)
                print("\n✅ Importation des adresses terminée.")
            finally:
                sys.stdin = original_stdin
        
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Erreur de Connexion Redis : {e}")
        print("Assurez-vous que Redis est en cours d'exécution !")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur critique : {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Prochaine étape: ngrams (addok ngrams).")