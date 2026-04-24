import sys
import json

def filter_ndjson():
    # Lit depuis l'entrée standard (le pipe)
    for line in sys.stdin:
        try:
            # Tente de charger la ligne comme un objet JSON
            doc = json.loads(line)
            
            # Vérifie les conditions de validité d'Addok
            if doc and doc.get('name'):
                # Si le document est valide et a un nom, on le réimprime sur la sortie standard
                sys.stdout.write(line)
                
        except json.JSONDecodeError:
            # Ignore les lignes qui ne sont pas du JSON valide
            continue
        except Exception:
            # Ignore toute autre erreur
            continue

if __name__ == "__main__":
    filter_ndjson()
