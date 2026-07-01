# addok_conf.py

# Langue
LANGUAGE = 'fr'

# Activation explicite des plugins (par sécurité)
PLUGINS = [
    'addok.batch',
    'addok.pairs',
    'addok.fuzzy',
    'addok.autocomplete',
    'addok.http.base', 
    'addok.shell',
]

# **CORRECTION CRITIQUE FINALE** : Évite le problème de 'fork' sur Windows/WSL
BATCH_PROCESSORS = 1 

# Force la route Batch (par sécurité)
BATCH_ENDPOINT = 'batch'

# Configuration Redis pour Docker
REDIS_HOST = "addok-redis"
REDIS_PORT = 6379