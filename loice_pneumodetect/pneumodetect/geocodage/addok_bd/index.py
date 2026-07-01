import gzip
import os
import subprocess
import sys

FILE_PATH = "adresses-addok-france.ndjson.gz"

# Chemin direct vers l'exe addok du venv (pas de "-m", pas de PATH)
ADDOK_EXE = os.path.join(os.path.dirname(sys.executable), "addok.exe")

print("🚀 Indexation avec filter_ndjson.py...")

with gzip.open(FILE_PATH, 'rb') as f_in:
    # Lancer filter_ndjson.py → addok batch
    proc_filter = subprocess.Popen(
        [sys.executable, "filter_ndjson.py"],
        stdin=f_in,
        stdout=subprocess.PIPE,
        bufsize=65536
    )

    proc_batch = subprocess.Popen(
        [ADDOK_EXE, "batch"],   # appelle directement l'exe du venv, aucune ambiguite possible
        stdin=proc_filter.stdout,
        bufsize=65536
    )

    proc_batch.wait()

# On verifie vraiment le code de sortie avant d'annoncer un succes
if proc_batch.returncode != 0:
    print(f"Indexation ECHOUEE (code {proc_batch.returncode}). Regardez le message d'erreur ci-dessus.")
    sys.exit(proc_batch.returncode)

print("Indexation terminee !")
