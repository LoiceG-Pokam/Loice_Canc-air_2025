---

## ÉTAPE 2️⃣ : COPIER LES FICHIERS VERS C:\

⚠️ **TRÈS IMPORTANT !** Les fichiers sur des lecteurs réseau ou avec espaces dans le chemin causent des problèmes avec Docker et WSL.

**On copie vers C:\ (disque local, chemin simple).**

### Créer le dossier

```powershell
mkdir "C:\Users\lpokambo\addok_work"
```

⚠️ **Remplacez `lpokambo` par VOTRE nom d'utilisateur Windows !**

### Copier les fichiers

**Adaptez le chemin source selon votre PC :**

```powershell
# Exemple 1 : Si les fichiers sont sur H:\
copy "H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\pneumodetect\geocodage\addok_bd\*" "C:\Users\lpokambo\addok_work\"

# Exemple 2 : Si loice_pneumodetect est ailleurs, adaptez :
copy "D:\Projets\loice_pneumodetect\pneumodetect\geocodage\addok_bd\*" "C:\Users\lpokambo\addok_work\"

# Exemple 3 : Chemin complet depuis votre PC
copy "VOTRE_CHEMIN\loice_pneumodetect\pneumodetect\geocodage\addok_bd\*" "C:\Users\lpokambo\addok_work\"
```

### Vérifier que tout est copié

```powershell
dir "C:\Users\lpokambo\addok_work\"
```

**✅ Vous devriez voir :**
- `adresses-addok-france.ndjson.gz` (le fichier principal ~1.3 GB)
- `filter_ndjson.py`
- `addok_conf.py`
- `chunkcsv.py` ou `chunk_csv.py`
- Dossier `venv` (environnement virtuel)

⚠️ **IMPORTANT : Vérifiez que le dossier `venv` n'est PAS vide**

```powershell
dir "C:\Users\lpokambo\addok_work\venv"
```

**Si venv est vide ou corrompu :** Copiez-le manuellement depuis la source

```powershell
# Vérifiez d'abord que venv existe dans la source
dir "H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\pneumodetect\geocodage\addok_bd\venv"

# Si oui, copiez-le manuellement vers C:\
copy "H:\PFE Loice\Notebooks\Loice_Canc-air_2025\loice_pneumodetect\pneumodetect\geocodage\addok_bd\venv\*" "C:\Users\lpokambo\addok_work\venv\" /E

# Vérifiez que c'est bien copié
dir "C:\Users\lpokambo\addok_work\venv"
```

---

## ÉTAPE 3️⃣ : NAVIGUER AU BON DOSSIER

Maintenant tous les fichiers sont sur C:\ (local, aucun problème Docker)

```powershell
cd "C:\Users\lpokambo\addok_work"
```

⚠️ **Remplacez `lpokambo` par VOTRE nom d'utilisateur Windows !**

---

## ÉTAPE 4️⃣ : ACTIVER L'ENVIRONNEMENT VIRTUEL# 🌍 Guide Addok - Géocodage Complet

**Guide réutilisable pour tous - À adapter selon votre PC**

---

## 📍 Chemins à Adapter

Les chemins dans ce guide commencent à partir de `loice_pneumodetect`. 

**Adaptez selon votre PC :**

- **Votre chemin :** `C:\...\loice_pneumodetect\` ou `D:\...\loice_pneumodetect\` ou autre
- **Exemple Windows :** `C:\Users\VotreNom\loice_pneumodetect\`
- **Exemple :** `D:\Projets\loice_pneumodetect\`

⚠️ **Remplacez le chemin racine selon votre machine, mais gardez le reste identique !**

---

## 📋 Situation de Départ

- ✅ Dossier `loice_pneumodetect` avec tous les fichiers
- ✅ Environment virtuel déjà créé (on l'utilise !)
- ❌ Docker pas installé
- ❌ Les certificats Zcaler gèrent les connexions réseau

---

## 🎯 Ce que vous allez faire

1. **Installer Docker Desktop** (~15 min)
2. **Copier les fichiers au bon endroit** (simplifier les chemins)
3. **Activer l'environnement virtuel existant**
4. **Lancer l'indexation** (~4 heures)
5. **Lancer le serveur** et tester
6. **Utiliser Addok** pour géocoder

---

## ÉTAPE 1️⃣ : INSTALLER DOCKER DESKTOP

### Téléchargement

1. Ouvrez votre navigateur
2. Allez sur 👉 `https://www.docker.com/products/docker-desktop`
3. Cliquez **"Download for Windows"**
4. Attendez que `Docker Desktop Installer.exe` se télécharge (~500 MB)

### Installation

1. **Double-cliquez** sur `Docker Desktop Installer.exe`
2. L'écran "Configuration" apparaît avec ces options :
   ```
   ☑ Use WSL 2 instead of Hyper-V
   ☑ Add shortcut to desktop
   ```
3. **✅ COCHEZ** `Use WSL 2 instead of Hyper-V`
4. Cliquez **"OK"**
5. L'installation commence (~3-5 min)

### ⚠️ Redémarrage Obligatoire

Docker demande de redémarrer. **CLIQUEZ "Restart Now"** - c'est nécessaire !

### Vérification

Après redémarrage, ouvrez **PowerShell** et tapez :

```powershell
docker --version
```

**✅ Vous devriez voir :** `Docker version 29.5.3, build d1c06ef` (ou similaire)

Si erreur, redémarrez PowerShell ou l'ordinateur.

### Vérification que Docker tourne

```powershell
docker run hello-world
```

**✅ Si vous voyez "Hello from Docker!" :** Docker fonctionne ! 🎉

---

## ÉTAPE 2️⃣ : NAVIGUER AU BON DOSSIER

⚠️ **Adaptez le chemin selon votre PC !**

Le dossier `pneumodetect` contient tous les fichiers nécessaires.

```powershell
cd "VOTRE_CHEMIN\loice_pneumodetect\pneumodetect\geocodage\addok_bd"
```

**Exemples selon votre PC :**
- `C:\Users\VotreNom\loice_pneumodetect\pneumodetect\geocodage\addok_bd`
- `D:\Projets\loice_pneumodetect\pneumodetect\geocodage\addok_bd`
- `E:\Data\loice_pneumodetect\pneumodetect\geocodage\addok_bd`

### Vérifier que tout est là

```powershell
dir
```

**✅ Vous devriez voir :**
- `adresses-addok-france.ndjson.gz` (le fichier principal ~1.3 GB)
- `filter_ndjson.py`
- `addok_conf.py`
- `chunkcsv.py` ou `chunk_csv.py`
- Dossier `venv` (environnement virtuel existant)

---

## ÉTAPE 5️⃣ : ACTIVER L'ENVIRONNEMENT VIRTUEL

Un environnement virtuel Python existe déjà. Activez-le :

```powershell
# Activez venv
.\venv\Scripts\Activate.ps1

# Vous verrez (venv) au début de la ligne PowerShell
(venv) PS C:\Users\lpokambo\addok_work>
```

⚠️ **Si erreur "cannot be loaded"** : Exécutez PowerShell en **Administrateur** ou tapez :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## ÉTAPE 6️⃣ : LANCER REDIS

```powershell
docker run --name addok-redis -d redis:latest
```

**✅ Résultat attendu :** Un long ID (exemple: `02900eb9d46f...`)

---

## ÉTAPE 7️⃣ : LANCER L'INDEXATION (⏳ ~4 HEURES)

⚠️ **IMPORTANT :**
- NE FERMEZ PAS ce terminal pendant 4 heures
- Vous verrez des logs défiler (c'est normal)
- Silence à la fin = succès

**Vous êtes toujours dans** `addok_bd` **avec l'env virtuel activé**, tapez :

```powershell
docker run --rm -v "$(pwd):/data" --link addok-redis python:3.9-slim bash -c "pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org addok && gzip -cd /data/adresses-addok-france.ndjson.gz | python3 /data/filter_ndjson.py | addok batch"
```

**Ce qui se passe :**
1. `pip install addok` → installe Addok (gère les certificats Zcaler)
2. `gzip -cd` → décompresse le fichier (.gz → fichier complet)
3. `filter_ndjson.py` → nettoie les données invalides
4. `addok batch` → crée l'index (~27 millions d'adresses)
5. Crée le dossier `addok_index_idf/`

**⏱️ Durée :** ~3-4 heures selon la machine

**✅ Fin attendue :** Pas d'erreur, l'index `addok_index_idf/` créé

---

## ÉTAPE 8️⃣ : LANCER LE SERVEUR

Une fois l'indexation finie, **ouvrez un NOUVEAU terminal PowerShell** :

```powershell
cd "C:\Users\lpokambo\addok_work"

# Activez venv
.\venv\Scripts\Activate.ps1

# Lancez le serveur
docker run -d --name addok-server -p 7878:7878 --link addok-redis addok/addok serve
```

**✅ Résultat attendu :** Un ID de conteneur

---

## ÉTAPE 9️⃣ : TESTER LE SERVEUR

Ouvrez votre navigateur et allez à :

```
http://localhost:7878
```

**✅ Vous devriez voir :** Interface web Addok

Testez une recherche :

```
http://localhost:7878/search?q=5+avenue+paris+lyon
```

**✅ Vous devriez voir :** Les adresses correspondantes

---

## 🎉 C'EST PRÊT !

- ✔️ Docker installé
- ✔️ Environment virtuel activé
- ✔️ Redis lancé
- ✔️ Index créé
- ✔️ Serveur actif
- ✔️ Géocodage fonctionnel !

---

## 🔄 LA PROCHAINE FOIS

**L'indexation ne se refait JAMAIS.** Vous juste relancez les services.

```powershell
cd "VOTRE_CHEMIN\loice_pneumodetect\pneumodetect\geocodage\addok_bd"

# Activez venv
.\venv\Scripts\Activate.ps1

# Nettoyer les anciens conteneurs
docker rm -f addok-redis addok-server

# Lancer Redis
docker run --name addok-redis -d redis:latest

# Lancer le serveur
docker run -d --name addok-server -p 7878:7878 --link addok-redis addok/addok serve

# Tester
# Allez sur http://localhost:7878
```

**⏱️ Durée :** ~20 secondes (au lieu de 4 heures !)

---

## 💻 UTILISER ADDOK

### Interface Web
```
http://localhost:7878
```

### Requête Simple
```
http://localhost:7878/search?q=5+avenue+de+paris+lyon
```

### Géocodage en Masse (CSV)
```powershell
python chunk_csv.py --input adresses.csv --output resultats.csv
```

### Via Python
```python
import requests

response = requests.get('http://localhost:7878/search',
                        params={'q': '5 avenue paris lyon'})
print(response.json())
```

---

## 🛠️ PROBLÈMES & SOLUTIONS

### ❌ "Docker not found"
**Solution :** Redémarrez PowerShell ou l'ordinateur

### ❌ "Container already exists"
**Solution :**
```powershell
docker rm -f addok-redis addok-server
```

### ❌ "Le serveur ne répond pas"
**Solution :** Attendez 10-15 secondes après le lancement

### ❌ "L'indexation s'est arrêtée"
**Solution :** Relancez la commande (elle reprendra)

### ❌ Erreur "python-geohash: No such file or directory" ou "g++ failed"

**Cause :** Le conteneur `python:3.9-slim` est allégé et manque les outils de compilation (g++)

**Solution 1️⃣ : Utiliser python:3.9 complet (RECOMMANDÉ - Plus rapide)**

Remplacez la commande d'indexation par :

```powershell
docker run --rm -v "$(pwd):/data" --link addok-redis python:3.9 bash -c "pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org addok && gzip -cd /data/adresses-addok-france.ndjson.gz | python3 /data/filter_ndjson.py | addok batch"
```

**Solution 2️⃣ : Installer les build tools dans le slim (Plus lent)**

```powershell
docker run --rm -v "$(pwd):/data" --link addok-redis python:3.9-slim bash -c "apt-get update && apt-get install -y build-essential && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org addok && gzip -cd /data/adresses-addok-france.ndjson.gz | python3 /data/filter_ndjson.py | addok batch"
```

**⚠️ Utilisez la Solution 1 - elle est plus rapide !**

### ❌ Erreur SSL (certificats)
**Solution :** Déjà gérée avec `--trusted-host` dans la commande

---

## 📊 RÉSUMÉ

| Point | Détail |
|-------|--------|
| **Dossier** | `C:\Users\lpokambo\addok_work` |
| **Première fois** | ~4 heures |
| **Fois suivantes** | ~20 secondes |
| **Port** | 7878 |
| **URL** | http://localhost:7878 |
| **Certificats** | Gérés automatiquement (Zcaler) |

---

**Documentation Addok | Pneumodetect Project**

Base Adresse Nationale - Licence ouverte