@echo off
REM Créer un environnement virtuel dans le dossier
cd /d "H:\PFE Loice\Notebooks"
python -m venv my_env_pfe

REM Activer l'environnement virtuel
call my_env\Scripts\activate

REM Mettre pip à jour
python -m pip install --upgrade pip

REM Installer les packages depuis requirements.txt
pip install -r requirements.txt

REM Ajouter le kernel à Jupyter (facultatif mais utile)
python -m ipykernel install --user --name=my_env --display-name "Python (PFE)"

echo.
echo ✅ Environnement créé et packages installés !
pause
