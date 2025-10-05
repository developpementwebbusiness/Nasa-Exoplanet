from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import pathlib
from pydantic import BaseModel, Field
from typing import List
import logging
from model import MonModeleIA
#from utils.STARPredict import predict_rows
from utils.utils_json import convert, output_json


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="API Modèle IA",
    description="API pour communiquer avec un modèle d'intelligence artificielle",
    version="1.0.0"
)

# Charger le modèle IA au démarrage
modele_ia = MonModeleIA()

# Ensure uploads directory exists
BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Définir la structure des données d'entrée avec validation
class DonneesEntree(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
    user_id: str = Field(default="anonyme")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "user_id": "utilisateur_123"
            }
        }

# Définir la structure des données de sortie
class ReponseIA(BaseModel):
    data : list[dict]

# Endpoint de santé pour vérifier que l'API fonctionne
@app.get("/")
async def root():
    return {
        "message": "API Modèle IA opérationnelle",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": modele_ia is not None}

# Endpoint principal: reçoit JSON, fait tourner l'IA, renvoie JSON
@app.post("/predict", response_model=ReponseIA)
async def predire(donnees: DonneesEntree):
    """
    Endpoint qui:
    1. Reçoit les données JSON via l'API
    2. Fait tourner le modèle IA
    3. Renvoie la prédiction en JSON
    """
    try:
        logger.info(f"Requête reçue de {donnees.user_id}")
        
        # Faire tourner l'IA sur les données reçues
        data = convert(donnees.features)
        resultat_ia = []#predict_rows(data[0])

        "Faire la partie database"
        logger.info(f"Prédiction effectuée: {resultat_ia}")
        
        # Préparer la réponse JSON
        #Ajouter la bonne liste de hashage et donner la liste de donnée
        return {"data" : output_json(data_output=resultat_ia,list_hash=data[1],list_name=data[2])}
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )


# ----------------------
# File upload / download
# ----------------------
# Endpoint pour importer 3 fichiers simultanément
# Endpoint pour importer 3 fichiers dans un dossier IA spécifique
@app.post("/import_ia_files")
async def import_ia_files(
    files: List[UploadFile] = File(...),
    ia_folder: str = "IA_1",  # IA_1, IA_2, etc.
    user_id: str = "anon"
):
    """
    Importe exactement 3 fichiers dans un dossier IA spécifique.
    Exemple: Data/IA_1/
    """
    if len(files) != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Exactement 3 fichiers requis, {len(files)} reçus"
        )
    
    # Définir le chemin destination: utils/Data/IA_X/
    dest_path = BASE_DIR / "utils" / "Data" / ia_folder
    dest_path.mkdir(parents=True, exist_ok=True)
    
    result = {"imported": [], "failures": [], "destination": str(dest_path)}
    
    for file in files:
        dest_file = dest_path / file.filename
        
        try:
            with dest_file.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result["imported"].append({
                "filename": file.filename,
                "size": dest_file.stat().st_size,
                "path": str(dest_file)
            })
            logger.info(f"✓ Importé dans {ia_folder}: {file.filename} par {user_id}")
            
        except Exception as e:
            result["failures"].append({
                "filename": file.filename,
                "error": str(e)
            })
            logger.error(f"✗ Échec import {file.filename}: {str(e)}")
        finally:
            await file.close()
    
    return {
        "status": "completed",
        "ia_folder": ia_folder,
        "user_id": user_id,
        "imported_count": len(result["imported"]),
        "details": result
    }


# Endpoint pour importer 3 fichiers dans un dossier IA spécifique
@app.post("/import_ia_files")
async def import_ia_files(
    files: List[UploadFile] = File(...),
    ia_folder: str = "AI/STAR_AI_v2",
    user_id: str = "anon"
):
    """
    Importe exactement 3 fichiers dans un dossier IA spécifique.
    Les fichiers sont stockés dans: utils/Data/{ia_folder}/
    """
    if len(files) != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Exactement 3 fichiers requis, {len(files)} reçus"
        )
    
    # Définir le chemin destination: utils/Data/{ia_folder}/
    dest_path = BASE_DIR / "utils" / "Data" / ia_folder
    dest_path.mkdir(parents=True, exist_ok=True)
    
    result = {"imported": [], "failures": [], "destination": str(dest_path)}
    
    for file in files:
        dest_file = dest_path / file.filename
        
        try:
            with dest_file.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result["imported"].append({
                "filename": file.filename,
                "size": dest_file.stat().st_size,
                "path": str(dest_file)
            })
            logger.info(f"✓ Importé dans {ia_folder}: {file.filename} par {user_id}")
            
        except Exception as e:
            result["failures"].append({
                "filename": file.filename,
                "error": str(e)
            })
            logger.error(f"✗ Échec import {file.filename}: {str(e)}")
        finally:
            await file.close()
    
    return {
        "status": "completed",
        "ia_folder": ia_folder,
        "user_id": user_id,
        "imported_count": len(result["imported"]),
        "details": result
    }


# Endpoint pour exporter 3 fichiers depuis un dossier IA
@app.get("/export_ia_files/{ia_folder}")
async def export_ia_files(
    ia_folder: str,
    file_names: List[str] = None
):
    """
    Exporte 3 fichiers depuis un dossier IA spécifique dans un ZIP.
    Si file_names n'est pas fourni, exporte les 3 premiers fichiers trouvés.
    
    Exemples:
        GET /export_ia_files/STAR_AI_v2
        GET /export_ia_files/STAR_AI_v2?file_names=file1.pkl&file_names=file2.pkl&file_names=file3.pth
    """
    import zipfile
    import io
    from fastapi.responses import Response
    
    # Chemin source: utils/Data/{ia_folder}/
    source_path = BASE_DIR / "utils" / "Data" / ia_folder
    
    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dossier IA introuvable: {ia_folder}"
        )
    
    # Si aucun fichier spécifié, prendre les 3 premiers fichiers
    if file_names is None or len(file_names) == 0:
        all_files = [f.name for f in source_path.iterdir() if f.is_file()]
        file_names = all_files[:3]
        logger.info(f"Aucun fichier spécifié, export des 3 premiers: {file_names}")
    
    if len(file_names) != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Exactement 3 fichiers requis, {len(file_names)} fournis"
        )
    
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            files_added = 0
            
            for file_name in file_names:
                file_path = source_path / file_name
                
                if file_path.exists() and file_path.is_file():
                    # Ajouter au ZIP avec structure: {ia_folder}/filename
                    arcname = f"{ia_folder}/{file_name}"
                    zip_file.write(file_path, arcname=arcname)
                    files_added += 1
                    logger.info(f"✓ Ajouté au ZIP depuis {ia_folder}: {file_name}")
                else:
                    logger.warning(f"✗ Fichier introuvable dans {ia_folder}: {file_name}")
            
            if files_added == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Aucun des fichiers spécifiés n'a été trouvé dans {ia_folder}"
                )
        
        zip_buffer.seek(0)
        
        headers = {
            "Content-Disposition": f"attachment; filename={ia_folder}_export.zip"
        }
        
        return Response(
            zip_buffer.getvalue(),
            headers=headers,
            media_type="application/zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'export depuis {ia_folder}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'export: {str(e)}"
        )
    finally:
        zip_buffer.close()

# Lancer le serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
