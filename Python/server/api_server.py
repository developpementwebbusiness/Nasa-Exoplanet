from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import pathlib
from pydantic import BaseModel, Field
from typing import List
import logging
from model import MonModeleIA
from utils.STAPredict import predict_rows
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
    statut: str
    prediction: dict
    user_id: str

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
        resultat_ia = predict_rows(data)
        
        logger.info(f"Prédiction effectuée: {resultat_ia}")
        
        # Préparer la réponse JSON
        
        return output_json(data_input = donnees.features, data_output=resultat_ia)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )


# ----------------------
# File upload / download
# ----------------------
@app.post("/upload_txt")
async def upload_txt(file: UploadFile = File(...), user_id: str = "anon"):
    """Receive a .txt file (multipart/form-data) and save it to the uploads directory."""
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted")
    dest = UPLOAD_DIR / file.filename
    try:
        with dest.open('wb') as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    logger.info(f"Uploaded file {file.filename} from user {user_id}")
    return {"status": "uploaded", "filename": file.filename, "user_id": user_id}


@app.get("/download_txt/{filename}")
async def download_txt(filename: str):
    """Return a .txt file from the uploads directory if it exists."""
    if '..' in filename or filename.startswith('/'):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = UPLOAD_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type='text/plain', filename=filename)

# Lancer le serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
