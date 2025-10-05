import shutil
import pathlib
import logging
import zipfile
import uvicorn
import io

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.STARPredict import predict_rows
from utils.utils_json import convert, output_json
from utils.database import KVStore
from fastapi.responses import Response

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="API Modèle IA",
    description="API pour communiquer avec un modèle d'intelligence artificielle",
    version="1.0.0"
)

# Charger la database
db = KVStore("store.db")

# Ensure uploads directory exists
BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ✅ CORRIGÉ: 35 features (retiré TempUp et TempDown)
class DonneesEntree(BaseModel):
    features: List[float] = Field(..., min_items=35, max_items=35)
    user_id: str = Field(default="anonyme")

class ExoplanetsData(BaseModel):
    data: List[dict] = Field(..., description="Liste des dictionnaires d'exoplanètes")
    user_id: str = Field(default="anonyme")

class ReponseIA(BaseModel):
    data: List[dict]

@app.post("/predict", response_model=ReponseIA)
async def predire(donnees: DonneesEntree):
    try:
        logger.info(f"Requête de {donnees.user_id}: {len(donnees.features)} features")

        rows = [donnees.features]
        predictions = predict_rows(rows)
        
        result = []
        for i, prediction in enumerate(predictions):
            if isinstance(prediction, (list, tuple)):
                score = float(prediction[0]) if len(prediction) > 0 else 0.5
            else:
                score = float(prediction) if isinstance(prediction, (int, float)) else 0.5
            
            result.append({
                "name": f"Exoplanet_{i+1}",
                "score": round(score, 4),
                "label": "CONFIRMED" if score > 0.5 else "FALSE POSITIVE"
            })
        
        return {"data": result}

    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/predict_batch", response_model=ReponseIA)
async def predire_batch(donnees: ExoplanetsData):
    try:
        logger.info(f"Batch de {donnees.user_id}: {len(donnees.data)} exoplanètes")
        
        # ✅ CORRIGÉ: 35 clés - RETIRÉ TempUp et TempDown (qui sont None de toute façon)
        keys_order = [
            'OrbitalPeriod', 'OPup', 'OPdown', 'TransEpoch', 'TEup', 'TEdown',
            'Impact', 'ImpactUp', 'ImpactDown', 'TransitDur', 'DurUp', 'DurDown',
            'TransitDepth', 'DepthUp', 'DepthDown', 'PlanetRadius', 'RadiusUp', 'RadiusDown',
            'EquilibriumTemp', 'InsolationFlux', 'InsolationUp', 'InsolationDown',
            'TransitSNR', 'StellarEffTemp', 'SteffUp', 'SteffDown', 'StellarLogG', 'LogGUp', 'LogGDown',
            'StellarRadius', 'SradUp', 'SradDown', 'RA', 'Dec', 'KeplerMag'
        ]
        
        logger.info(f"Conversion avec {len(keys_order)} features (sans TempUp/TempDown)")
        
        rows = []
        for j, exoplanet in enumerate(donnees.data):
            features = []
            for key in keys_order:
                value = exoplanet.get(key)
                if value is None:
                    features.append(0.0)
                else:
                    try:
                        features.append(float(value))
                    except (ValueError, TypeError):
                        features.append(0.0)
            
            logger.info(f"Exoplanète {j+1}: {len(features)} features")
            rows.append(features)
        
        # Vérification avant predict_rows
        logger.info(f"Envoi à predict_rows: {len(rows)} lignes de {len(rows[0])} features chacune")
        
        predictions = predict_rows(rows)
        logger.info(f"Prédictions reçues: {predictions}")
        
        result = []
        for i, prediction in enumerate(predictions):
            if isinstance(prediction, (list, tuple)):
                score = float(prediction[0]) if len(prediction) > 0 else 0.5
            else:
                score = float(prediction) if isinstance(prediction, (int, float)) else 0.5
            
            result.append({
                "name": f"Exoplanet_{i+1}",
                "score": round(score, 4),
                "label": True if score > 0.5 else False
            })
        
        return {"data": result}

    except Exception as e:
        logger.error(f"Erreur batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur batch: {str(e)}")

@app.get("/test_prediction")
async def test_prediction():
    try:
        # ✅ 35 valeurs de test
        test_row = [0.1 + (i * 0.02) for i in range(35)]
        
        predictions = predict_rows([test_row])
        
        result = [{
            "name": "Test_Exoplanet",
            "score": round(float(predictions[0]), 4) if predictions else 0.5,
            "label": "CONFIRMED" if (predictions and float(predictions[0]) > 0.5) else "FALSE POSITIVE"
        }]
        
        return {"data": result}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)