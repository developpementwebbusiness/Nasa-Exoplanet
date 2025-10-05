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
from fastapi.responses import Response, FileResponse

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
        labels, confidence_scores = predict_rows(rows)
        
        result = []
        for i, (label, confidence) in enumerate(zip(labels, confidence_scores)):
            result.append({
                "name": f"Exoplanet_{i+1}",
                "score": float(confidence),
                "label": bool(label)
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
        
        labels, confidence_scores = predict_rows(rows)
        logger.info(f"Prédictions reçues - Labels: {labels}, Scores: {confidence_scores}")
        
        result = []
        for i, (label, confidence) in enumerate(zip(labels, confidence_scores)):
            result.append({
                "name": donnees.data[i].get("name", f"Exoplanet_{i+1}"),
                "score": float(confidence),
                "label": bool(label)
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
        
        labels, confidence_scores = predict_rows([test_row])
        
        result = [{
            "name": "Test_Exoplanet",
            "score": float(confidence_scores[0]),
            "label": bool(labels[0])
        }]
        
        return {"data": result}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/export_model")
async def export_model():
    """
    Export the complete AI model package as a ZIP file.
    Includes all necessary files: model, scaler, label_encoder, and any custom uploaded models
    """
    try:
        model_dir = BASE_DIR / "utils" / "Data" / "AI" / "STAR_AI_v2"
        
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model directory not found")
        
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add model file (required)
            model_path = model_dir / "STAR_AI_v2.pth"
            if model_path.exists():
                zip_file.write(model_path, "STAR_AI_v2/STAR_AI_v2.pth")
            else:
                raise HTTPException(status_code=404, detail="Model file not found")
            
            # Add scaler file (required for predictions)
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                zip_file.write(scaler_path, "STAR_AI_v2/scaler.pkl")
            else:
                raise HTTPException(status_code=404, detail="Scaler file not found")
            
            # Add label encoder file (required for predictions)
            le_path = model_dir / "label_encoder.pkl"
            if le_path.exists():
                zip_file.write(le_path, "STAR_AI_v2/label_encoder.pkl")
            else:
                raise HTTPException(status_code=404, detail="Label encoder file not found")
            
            # Add custom uploaded models from uploads directory
            if UPLOAD_DIR.exists():
                custom_model_count = 0
                for file_path in UPLOAD_DIR.iterdir():
                    if file_path.is_file():
                        # Include .pth, .pkl, .pt, .h5 model files
                        if file_path.suffix.lower() in ['.pth', '.pkl', '.pt', '.h5', '.onnx']:
                            zip_file.write(file_path, f"custom_models/{file_path.name}")
                            custom_model_count += 1
                            logger.info(f"Added custom model: {file_path.name}")
                
                if custom_model_count > 0:
                    logger.info(f"Included {custom_model_count} custom model(s)")
        
        zip_buffer.seek(0)
        
        logger.info("Exporting complete model package (base model + custom uploads)")
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=STAR_AI_v2_complete.zip"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)