import shutil
import pathlib
import logging
import zipfile
import uvicorn
import io

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Union
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

# Ensure uploads directory exists
BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ✅ CORRIGÉ: 35 features (retiré TempUp et TempDown)
from typing import Union

class PredictRequest(BaseModel):
    # Support both formats: array of floats OR array of dicts OR single dict
    features: Optional[Union[List[float], List[List[float]], List[dict]]] = None
    data: Optional[List[dict]] = None  # For batch with named features
    user_id: str = Field(default="anonyme")

class ReponseIA(BaseModel):
    data: List[dict]

@app.post("/predict", response_model=ReponseIA)
async def predire(donnees: PredictRequest):
    """
    Unified prediction endpoint that handles both single and batch predictions.
    
    Supports three input formats:
    1. Single prediction with features array: {"features": [35 floats]}
    2. Batch prediction with features arrays: {"features": [[35 floats], [35 floats], ...]}
    3. Batch prediction with named features: {"data": [{"name": "...", "OrbitalPeriod": ..., ...}, ...]}
    """
    try:
        logger.info(f"Requête de {donnees.user_id}")
        
        rows = []
        names = []
        
        # Feature keys order (35 features, TempUp and TempDown removed)
        keys_order = [
            'OrbitalPeriod', 'OPup', 'OPdown', 'TransEpoch', 'TEup', 'TEdown',
            'Impact', 'ImpactUp', 'ImpactDown', 'TransitDur', 'DurUp', 'DurDown',
            'TransitDepth', 'DepthUp', 'DepthDown', 'PlanetRadius', 'RadiusUp', 'RadiusDown',
            'EquilibriumTemp', 'InsolationFlux', 'InsolationUp', 'InsolationDown',
            'TransitSNR', 'StellarEffTemp', 'SteffUp', 'SteffDown', 'StellarLogG', 'LogGUp', 'LogGDown',
            'StellarRadius', 'SradUp', 'SradDown', 'RA', 'Dec', 'KeplerMag'
        ]
        
        # Handle "data" field (batch with named features)
        if donnees.data is not None:
            logger.info(f"Batch request with {len(donnees.data)} samples (named features)")
            
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
                
                if len(features) != 35:
                    raise HTTPException(status_code=400, detail=f"Sample {j} has {len(features)} features, expected 35")
                
                rows.append(features)
                names.append(exoplanet.get("name", f"Exoplanet_{j+1}"))
        
        # Handle "features" field
        elif donnees.features is not None:
            # Check if it's a single array of floats or list of arrays
            if len(donnees.features) > 0:
                # Single prediction: [float, float, ...]
                if isinstance(donnees.features[0], (int, float)):
                    if len(donnees.features) != 35:
                        raise HTTPException(status_code=400, detail=f"Expected 35 features, got {len(donnees.features)}")
                    rows = [donnees.features]
                    names = ["Exoplanet_1"]
                    logger.info(f"Single prediction with {len(donnees.features)} features")
                
                # Batch prediction with arrays: [[float, ...], [float, ...]]
                elif isinstance(donnees.features[0], list):
                    for i, feature_row in enumerate(donnees.features):
                        if len(feature_row) != 35:
                            raise HTTPException(status_code=400, detail=f"Row {i} has {len(feature_row)} features, expected 35")
                        rows.append(feature_row)
                        names.append(f"Exoplanet_{i+1}")
                    logger.info(f"Batch prediction with {len(rows)} samples")
                
                # Batch prediction with dicts: [{"OrbitalPeriod": ..., ...}, ...]
                elif isinstance(donnees.features[0], dict):
                    for j, exoplanet in enumerate(donnees.features):
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
                        
                        if len(features) != 35:
                            raise HTTPException(status_code=400, detail=f"Sample {j} has {len(features)} features, expected 35")
                        
                        rows.append(features)
                        names.append(exoplanet.get("name", f"Exoplanet_{j+1}"))
                    logger.info(f"Batch prediction with {len(rows)} samples (dict format)")
                else:
                    raise HTTPException(status_code=400, detail="Invalid features format")
            else:
                raise HTTPException(status_code=400, detail="Empty features array")
        else:
            raise HTTPException(status_code=400, detail="Either 'features' or 'data' field is required")
        
        # Make predictions
        logger.info(f"Sending {len(rows)} samples to model")
        labels, confidence_scores = predict_rows(rows)
        logger.info(f"Predictions received - Labels: {labels}, Scores: {confidence_scores}")
        
        # Format response
        result = []
        for i, (label, confidence, name) in enumerate(zip(labels, confidence_scores, names)):
            result.append({
                "name": name,
                "score": float(confidence),
                "label": bool(label)
            })
        
        return {"data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/test_prediction")
async def test_prediction():
    """Test endpoint with dummy data to verify the API is working"""
    try:
        # ✅ 35 test values
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
async def export_model(model_id: Optional[str] = Query(None, description="Model ID to export. Use 'all' for complete package, 'STAR_AI_v2' for base model only, or specify a custom model filename")):
    """
    Export AI model(s) as a ZIP file.
    
    Parameters:
    - model_id: Optional. Defaults to 'all' if not specified
        - 'all': Export everything (base model + custom models)
        - 'STAR_AI_v2': Export only the base STAR AI model
        - Custom filename: Export specific custom model (e.g., 'my_custom_model.pth')
    
    Returns: ZIP file with requested model(s) and dependencies
    """
    try:
        # Default to 'all' if no model_id specified
        if model_id is None:
            model_id = 'all'
        
        model_dir = BASE_DIR / "utils" / "Data" / "AI" / "STAR_AI_v2"
        
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Export base STAR_AI_v2 model
            if model_id in ['all', 'STAR_AI_v2']:
                if not model_dir.exists():
                    raise HTTPException(status_code=404, detail="Base model directory not found")
                
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
                
                logger.info("Added base STAR_AI_v2 model to export")
            
            # Export custom models
            if model_id == 'all':
                # Export all custom models
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
            
            elif model_id not in ['STAR_AI_v2']:
                # Export specific custom model
                custom_model_path = UPLOAD_DIR / model_id
                if not custom_model_path.exists():
                    raise HTTPException(status_code=404, detail=f"Custom model '{model_id}' not found")
                
                if not custom_model_path.is_file():
                    raise HTTPException(status_code=400, detail=f"'{model_id}' is not a file")
                
                # Check if it's a valid model file
                if custom_model_path.suffix.lower() not in ['.pth', '.pkl', '.pt', '.h5', '.onnx']:
                    raise HTTPException(status_code=400, detail=f"'{model_id}' is not a valid model file")
                
                zip_file.write(custom_model_path, f"custom_models/{custom_model_path.name}")
                logger.info(f"Exporting specific custom model: {model_id}")
        
        zip_buffer.seek(0)
        
        # Generate filename based on model_id
        if model_id == 'all':
            filename = "STAR_AI_v2_complete.zip"
        elif model_id == 'STAR_AI_v2':
            filename = "STAR_AI_v2.zip"
        else:
            filename = f"{pathlib.Path(model_id).stem}.zip"
        
        logger.info(f"Exporting model package: {model_id}")
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)