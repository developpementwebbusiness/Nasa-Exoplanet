import shutil
import pathlib
import logging
import zipfile
import uvicorn
import io
import asyncio
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from utils.STARPredict import predict_rows
from utils.AITrainer import AITRAIN
from utils.utils_json import convert, output_json
from utils.database import KVStore
from fastapi.responses import Response

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="API Modèle IA - Training & Prediction",
    description="API pour entraîner et utiliser des modèles d'intelligence artificielle",
    version="2.0.0"
)

# Ensure directories exist
BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AI_MODELS_DIR = BASE_DIR / "utils" / "Data" / "AI"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Store pour suivre les entraînements en cours
training_status = {}

# ================== MODÈLES PYDANTIC ==================

class PredictRequest(BaseModel):
    # Support both formats: array of floats OR array of dicts OR single dict
    features: Optional[Union[List[float], List[List[float]], List[dict]]] = None
    data: Optional[List[dict]] = None  # For batch with named features
    user_id: str = Field(default="anonyme")

class TrainingData(BaseModel):
    data: List[dict] = Field(..., description="Données d'entraînement (même format que predict + labels)")
    ai_name: str = Field(..., description="Nom du modèle IA à créer")
    hidden_layers: List[int] = Field(default=[128, 64], description="Architecture des couches cachées")
    epochs: int = Field(default=100, description="Nombre d'époques d'entraînement")
    user_id: str = Field(default="anonyme")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"OrbitalPeriod": "11.5", "PlanetRadius": "150", "label": "CONFIRMED"},
                    {"OrbitalPeriod": "19.4", "PlanetRadius": "7.18", "label": "FALSE POSITIVE"}
                ],
                "ai_name": "ExoplanetClassifier_v1",
                "hidden_layers": [128, 64, 32],
                "epochs": 50,
                "user_id": "data_scientist_1"
            }
        }

class ReponseIA(BaseModel):
    data: List[dict]

class TrainingResponse(BaseModel):
    message: str
    training_id: str
    ai_name: str
    status: str
    estimated_duration: str

class TrainingStatus(BaseModel):
    training_id: str
    ai_name: str
    status: str  # "queued", "training", "completed", "failed"
    current_epoch: Optional[int]
    total_epochs: int
    progress_percent: float
    loss: Optional[float]
    accuracy: Optional[float]
    estimated_time_remaining: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    error_message: Optional[str]
    model_path: Optional[str]
    scaler_path: Optional[str]
    le_path: Optional[str]

# ================== ENDPOINT DE PRÉDICTION UNIFIÉ ==================

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

# ================== FONCTIONS D'ENTRAÎNEMENT ==================

def training_worker(training_id: str, training_data: TrainingData):
    """
    Worker function : exécute l'entraînement en arrière-plan et prépare les données sous forme 
    rows = [[np.float64, ...], ...], labels = [label ...], comme le demande votre fonction training().
    """
    try:
        import numpy as np
        training_status[training_id]["status"] = "training"
        training_status[training_id]["start_time"] = datetime.now().isoformat()
        
        logger.info(f"Début de l'entraînement {training_id} pour {training_data.ai_name}")

        # ~~~~ Conversion stricte entrée (features 35 floats, autant que keys_order) ~~~~
        keys_order = [
            'OrbitalPeriod', 'OPup', 'OPdown', 'TransEpoch', 'TEup', 'TEdown',
            'Impact', 'ImpactUp', 'ImpactDown', 'TransitDur', 'DurUp', 'DurDown',
            'TransitDepth', 'DepthUp', 'DepthDown', 'PlanetRadius', 'RadiusUp', 'RadiusDown',
            'EquilibriumTemp', 'InsolationFlux', 'InsolationUp', 'InsolationDown',
            'TransitSNR', 'StellarEffTemp', 'SteffUp', 'SteffDown', 'StellarLogG', 'LogGUp', 'LogGDown',
            'StellarRadius', 'SradUp', 'SradDown', 'RA', 'Dec', 'KeplerMag'
        ]
        rows = []
        labels = []
        for row_data in training_data.data:
            row = []
            for key in keys_order:
                value = row_data.get(key)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    row.append(np.float64(0.0))
                else:
                    try:
                        row.append(np.float64(float(value)))
                    except Exception:
                        row.append(np.float64(0.0))
            rows.append(row)
            # Encodage du label string -> 1/0, à adapter à votre code d'entraînement réel
            label_raw = row_data.get('label', None)
            if label_raw is None:
                labels.append(0)
            elif isinstance(label_raw, (int, float)):
                labels.append(int(label_raw))
            else:
                labels.append(1 if str(label_raw).lower().startswith('conf') else 0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Callback pour suivi
        def progress_callback(epoch, total_epochs, loss, accuracy):
            if training_id in training_status:
                training_status[training_id]["current_epoch"] = epoch
                training_status[training_id]["progress_percent"] = (epoch / total_epochs) * 100
                training_status[training_id]["loss"] = float(loss) if loss is not None else None
                training_status[training_id]["accuracy"] = float(accuracy) if accuracy is not None else None
                elapsed_time = datetime.now() - datetime.fromisoformat(training_status[training_id]["start_time"])
                if epoch > 0:
                    time_per_epoch = elapsed_time.total_seconds() / epoch
                    remaining_epochs = total_epochs - epoch
                    estimated_remaining = remaining_epochs * time_per_epoch
                    training_status[training_id]["estimated_time_remaining"] = f"{estimated_remaining/60:.1f} min"
                logger.info(f"Training {training_id}: Epoch {epoch}/{total_epochs}, Loss: {loss}, Acc: {accuracy}")
        # Appel training
        result = AITRAIN(
            data=rows,
            hiddenlayers=training_data.hidden_layers,
            epochs=training_data.epochs,
            AIname=training_data.ai_name,
        )
        # Chemins de fichiers
        model_dir = AI_MODELS_DIR / training_data.ai_name
        model_path = str(model_dir / f"{training_data.ai_name}.pth")
        scaler_path = str(model_dir / "scaler.pkl")
        le_path = str(model_dir / "label_encoder.pkl")
        training_status[training_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "progress_percent": 100.0,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "le_path": le_path
        })
        logger.info(f"Entraînement {training_id} terminé avec succès")
    except Exception as e:
        training_status[training_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error_message": str(e)
        })
        logger.error(f"Erreur dans l'entraînement {training_id}: {str(e)}")
        import traceback; traceback.print_exc()


# ================== ENDPOINTS D'ENTRAÎNEMENT ==================

@app.post("/train", response_model=TrainingResponse)
async def train_model(training_data: TrainingData, background_tasks: BackgroundTasks):
    """
    Lance l'entraînement d'un nouveau modèle IA en arrière-plan
    """
    try:
        # Générer un ID unique pour cet entraînement
        training_id = f"train_{training_data.ai_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validation des données
        if len(training_data.data) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Minimum 10 échantillons requis pour l'entraînement"
            )
        
        # Vérifier que le nom de l'IA est valide
        if not training_data.ai_name.replace('_', '').isalnum():
            raise HTTPException(
                status_code=400,
                detail="Le nom de l'IA doit contenir uniquement des lettres, chiffres et underscores"
            )
        
        # Créer le dossier pour ce modèle
        model_dir = AI_MODELS_DIR / training_data.ai_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le statut de l'entraînement
        training_status[training_id] = {
            "training_id": training_id,
            "ai_name": training_data.ai_name,
            "status": "queued",
            "current_epoch": 0,
            "total_epochs": training_data.epochs,
            "progress_percent": 0.0,
            "loss": None,
            "accuracy": None,
            "estimated_time_remaining": None,
            "start_time": None,
            "end_time": None,
            "error_message": None,
            "model_path": None,
            "scaler_path": None,
            "le_path": None
        }
        
        # Lancer l'entraînement en arrière-plan
        background_tasks.add_task(training_worker, training_id, training_data)
        
        estimated_duration = f"{training_data.epochs * 0.5:.1f} minutes"
        
        return TrainingResponse(
            message=f"Entraînement lancé pour le modèle {training_data.ai_name}",
            training_id=training_id,
            ai_name=training_data.ai_name,
            status="queued",
            estimated_duration=estimated_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/train/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Récupère le statut d'un entraînement en cours"""
    if training_id not in training_status:
        raise HTTPException(status_code=404, detail="ID d'entraînement introuvable")
    
    status = training_status[training_id]
    return TrainingStatus(**status)

@app.get("/train/list")
async def list_trainings():
    """Liste tous les entraînements"""
    return {
        "trainings": [
            {
                "training_id": tid,
                "ai_name": status["ai_name"],
                "status": status["status"],
                "progress_percent": status["progress_percent"],
                "start_time": status["start_time"]
            }
            for tid, status in training_status.items()
        ],
        "total": len(training_status)
    }

@app.delete("/train/{training_id}")
async def cancel_training(training_id: str):
    """Annule un entraînement (si possible)"""
    if training_id not in training_status:
        raise HTTPException(status_code=404, detail="ID d'entraînement introuvable")
    
    status = training_status[training_id]["status"]
    
    if status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail=f"Impossible d'annuler: entraînement {status}")
    
    training_status[training_id]["status"] = "cancelled"
    training_status[training_id]["end_time"] = datetime.now().isoformat()
    
    return {"message": f"Entraînement {training_id} annulé"}

@app.get("/models/list")
async def list_models():
    """Liste tous les modèles IA disponibles"""
    models = []
    
    if AI_MODELS_DIR.exists():
        for model_dir in AI_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "files": []
                }
                
                # Lister les fichiers du modèle
                for file_path in model_dir.iterdir():
                    if file_path.is_file():
                        model_info["files"].append({
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                
                models.append(model_info)
    
    return {"models": models, "total": len(models)}

# ================== ENDPOINT D'EXPORT DE MODÈLES ==================

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
        
        model_dir = AI_MODELS_DIR / "STAR_AI_v2"
        
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
                # Export all custom models from AI_MODELS_DIR
                if AI_MODELS_DIR.exists():
                    custom_model_count = 0
                    for custom_dir in AI_MODELS_DIR.iterdir():
                        if custom_dir.is_dir() and custom_dir.name != "STAR_AI_v2":
                            for file_path in custom_dir.iterdir():
                                if file_path.is_file() and file_path.suffix.lower() in ['.pth', '.pkl', '.pt', '.h5', '.onnx']:
                                    zip_file.write(file_path, f"custom_models/{custom_dir.name}/{file_path.name}")
                                    custom_model_count += 1
                                    logger.info(f"Added custom model: {custom_dir.name}/{file_path.name}")
                    
                    if custom_model_count > 0:
                        logger.info(f"Included {custom_model_count} custom model files")
            
            elif model_id not in ['STAR_AI_v2']:
                # Export specific custom model directory
                custom_model_dir = AI_MODELS_DIR / model_id
                if not custom_model_dir.exists():
                    raise HTTPException(status_code=404, detail=f"Custom model '{model_id}' not found")
                
                if not custom_model_dir.is_dir():
                    raise HTTPException(status_code=400, detail=f"'{model_id}' is not a model directory")
                
                # Add all files from the custom model directory
                for file_path in custom_model_dir.iterdir():
                    if file_path.is_file():
                        zip_file.write(file_path, f"custom_models/{model_id}/{file_path.name}")
                        logger.info(f"Added file: {model_id}/{file_path.name}")
                
                logger.info(f"Exporting specific custom model: {model_id}")
        
        zip_buffer.seek(0)
        
        # Generate filename based on model_id
        if model_id == 'all':
            filename = "AI_models_complete.zip"
        elif model_id == 'STAR_AI_v2':
            filename = "STAR_AI_v2.zip"
        else:
            filename = f"{model_id}.zip"
        
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

# ================== ENDPOINTS DE TEST ==================

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

@app.get("/")
async def root():
    return {
        "message": "API Modèle IA - Training & Prediction",
        "version": "2.0.0",
        "endpoints": {
            "prediction": ["/predict"],
            "training": ["/train", "/train/status/{training_id}", "/train/list"],
            "models": ["/models/list", "/export_model"],
            "test": ["/test_prediction"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)