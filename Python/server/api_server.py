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

        # 1) Convertir les features -> (rows, hash_list, name_list)
        data = convert(donnees.features)
        rows, hash_list, name_list = data[0], data[1], data[2]

        # 2) Interroger la DB : séparer connus / inconnus
        known_vals, unknown_hashes = db.split_with_data(hash_list)

        if unknown_hashes:
            # 3) Construire les lignes à prédire alignées à unknown_hashes (ordre & doublons)
            idx_by_hash = {}
            for i, h in enumerate(hash_list):
                idx_by_hash.setdefault(h, []).append(i)

            unknown_rows = []
            for h in unknown_hashes:
                i = idx_by_hash[h].pop(0)
                unknown_rows.append(rows[i])

            # 4) IA uniquement sur les inconnus
            new_values = predict_rows(unknown_rows)  # len(new_values) == len(unknown_hashes)

            # 5) Insert-only en DB (ne remplace jamais l’existant)
            insert_map = {}
            for h, val in zip(unknown_hashes, new_values):
                if h not in insert_map:  # si hash répété, garder la 1re valeur prédite
                    insert_map[h] = val
            db.insert_new_many(insert_map)

            # 6) Relecture DB pour obtenir toutes les valeurs alignées à hash_list
            final_vals, still_unknown = db.split_with_data(hash_list)
            if still_unknown:
                logger.warning(f"Encore inconnus après insertion: {still_unknown}")
            resultat_ia = final_vals
        else:
            # Tout était déjà en cache
            resultat_ia = known_vals

        logger.info(f"Prédiction effectuée: {resultat_ia}")

        # 7) Réponse JSON finale (valeurs alignées à hash_list)
        return {"data": output_json(data_output=resultat_ia, list_hash=hash_list, list_name=name_list)}

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

# Lancer le serveur
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
