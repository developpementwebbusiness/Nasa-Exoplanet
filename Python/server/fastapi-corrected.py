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

# 🔧 AJOUT: Configuration de sécurité
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".pkl", ".json", ".csv", ".txt", ".h5", ".pt", ".pth"}


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


# 🔧 AJOUT: Fonction de validation sécurisée des chemins
def validate_folder_path(ia_folder: str) -> pathlib.Path:
    """Valide et sécurise le chemin du dossier IA contre les attaques path traversal."""
    # Empêcher path traversal
    if ".." in ia_folder or ia_folder.startswith("/") or ia_folder.startswith("\\"):
        raise HTTPException(status_code=400, detail="Chemin invalide")
    
    # Construire le chemin sécurisé
    base_data_dir = BASE_DIR / "utils" / "Data"
    safe_path = base_data_dir / ia_folder
    
    # Vérifier que le chemin résolu reste dans la zone autorisée
    try:
        resolved_path = safe_path.resolve()
        resolved_base = base_data_dir.resolve()
        
        if resolved_base not in resolved_path.parents and resolved_path != resolved_base:
            raise HTTPException(status_code=403, detail="Accès interdit")
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Erreur de chemin: {str(e)}")
    
    return resolved_path


# 🔧 AJOUT: Fonction de validation des fichiers
def validate_file(file: UploadFile) -> None:
    """Valide le fichier uploadé (extension, nom, etc.)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")
    
    # Vérifier l'extension
    file_ext = pathlib.Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non autorisée: {file_ext}. Autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Empêcher path traversal dans le nom de fichier
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")


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
    return {"status": "healthy", "model_loaded": "None"}


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

            # 🔧 CORRECTION: Activer la prédiction IA réelle
            new_values = predict_rows(unknown_rows)  # ÉTAIT: "UwU :3"

            # 5) Insert-only en DB (ne remplace jamais l'existant)
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


# ----------------------
# File upload / download
# ----------------------


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
    
    # 🔧 CORRECTION: Utiliser la validation sécurisée
    dest_path = validate_folder_path(ia_folder)  # ÉTAIT: BASE_DIR / "utils" / "Data" / ia_folder
    dest_path.mkdir(parents=True, exist_ok=True)
    
    result = {"imported": [], "failures": [], "destination": str(dest_path)}
    
    for file in files:
        try:
            # 🔧 AJOUT: Validation du fichier
            validate_file(file)
            
            dest_file = dest_path / file.filename
            
            # 🔧 AMÉLIORATION: Lecture par chunks avec limitation de taille
            file_size = 0
            with dest_file.open("wb") as buffer:
                while chunk := await file.read(8192):  # 8KB chunks
                    file_size += len(chunk)
                    if file_size > MAX_FILE_SIZE:
                        dest_file.unlink(missing_ok=True)  # Supprimer fichier partiel
                        raise HTTPException(
                            status_code=413,
                            detail=f"Fichier trop volumineux: {file.filename}"
                        )
                    buffer.write(chunk)
            
            result["imported"].append({
                "filename": file.filename,
                "size": dest_file.stat().st_size,
                "path": str(dest_file)
            })
            logger.info(f"✓ Importé dans {ia_folder}: {file.filename} ({file_size} bytes) par {user_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            result["failures"].append({
                "filename": file.filename,
                "error": str(e)
            })
            logger.error(f"✗ Échec import {file.filename}: {str(e)}")
        finally:
            await file.close()
    
    # 🔧 AJOUT: Vérifier si tous les imports ont échoué
    if len(result["failures"]) > 0 and len(result["imported"]) == 0:
        raise HTTPException(
            status_code=500,
            detail=f"Tous les imports ont échoué: {result['failures']}"
        )
    
    return {
        "status": "completed",
        "ia_folder": ia_folder,
        "user_id": user_id,
        "imported_count": len(result["imported"]),
        "details": result
    }


# Endpoint pour exporter TOUS les fichiers depuis un dossier IA
@app.get("/export_ia_files/{ia_folder:path}")  # 🔧 AJOUT: :path pour supporter les slashes
async def export_ia_files(
    ia_folder: str,
    file_names: Optional[List[str]] = Query(None)
):
    """
    Exporte tous les fichiers depuis un dossier IA spécifique dans un ZIP.
    
    - Si file_names est fourni: exporte uniquement ces fichiers
    - Si file_names est vide/None: exporte TOUS les fichiers du dossier
    
    Exemples:
        GET /export_ia_files/STAR_AI_v2  (exporte TOUS les fichiers)
        GET /export_ia_files/STAR_AI_v2?file_names=file1.pkl&file_names=file2.pkl  (fichiers spécifiques)
    """

    # 🔧 CORRECTION: Utiliser la validation sécurisée
    source_path = validate_folder_path(ia_folder)  # ÉTAIT: "utils" / "Data" / ia_folder
    
    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dossier IA introuvable"  # 🔧 CORRECTION: Moins d'infos pour la sécurité
        )
    
    # Si aucun fichier spécifié, prendre TOUS les fichiers
    if file_names is None or len(file_names) == 0:
        all_files = [f.name for f in source_path.iterdir() if f.is_file()]
        file_names = all_files
        logger.info(f"Export de tous les fichiers ({len(file_names)})")  # 🔧 CORRECTION: Moins d'infos
    else:
        # 🔧 AJOUT: Valider chaque nom de fichier
        for fname in file_names:
            if ".." in fname or "/" in fname or "\\" in fname:
                raise HTTPException(status_code=400, detail=f"Nom de fichier invalide: {fname}")
    
    if len(file_names) == 0:
        raise HTTPException(
            status_code=404,
            detail="Aucun fichier trouvé dans le dossier"
        )
    
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            files_added = 0
            
            for file_name in file_names:
                file_path = source_path / file_name
                
                # 🔧 AJOUT: Vérification de sécurité supplémentaire
                try:
                    resolved_file = file_path.resolve()
                    if source_path.resolve() not in resolved_file.parents:
                        logger.warning(f"✗ Tentative d'accès hors dossier: {file_name}")
                        continue
                except (OSError, RuntimeError):
                    logger.warning(f"✗ Erreur résolution chemin: {file_name}")
                    continue
                
                if file_path.exists() and file_path.is_file():
                    # 🔧 CORRECTION: Nom simple dans le ZIP (pas de chemin complet)
                    arcname = file_name  # ÉTAIT: f"{ia_folder}/{file_name}"
                    zip_file.write(file_path, arcname=arcname)
                    files_added += 1
                    logger.info(f"✓ Ajouté au ZIP: {file_name}")
                else:
                    logger.warning(f"✗ Fichier introuvable: {file_name}")
            
            if files_added == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Aucun des fichiers spécifiés n'a été trouvé"
                )
        
        zip_buffer.seek(0)
        
        # 🔧 CORRECTION: Nettoyer le nom pour éviter les caractères dangereux
        safe_folder_name = ia_folder.replace("/", "_").replace("\\", "_")
        headers = {
            "Content-Disposition": f"attachment; filename={safe_folder_name}_export.zip"
        }
        
        logger.info(f"Export réussi: {files_added} fichiers")
        
        return Response(
            zip_buffer.getvalue(),
            headers=headers,
            media_type="application/zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de l'export"  # 🔧 CORRECTION: Message d'erreur générique
        )
    finally:
        zip_buffer.close()


# Lancer le serveur
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)