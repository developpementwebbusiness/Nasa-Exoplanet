# AI Model API (Python/server)

This folder contains a small FastAPI application that exposes a toy machine learning model (`MonModeleIA` in `model.py`) and endpoints for making predictions and uploading/downloading text files.

## Overview
- `api_server.py`: main FastAPI server. Available endpoints:
  - `GET /` : welcome message and list of endpoints
  - `GET /health` : health status and whether the model is loaded
  - `POST /predict` : send JSON with `features` (list of 4 floats) and optional `user_id` to get a prediction
  - `POST /upload_txt` : upload a `.txt` file (multipart/form-data)
  - `GET /download_txt/{filename}` : download a previously uploaded `.txt` file

- `model.py`: example class `MonModeleIA` that builds a `RandomForestClassifier` and returns predicted class, probabilities and confidence.

## Requirements
- Python 3.11 recommended
- (Optional but recommended) use the provided virtual environment `.venv`

## Installation (PowerShell)
From the `Python/server` folder:

```powershell
cd C:\Users\boula\Desktop\NSC\Nasa-Exoplanet\Python\server
```

If the `.venv` exists and is already activated (your prompt shows `(.venv)`), install dependencies:

```powershell
pip install -r requirements.txt
```

If the venv exists but is not activated:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To create a new virtual environment, activate it and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer to install without activating the venv:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Run the server
From `Python/server` (with venv activated):

```powershell
python api_server.py
```

The server will start on `http://0.0.0.0:8000`. Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Example `POST /predict` request
Send a JSON payload with `features` (a list of 4 floats) and optionally `user_id`.

Example with `curl`:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2], \"user_id\": \"user_123\"}"
```

Response: JSON with `statut`, `prediction` (predicted class, probabilities, confidence) and `user_id`.

## Upload / Download text files
- Upload (multipart/form-data):

```powershell
curl -X POST "http://127.0.0.1:8000/upload_txt" -F "file=@C:\path\to\myfile.txt" -F "user_id=me"
```

- Download:

```powershell
curl -X GET "http://127.0.0.1:8000/download_txt/myfile.txt" -o myfile.txt
```

## Troubleshooting
- `ModuleNotFoundError: No module named 'sklearn'` means scikit-learn is not installed. Make sure you run `pip install -r requirements.txt` inside the correct venv.
- If `pipreqs` fails with `UnicodeDecodeError`, run it while ignoring binary folders such as `.venv` and `.git`:

```powershell
.\.venv\Scripts\pipreqs.exe . --force --ignore .venv,.git
```

- To generate a `requirements.txt` that reflects the current venv exactly:

```powershell
.\.venv\Scripts\pip.exe freeze > requirements.txt
```

## Security notes
- The upload endpoint accepts only `.txt` files. Always validate and sanitize uploaded content before processing.
- Do not expose `.venv` or other folders containing executables to the public.

## Improvements you can make
- Load a trained model from disk using `joblib.load('model.joblib')` instead of training a fresh model on startup.
- Add authentication to protect the prediction and upload endpoints.
- Add unit tests and a CI pipeline.

---

If you'd like, I can:
- Install dependencies inside the venv for you.
- Run `pipreqs` to regenerate a lean `requirements.txt` (I will ignore `.venv` and `.git`).
- Modify `model.py` to load a joblib model file if you have one.
