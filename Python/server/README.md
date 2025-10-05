# STAR AI API Documentation

FastAPI-based REST API for exoplanet classification using the STAR AI v2 neural network model.

## Base URL

```
http://localhost:8000
```

## Table of Contents

- [Getting Started](#getting-started)
- [Endpoints](#endpoints)
  - [POST /predict](#post-predict)
  - [GET /test_prediction](#get-test_prediction)
  - [GET /export_model](#get-export_model)
- [Data Format](#data-format)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Getting Started

### Starting the API Server

```bash
cd Python/server
python api.py
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints

### POST /predict

Unified prediction endpoint that handles both single and batch predictions.

**Supports 3 Input Formats:**

#### 1. Single Prediction (Array of Floats)

```json
{
  "features": [0.1, 0.2, 0.3, ..., 35 values total],
  "user_id": "user123"  // optional
}
```

#### 2. Batch Prediction (Array of Arrays)

```json
{
  "features": [
    [0.1, 0.2, 0.3, ..., 35 values],
    [1.1, 1.2, 1.3, ..., 35 values],
    [2.1, 2.2, 2.3, ..., 35 values]
  ],
  "user_id": "user123"  // optional
}
```

#### 3. Batch Prediction with Named Features

```json
{
  "data": [
    {
      "name": "KOI-123",
      "OrbitalPeriod": 2.7,
      "OPup": 0.0,
      "OPdown": 0.0,
      ... // all 35 features
    },
    {
      "name": "KOI-456",
      "OrbitalPeriod": 5.4,
      ... // all 35 features
    }
  ],
  "user_id": "user123"  // optional
}
```

**Feature List (35 features, in order):**
```
OrbitalPeriod, OPup, OPdown, TransEpoch, TEup, TEdown,
Impact, ImpactUp, ImpactDown, TransitDur, DurUp, DurDown,
TransitDepth, DepthUp, DepthDown, PlanetRadius, RadiusUp, RadiusDown,
EquilibriumTemp, InsolationFlux, InsolationUp, InsolationDown,
TransitSNR, StellarEffTemp, SteffUp, SteffDown, StellarLogG, LogGUp, LogGDown,
StellarRadius, SradUp, SradDown, RA, Dec, KeplerMag
```

**Note:** `TempUp` and `TempDown` columns are NOT included (removed from the original 37 features).

**Note:** `TempUp` and `TempDown` columns are NOT included (removed from the original 37 features).

**Response (Same for all formats):**

```json
{
  "data": [
    {
      "name": "Exoplanet_1",  // or custom name if provided
      "score": 0.9523456789,
      "label": true
    },
    {
      "name": "KOI-456",
      "score": 0.2341234567,
      "label": false
    }
  ]
}
```

**Fields:**
- `features` or `data`: Input data (choose one format)
- `user_id`: Optional string to identify the user making the request
- `score`: Confidence score from the AI model (0.0 to 1.0)
- `label`: Boolean prediction (true = confirmed exoplanet, false = false positive/candidate)

**Examples:**

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [2.7, 0.0, 0.0, 170.7, 0.002, -0.002, 1.016, 3.967, -0.059, 3.38, 0.089, -0.089, 860.4, 20.9, -20.9, 6.54, 1.64, -0.5, 1222.0, 527.8, 411.3, -166.4, 48.3, 5853.0, 158.0, -176.0, 4.544, 0.044, -0.176, 0.868, 0.233, -0.078, 297.0, 48.1, 15.4],
    "user_id": "test_user"
  }'

# Batch prediction with arrays
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [2.7, 0.0, ..., 35 values],
      [5.4, 0.0, ..., 35 values]
    ]
  }'

# Batch prediction with named features
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"name": "KOI-123", "OrbitalPeriod": 2.7, "OPup": 0.0, ...},
      {"name": "KOI-456", "OrbitalPeriod": 5.4, "OPup": 0.0, ...}
    ]
  }'
```

---

### GET /test_prediction

Test endpoint with dummy data to verify the API and model are working correctly.

**Request:**

```bash
curl http://localhost:8000/test_prediction
```

**Response:**

```json
{
  "data": [
    {
      "name": "Test_Exoplanet",
      "score": 0.8234567890,
      "label": true
    }
  ]
}
```

---

### GET /export_model

Export AI model files as a ZIP archive. Supports downloading the base model, custom models, or everything.

**Query Parameters:**
- `model_id` (optional): Specifies which model to export
  - `all` (default): Export base model + all custom models
  - `STAR_AI_v2`: Export only the base STAR AI model
  - `<filename>`: Export specific custom model (e.g., `my_model.pth`)

**Examples:**

```bash
# Export everything (default)
curl -O http://localhost:8000/export_model

# Export only base model
curl -O "http://localhost:8000/export_model?model_id=STAR_AI_v2"

# Export specific custom model
curl -O "http://localhost:8000/export_model?model_id=custom_model.pth"
```

**Response:**
- Content-Type: `application/zip`
- Downloads a ZIP file containing:
  - **Base Model** (`STAR_AI_v2/`): `STAR_AI_v2.pth`, `scaler.pkl`, `label_encoder.pkl`
  - **Custom Models** (`custom_models/`): Any uploaded custom model files

**ZIP Structure:**
```
STAR_AI_v2_complete.zip
├── STAR_AI_v2/
│   ├── STAR_AI_v2.pth         # Neural network model
│   ├── scaler.pkl             # Data preprocessing scaler
│   └── label_encoder.pkl      # Label encoder
└── custom_models/
    ├── custom_model_1.pth
    └── custom_model_2.pkl
```

---

## Data Format

### Feature Requirements

- **Count**: Exactly 35 numeric features per sample
- **Type**: Float values (can include negative numbers, decimals, zeros)
- **Missing Values**: Use `0.0` or `null` (will be converted to 0.0)
- **Removed Features**: `TempUp` and `TempDown` are excluded from the original dataset

### Feature Order (Important!)

When using `/predict` endpoint, features must be in this exact order:

```
1. OrbitalPeriod       11. DurUp             21. InsolationUp      31. SradUp
2. OPup                12. DurDown           22. InsolationDown    32. SradDown
3. OPdown              13. TransitDepth      23. TransitSNR        33. RA
4. TransEpoch          14. DepthUp           24. StellarEffTemp    34. Dec
5. TEup                15. DepthDown         25. SteffUp           35. KeplerMag
6. TEdown              16. PlanetRadius      26. SteffDown
7. Impact              17. RadiusUp          27. StellarLogG
8. ImpactUp            18. RadiusDown        28. LogGUp
9. ImpactDown          19. EquilibriumTemp   29. LogGDown
10. TransitDur         20. InsolationFlux    30. StellarRadius
```

---

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Row 0 has 30 features, expected 35"
}
```

**404 Not Found:**
```json
{
  "detail": "Custom model 'nonexistent.pth' not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Error: [error message]"
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Invalid request (wrong number of features, invalid data)
- `404`: Resource not found (model file missing)
- `500`: Server error (model prediction failed)

---

## Examples

### Python Example

```python
import requests
import json

API_URL = "http://localhost:8000"

# Single prediction
def predict_single(features):
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features, "user_id": "python_client"}
    )
    return response.json()

# Batch prediction with arrays
def predict_batch_arrays(features_list):
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features_list, "user_id": "python_client"}
    )
    return response.json()

# Batch prediction with named features
def predict_batch_named(exoplanets):
    response = requests.post(
        f"{API_URL}/predict",
        json={"data": exoplanets, "user_id": "python_client"}
    )
    return response.json()

# Example usage - Single
features = [2.7, 0.0, 0.0, 170.7, ...] # 35 values
result = predict_single(features)
print(f"Prediction: {result['data'][0]['label']}")
print(f"Confidence: {result['data'][0]['score']:.2%}")

# Example usage - Batch with arrays
features_list = [
    [2.7, 0.0, ...],  # 35 values
    [5.4, 0.0, ...]   # 35 values
]
results = predict_batch_arrays(features_list)

# Example usage - Batch with named features
exoplanets = [
    {"name": "KOI-123", "OrbitalPeriod": 2.7, "OPup": 0.0, ...},
    {"name": "KOI-456", "OrbitalPeriod": 5.4, "OPup": 0.0, ...}
]
results = predict_batch_named(exoplanets)
```

### JavaScript Example

```javascript
// Single prediction
async function predictSingle(features) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      features: features,
      user_id: 'js_client'
    })
  });
  return await response.json();
}

// Batch prediction with arrays
async function predictBatchArrays(featuresList) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      features: featuresList,
      user_id: 'js_client'
    })
  });
  return await response.json();
}

// Batch prediction with named features
async function predictBatchNamed(exoplanets) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      data: exoplanets,
      user_id: 'js_client'
    })
  });
  return await response.json();
}

// Example usage - Single
const features = [2.7, 0.0, 0.0, 170.7, /* ... 35 values */];
const result = await predictSingle(features);
console.log('Prediction:', result.data[0].label);
console.log('Confidence:', (result.data[0].score * 100).toFixed(2) + '%');

// Example usage - Batch
const exoplanets = [
  {name: "KOI-123", OrbitalPeriod: 2.7, OPup: 0.0, /* ... */},
  {name: "KOI-456", OrbitalPeriod: 5.4, OPup: 0.0, /* ... */}
];
const results = await predictBatchNamed(exoplanets);
```

### cURL Examples

```bash
# Test the API
curl http://localhost:8000/test_prediction

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [2.7, 0.0, 0.0, 170.7, 0.002, -0.002, 1.016, 3.967, -0.059, 3.38, 0.089, -0.089, 860.4, 20.9, -20.9, 6.54, 1.64, -0.5, 1222.0, 527.8, 411.3, -166.4, 48.3, 5853.0, 158.0, -176.0, 4.544, 0.044, -0.176, 0.868, 0.233, -0.078, 297.0, 48.1, 15.4]}'

# Download model
curl -O "http://localhost:8000/export_model?model_id=all"
```

---

## Model Information

### STAR AI v2 Model

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Framework**: PyTorch
- **Input**: 35 numeric features
- **Output**: Binary classification (exoplanet vs. false positive) + confidence score
- **Hidden Layers**: [128, 64]
- **Activation**: ReLU
- **Dropout**: 0.2

### Model Files

1. **STAR_AI_v2.pth**: PyTorch model weights
2. **scaler.pkl**: StandardScaler for input normalization
3. **label_encoder.pkl**: LabelEncoder for output labels

All three files are required for predictions and are included in the model export.

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production use, consider adding rate limiting middleware.

---

## CORS

CORS is enabled for all origins by default. This can be configured in the API code if needed.

---

## Support

For issues or questions:
- Check the interactive docs at `/docs`
- Review the logs in the terminal where the API is running
- Ensure you have exactly 35 features per sample
- Verify all numeric values are valid (not NaN or Infinity)

---

## Version

API Version: 1.0.0  
Model Version: STAR_AI_v2
