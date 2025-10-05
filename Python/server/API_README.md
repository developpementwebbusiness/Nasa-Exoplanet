# STAR AI Flask API

A Flask API wrapper for the STAR AI exoplanet classification model.

## Features

- **Health Check**: Verify the API is running
- **Predictions**: Classify exoplanet candidates
- **Model Info**: Get information about the AI model
- **CORS Enabled**: Ready for cross-origin requests
- **Error Handling**: Comprehensive validation and error messages

## Installation

1. Install dependencies:

```bash
pip install -r ../requirements.txt
```

Or install Flask specifically:

```bash
pip install flask flask-cors
```

## Running the API

From the `Python/server` directory:

```bash
python api.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "message": "STAR AI API is running"
}
```

---

### 2. Model Information

**GET** `/model-info`

Get details about the AI model.

**Response:**

```json
{
  "model_name": "STAR_AI_v2",
  "input_features": 35,
  "output_classes": ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"],
  "description": "Exoplanet classification model using Multi-Layer Perceptron"
}
```

---

### 3. Make Predictions

**POST** `/predict`

Classify exoplanet candidates based on input features.

**Request Body:**

Single prediction:

```json
{
  "features": [1.0, 2.0, 3.0, ..., 35.0]
}
```

Batch predictions:

```json
{
  "features": [
    [1.0, 2.0, 3.0, ..., 35.0],
    [1.1, 2.1, 3.1, ..., 35.1],
    [1.2, 2.2, 3.2, ..., 35.2]
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {
      "label": "CONFIRMED",
      "confidence": 0.95
    },
    {
      "label": "FALSE POSITIVE",
      "confidence": 0.87
    }
  ],
  "count": 2
}
```

**Error Response:**

```json
{
  "error": "Row 0 has 3 features, expected 35"
}
```

## Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model-info

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, ..., 35.0]}'
```

### Using Python (requests)

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:5000/predict",
    json={"features": [1.0, 2.0, 3.0, ..., 35.0]}
)

result = response.json()
print(f"Label: {result['predictions'][0]['label']}")
print(f"Confidence: {result['predictions'][0]['confidence']}")
```

### Using JavaScript (fetch)

```javascript
// Make a prediction
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    features: [1.0, 2.0, 3.0, ..., 35.0]
  })
})
.then(response => response.json())
.then(data => {
  console.log('Label:', data.predictions[0].label);
  console.log('Confidence:', data.predictions[0].confidence);
});
```

## Testing

Run the test client:

```bash
python test_api.py
```

This will test all endpoints and demonstrate proper usage.

## Input Features

The model expects **35 features** per sample. Make sure your input data has been preprocessed and scaled appropriately to match the training data format.

## Model Details

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Hidden Layers**: [128, 64]
- **Output Classes**: 3 (CONFIRMED, FALSE POSITIVE, CANDIDATE)
- **Framework**: PyTorch
- **Scaling**: Standard scaling (pre-trained scaler included)

## Error Handling

The API validates:

- JSON format
- Required "features" key
- Correct number of features (35)
- Data types (numeric values)

All errors return appropriate HTTP status codes:

- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

## Production Deployment

For production use, consider:

1. Using a production WSGI server (e.g., Gunicorn):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

2. Adding authentication/API keys
3. Rate limiting
4. Logging and monitoring
5. Load balancing for high traffic

## License

Part of the NASA Exoplanet project.
