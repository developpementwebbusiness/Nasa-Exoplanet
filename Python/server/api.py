from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.STARPredict import predict_rows

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'STAR AI API is running'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict exoplanet classification from input features.
    
    Expected JSON format:
    {
        "features": [[feature1, feature2, ..., feature35], ...]
    }
    
    Or for a single row:
    {
        "features": [feature1, feature2, ..., feature35]
    }
    
    Returns:
    {
        "predictions": [
            {
                "label": "CONFIRMED" or "FALSE POSITIVE" or "CANDIDATE",
                "confidence": 0.95
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" key in request body'
            }), 400
        
        features = data['features']
        
        # Handle single row input (convert to list of lists)
        if features and not isinstance(features[0], list):
            features = [features]
        
        # Validate input
        if not features or len(features) == 0:
            return jsonify({
                'error': 'Empty features array'
            }), 400
        
        # Check that each row has exactly 35 features
        for idx, row in enumerate(features):
            if len(row) != 35:
                return jsonify({
                    'error': f'Row {idx} has {len(row)} features, expected 35'
                }), 400
        
        # Make predictions
        labels, confidence_scores = predict_rows(features)
        
        # Format response
        predictions = []
        for label, confidence in zip(labels, confidence_scores):
            predictions.append({
                'label': str(label),
                'confidence': float(confidence)
            })
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input data: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the AI model"""
    return jsonify({
        'model_name': 'STAR_AI_v2',
        'input_features': 35,
        'output_classes': ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'],
        'description': 'Exoplanet classification model using Multi-Layer Perceptron'
    }), 200

if __name__ == '__main__':
    print("🚀 Starting STAR AI API...")
    print("📊 Model loaded and ready for predictions")
    print("🌐 Server running on http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET  /health      : Health check")
    print("  - POST /predict     : Make predictions")
    print("  - GET  /model-info  : Model information")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
