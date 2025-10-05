"""
Test client for STAR AI API
This script demonstrates how to make requests to the Flask API
"""

import requests
import json

# API base URL
API_URL = "http://localhost:5000"

def test_health():
    """Test the health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_model_info():
    """Test the model info endpoint"""
    print("Testing /model-info endpoint...")
    response = requests.get(f"{API_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict_single():
    """Test prediction with a single sample"""
    print("Testing /predict endpoint with single sample...")
    
    # Example: 35 dummy features (replace with real data)
    sample_features = [1.0] * 35  # Replace with actual feature values
    
    payload = {
        "features": sample_features
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict_batch():
    """Test prediction with multiple samples"""
    print("Testing /predict endpoint with batch samples...")
    
    # Example: 3 samples with 35 features each
    batch_features = [
        [1.0] * 35,
        [2.0] * 35,
        [0.5] * 35
    ]
    
    payload = {
        "features": batch_features
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_invalid_request():
    """Test error handling with invalid input"""
    print("Testing /predict endpoint with invalid input...")
    
    # Wrong number of features
    payload = {
        "features": [1.0, 2.0, 3.0]  # Only 3 features instead of 35
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("STAR AI API Test Client")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_model_info()
        test_predict_single()
        test_predict_batch()
        test_invalid_request()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the API is running at", API_URL)
    except Exception as e:
        print(f"❌ Error: {e}")
