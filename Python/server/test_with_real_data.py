"""
Test the Flask API with REAL data from the Kepler dataset
"""

import requests
import json
import pandas as pd
import sys

# API base URL
API_URL = "http://localhost:5000"

def load_real_data():
    """Load real exoplanet data from the Kepler CSV"""
    print("📂 Loading real data from kepler.csv...")
    
    # Load the CSV file
    df = pd.read_csv("utils/Data/kepler.csv", comment='#')
    
    print(f"✅ Loaded {len(df)} rows from dataset")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    print(f"⚠️  Found {nan_count} NaN values in dataset")
    
    # Fill NaN values with 0 (the AI should be trained to handle this)
    df = df.fillna(0)
    print(f"✅ Filled NaN values with 0")
    
    # Remove the 'Confirmation' column (first column) for predictions
    # The AI shouldn't see the answer!
    feature_columns = df.columns[1:]  # Skip first column (koi_disposition)
    
    print(f"🔢 Using {len(feature_columns)} features for prediction (37 features, API will filter to 35)")
    print(f"ℹ️  Note: TempUp and TempDown columns will be removed by the API")
    
    return df, feature_columns

def test_with_confirmed_exoplanet(df, feature_columns):
    """Test with a CONFIRMED exoplanet"""
    print("\n" + "="*60)
    print("TEST 1: CONFIRMED Exoplanet")
    print("="*60)
    
    # Get first confirmed exoplanet
    confirmed = df[df['koi_disposition'] == 'CONFIRMED'].iloc[0]
    
    print(f"📍 Expected Label: CONFIRMED")
    print(f"🔭 Orbital Period: {confirmed['koi_period']} days")
    print(f"🌡️  Equilibrium Temp: {confirmed['koi_teq']} K")
    print(f"📏 Planet Radius: {confirmed['koi_prad']} Earth radii")
    
    # Extract features (excluding the label)
    features = confirmed[feature_columns].values.tolist()
    
    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['predictions'][0]
        
        print(f"\n🤖 AI Prediction: {prediction['label']}")
        print(f"📊 Confidence: {prediction['confidence']:.2%}")
        
        if prediction['label'] == 'CONFIRMED':
            print("✅ CORRECT PREDICTION!")
        else:
            print("❌ WRONG PREDICTION!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_with_false_positive(df, feature_columns):
    """Test with a FALSE POSITIVE"""
    print("\n" + "="*60)
    print("TEST 2: FALSE POSITIVE")
    print("="*60)
    
    # Get first false positive
    false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE'].iloc[0]
    
    print(f"📍 Expected Label: FALSE POSITIVE")
    print(f"🔭 Orbital Period: {false_pos['koi_period']} days")
    print(f"📏 Planet Radius: {false_pos['koi_prad']} Earth radii")
    
    # Extract features
    features = false_pos[feature_columns].values.tolist()
    
    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['predictions'][0]
        
        print(f"\n🤖 AI Prediction: {prediction['label']}")
        print(f"📊 Confidence: {prediction['confidence']:.2%}")
        
        if prediction['label'] == 'FALSE POSITIVE':
            print("✅ CORRECT PREDICTION!")
        else:
            print("❌ WRONG PREDICTION!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_with_candidate(df, feature_columns):
    """Test with a CANDIDATE"""
    print("\n" + "="*60)
    print("TEST 3: CANDIDATE")
    print("="*60)
    
    # Get first candidate
    candidate = df[df['koi_disposition'] == 'CANDIDATE'].iloc[0]
    
    print(f"📍 Expected Label: CANDIDATE")
    print(f"🔭 Orbital Period: {candidate['koi_period']} days")
    print(f"📏 Planet Radius: {candidate['koi_prad']} Earth radii")
    
    # Extract features
    features = candidate[feature_columns].values.tolist()
    
    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['predictions'][0]
        
        print(f"\n🤖 AI Prediction: {prediction['label']}")
        print(f"📊 Confidence: {prediction['confidence']:.2%}")
        
        if prediction['label'] == 'CANDIDATE':
            print("✅ CORRECT PREDICTION!")
        else:
            print("❌ WRONG PREDICTION!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_batch_prediction(df, feature_columns):
    """Test with multiple samples at once"""
    print("\n" + "="*60)
    print("TEST 4: BATCH PREDICTION (5 samples)")
    print("="*60)
    
    # Get 5 random samples
    samples = df.sample(n=5)
    
    # Extract features for all samples
    features_batch = []
    expected_labels = []
    
    for idx, row in samples.iterrows():
        features = row[feature_columns].values.tolist()
        features_batch.append(features)
        expected_labels.append(row['koi_disposition'])
    
    print(f"📦 Sending {len(features_batch)} samples to API...")
    
    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features_batch},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        predictions = result['predictions']
        
        print(f"\n📊 Results:")
        correct = 0
        for i, (pred, expected) in enumerate(zip(predictions, expected_labels)):
            match = "✅" if pred['label'] == expected else "❌"
            print(f"  {i+1}. Expected: {expected:15s} | Predicted: {pred['label']:15s} | Confidence: {pred['confidence']:.2%} {match}")
            if pred['label'] == expected:
                correct += 1
        
        accuracy = (correct / len(predictions)) * 100
        print(f"\n🎯 Batch Accuracy: {accuracy:.1f}% ({correct}/{len(predictions)})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_accuracy_on_subset(df, feature_columns, n_samples=20):
    """Test accuracy on a subset of data"""
    print("\n" + "="*60)
    print(f"TEST 5: ACCURACY TEST ({n_samples} samples)")
    print("="*60)
    
    # Get n random samples
    samples = df.sample(n=n_samples)
    
    # Extract features
    features_batch = []
    expected_labels = []
    
    for idx, row in samples.iterrows():
        features = row[feature_columns].values.tolist()
        features_batch.append(features)
        expected_labels.append(row['koi_disposition'])
    
    print(f"📦 Testing {len(features_batch)} samples...")
    
    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features_batch},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        predictions = result['predictions']
        
        # Calculate accuracy by category
        correct_by_category = {'CONFIRMED': 0, 'FALSE POSITIVE': 0, 'CANDIDATE': 0}
        total_by_category = {'CONFIRMED': 0, 'FALSE POSITIVE': 0, 'CANDIDATE': 0}
        
        for pred, expected in zip(predictions, expected_labels):
            total_by_category[expected] += 1
            if pred['label'] == expected:
                correct_by_category[expected] += 1
        
        print(f"\n📊 Accuracy by Category:")
        for category in ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']:
            if total_by_category[category] > 0:
                acc = (correct_by_category[category] / total_by_category[category]) * 100
                print(f"  {category:15s}: {acc:.1f}% ({correct_by_category[category]}/{total_by_category[category]})")
        
        total_correct = sum(correct_by_category.values())
        total = sum(total_by_category.values())
        overall_accuracy = (total_correct / total) * 100
        
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    print("\n" + "🌟" * 30)
    print("   STAR AI API - REAL DATA TEST")
    print("🌟" * 30 + "\n")
    
    try:
        # Check if API is running
        print("🔍 Checking API connection...")
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is running!\n")
        else:
            print("⚠️  API responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API!")
        print("   Make sure the API is running:")
        print("   > cd Python/server")
        print("   > python app.py")
        sys.exit(1)
    
    try:
        # Load real data
        df, feature_columns = load_real_data()
        
        # Run tests
        test_with_confirmed_exoplanet(df, feature_columns)
        test_with_false_positive(df, feature_columns)
        test_with_candidate(df, feature_columns)
        test_batch_prediction(df, feature_columns)
        test_accuracy_on_subset(df, feature_columns, n_samples=50)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED!")
        print("="*60 + "\n")
        
    except FileNotFoundError:
        print("❌ ERROR: Could not find kepler.csv file")
        print("   Make sure you're running this from Python/server directory")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
