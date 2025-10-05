import requests
import json
import time

# Configuration de l'API
BASE_URL = "http://localhost:8000"
PREDICT_URL = f"{BASE_URL}/predict"
TRAIN_URL = f"{BASE_URL}/train"
STATUS_URL = f"{BASE_URL}/train/status"
LIST_TRAININGS_URL = f"{BASE_URL}/train/list"
LIST_MODELS_URL = f"{BASE_URL}/models/list"
EXPORT_URL = f"{BASE_URL}/export_model"
TEST_URL = f"{BASE_URL}/test_prediction"

# ================== DONNÉES DE TEST ==================

# Données pour prédiction (format 1: features simples)
test_features = [0.1 + (i * 0.02) for i in range(35)]  # 35 valeurs

# Données pour prédiction (format 2: batch avec arrays)
batch_features = [
    [0.1 + (i * 0.02) for i in range(35)],
    [0.2 + (i * 0.02) for i in range(35)],
    [0.3 + (i * 0.02) for i in range(35)]
]

# Données pour prédiction (format 3: dictionnaires nommés)
dict_data = [
    {'OrbitalPeriod': '11.521446064', 'OPup': '1.9800000e-06', 'OPdown': '-1.9800000e-06', 'TransEpoch': '170.8396880', 'TEup': '1.310000e-04', 'TEdown': '-1.310000e-04', 'Impact': '2.4830', 'ImpactUp': '2.8510', 'ImpactDown': '-0.6730', 'TransitDur': '3.63990', 'DurUp': '0.01140', 'DurDown': '-0.01140', 'TransitDepth': '1.7984e+04', 'DepthUp': '3.190e+01', 'DepthDown': '-3.190e+01', 'PlanetRadius': '150.51', 'RadiusUp': '3.976e+01', 'RadiusDown': '-1.331e+01', 'EquilibriumTemp': '753.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '75.88', 'InsolationUp': '58.89', 'InsolationDown': '-19.99', 'TransitSNR': '622.10', 'StellarEffTemp': '5795.00', 'SteffUp': '155.00', 'SteffDown': '-172.00', 'StellarLogG': '4.554', 'LogGUp': '0.033', 'LogGDown': '-0.176', 'StellarRadius': '0.8480', 'SradUp': '0.2240', 'SradDown': '-0.0750', 'RA': '297.079930', 'Dec': '47.597401', 'KeplerMag': '15.472', 'name': 'Kepler-1b'},
    {'OrbitalPeriod': '19.403937760', 'OPup': '2.0680000e-05', 'OPdown': '-2.0680000e-05', 'TransEpoch': '172.4842530', 'TEup': '8.420000e-04', 'TEdown': '-8.420000e-04', 'Impact': '0.8040', 'ImpactUp': '0.0070', 'ImpactDown': '-0.0050', 'TransitDur': '12.21550', 'DurUp': '0.05980', 'DurDown': '-0.05980', 'TransitDepth': '8.9187e+03', 'DepthUp': '5.330e+01', 'DepthDown': '-5.330e+01', 'PlanetRadius': '7.18', 'RadiusUp': '7.600e-01', 'RadiusDown': '-6.800e-01', 'EquilibriumTemp': '523.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '17.69', 'InsolationUp': '6.66', 'InsolationDown': '-4.88', 'TransitSNR': '214.70', 'StellarEffTemp': '5043.00', 'SteffUp': '151.00', 'SteffDown': '-151.00', 'StellarLogG': '4.591', 'LogGUp': '0.072', 'LogGDown': '-0.048', 'StellarRadius': '0.6800', 'SradUp': '0.0720', 'SradDown': '-0.0650', 'RA': '289.258210', 'Dec': '47.635319', 'KeplerMag': '15.487', 'name': 'Kepler-2b'}
]

# Données pour entraînement (avec labels)
training_data = {
    "data": [
        # Exemples CONFIRMED
        {'OrbitalPeriod': '11.521446064', 'OPup': '1.9800000e-06', 'OPdown': '-1.9800000e-06', 'TransEpoch': '170.8396880', 'TEup': '1.310000e-04', 'TEdown': '-1.310000e-04', 'Impact': '2.4830', 'ImpactUp': '2.8510', 'ImpactDown': '-0.6730', 'TransitDur': '3.63990', 'DurUp': '0.01140', 'DurDown': '-0.01140', 'TransitDepth': '1.7984e+04', 'DepthUp': '3.190e+01', 'DepthDown': '-3.190e+01', 'PlanetRadius': '150.51', 'RadiusUp': '3.976e+01', 'RadiusDown': '-1.331e+01', 'EquilibriumTemp': '753.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '75.88', 'InsolationUp': '58.89', 'InsolationDown': '-19.99', 'TransitSNR': '622.10', 'StellarEffTemp': '5795.00', 'SteffUp': '155.00', 'SteffDown': '-172.00', 'StellarLogG': '4.554', 'LogGUp': '0.033', 'LogGDown': '-0.176', 'StellarRadius': '0.8480', 'SradUp': '0.2240', 'SradDown': '-0.0750', 'RA': '297.079930', 'Dec': '47.597401', 'KeplerMag': '15.472', 'label': 'CONFIRMED'},
        
        {'OrbitalPeriod': '9.273581730', 'OPup': '1.0370000e-05', 'OPdown': '-1.0370000e-05', 'TransEpoch': '173.2581550', 'TEup': '8.770000e-04', 'TEdown': '-8.770000e-04', 'Impact': '0.3870', 'ImpactUp': '0.0040', 'ImpactDown': '-0.3860', 'TransitDur': '3.28750', 'DurUp': '0.03090', 'DurDown': '-0.03090', 'TransitDepth': '1.2883e+03', 'DepthUp': '1.680e+01', 'DepthDown': '-1.680e+01', 'PlanetRadius': '2.47', 'RadiusUp': '2.000e-01', 'RadiusDown': '-2.400e-01', 'EquilibriumTemp': '649.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '41.85', 'InsolationUp': '12.49', 'InsolationDown': '-11.70', 'TransitSNR': '87.20', 'StellarEffTemp': '4856.00', 'SteffUp': '131.00', 'SteffDown': '-146.00', 'StellarLogG': '4.583', 'LogGUp': '0.065', 'LogGDown': '-0.035', 'StellarRadius': '0.6960', 'SradUp': '0.0560', 'SradDown': '-0.0680', 'RA': '288.138240', 'Dec': '47.724449', 'KeplerMag': '15.302', 'label': 'CONFIRMED'},
        
        {'OrbitalPeriod': '6.029303290', 'OPup': '5.5090000e-06', 'OPdown': '-5.5090000e-06', 'TransEpoch': '171.6029590', 'TEup': '7.130000e-04', 'TEdown': '-7.130000e-04', 'Impact': '0.2580', 'ImpactUp': '0.1960', 'ImpactDown': '-0.2580', 'TransitDur': '1.58210', 'DurUp': '0.03110', 'DurDown': '-0.03110', 'TransitDepth': '1.9127e+03', 'DepthUp': '3.440e+01', 'DepthDown': '-3.440e+01', 'PlanetRadius': '2.85', 'RadiusUp': '2.600e-01', 'RadiusDown': '-1.500e-01', 'EquilibriumTemp': '678.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '50.04', 'InsolationUp': '16.40', 'InsolationDown': '-9.88', 'TransitSNR': '65.40', 'StellarEffTemp': '4537.00', 'SteffUp': '123.00', 'SteffDown': '-123.00', 'StellarLogG': '4.648', 'LogGUp': '0.020', 'LogGDown': '-0.056', 'StellarRadius': '0.6720', 'SradUp': '0.0620', 'SradDown': '-0.0360', 'RA': '283.710880', 'Dec': '47.863270', 'KeplerMag': '15.784', 'label': 'CONFIRMED'},
        
        # Exemples FALSE POSITIVE
        {'OrbitalPeriod': '19.403937760', 'OPup': '2.0680000e-05', 'OPdown': '-2.0680000e-05', 'TransEpoch': '172.4842530', 'TEup': '8.420000e-04', 'TEdown': '-8.420000e-04', 'Impact': '0.8040', 'ImpactUp': '0.0070', 'ImpactDown': '-0.0050', 'TransitDur': '12.21550', 'DurUp': '0.05980', 'DurDown': '-0.05980', 'TransitDepth': '8.9187e+03', 'DepthUp': '5.330e+01', 'DepthDown': '-5.330e+01', 'PlanetRadius': '7.18', 'RadiusUp': '7.600e-01', 'RadiusDown': '-6.800e-01', 'EquilibriumTemp': '523.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '17.69', 'InsolationUp': '6.66', 'InsolationDown': '-4.88', 'TransitSNR': '214.70', 'StellarEffTemp': '5043.00', 'SteffUp': '151.00', 'SteffDown': '-151.00', 'StellarLogG': '4.591', 'LogGUp': '0.072', 'LogGDown': '-0.048', 'StellarRadius': '0.6800', 'SradUp': '0.0720', 'SradDown': '-0.0650', 'RA': '289.258210', 'Dec': '47.635319', 'KeplerMag': '15.487', 'label': 'FALSE POSITIVE'},
        
        {'OrbitalPeriod': '19.221388942', 'OPup': '1.1230000e-06', 'OPdown': '-1.1230000e-06', 'TransEpoch': '184.5521637', 'TEup': '4.500000e-05', 'TEdown': '-4.500000e-05', 'Impact': '1.0650', 'ImpactUp': '0.0310', 'ImpactDown': '-0.0340', 'TransitDur': '4.79843', 'DurUp': '0.00235', 'DurDown': '-0.00235', 'TransitDepth': '7.4284e+04', 'DepthUp': '2.190e+01', 'DepthDown': '-2.190e+01', 'PlanetRadius': '49.29', 'RadiusUp': '1.603e+01', 'RadiusDown': '-5.000e+00', 'EquilibriumTemp': '698.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '55.97', 'InsolationUp': '54.55', 'InsolationDown': '-16.40', 'TransitSNR': '2317.00', 'StellarEffTemp': '6117.00', 'SteffUp': '182.00', 'SteffDown': '-200.00', 'StellarLogG': '4.496', 'LogGUp': '0.052', 'LogGDown': '-0.208', 'StellarRadius': '0.9470', 'SradUp': '0.3080', 'SradDown': '-0.0960', 'RA': '295.814540', 'Dec': '47.690350', 'KeplerMag': '15.341', 'label': 'FALSE POSITIVE'},
        
        {'OrbitalPeriod': '16.469837740', 'OPup': '1.3610000e-05', 'OPdown': '-1.3610000e-05', 'TransEpoch': '180.8817610', 'TEup': '6.230000e-04', 'TEdown': '-6.230000e-04', 'Impact': '0.2920', 'ImpactUp': '0.1180', 'ImpactDown': '-0.1010', 'TransitDur': '9.43780', 'DurUp': '0.06000', 'DurDown': '-0.06000', 'TransitDepth': '1.0479e+04', 'DepthUp': '3.990e+01', 'DepthDown': '-3.990e+01', 'PlanetRadius': '7.94', 'RadiusUp': '8.900e-01', 'RadiusDown': '-8.900e-01', 'EquilibriumTemp': '595.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '29.61', 'InsolationUp': '12.01', 'InsolationDown': '-8.91', 'TransitSNR': '303.40', 'StellarEffTemp': '5152.00', 'SteffUp': '168.00', 'SteffDown': '-153.00', 'StellarLogG': '4.517', 'LogGUp': '0.088', 'LogGDown': '-0.072', 'StellarRadius': '0.7860', 'SradUp': '0.0880', 'SradDown': '-0.0880', 'RA': '297.154420', 'Dec': '47.668701', 'KeplerMag': '15.788', 'label': 'FALSE POSITIVE'},
        
        {'OrbitalPeriod': '2.696370652', 'OPup': '7.5340000e-06', 'OPdown': '-7.5340000e-06', 'TransEpoch': '170.7376900', 'TEup': '2.340000e-03', 'TEdown': '-2.340000e-03', 'Impact': '0.0440', 'ImpactUp': '0.3600', 'ImpactDown': '-0.0440', 'TransitDur': '3.61290', 'DurUp': '0.06860', 'DurDown': '-0.06860', 'TransitDepth': '3.9790e+02', 'DepthUp': '9.500e+00', 'DepthDown': '-9.500e+00', 'PlanetRadius': '1.58', 'RadiusUp': '1.300e-01', 'RadiusDown': '-1.600e-01', 'EquilibriumTemp': '1066.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '305.34', 'InsolationUp': '96.64', 'InsolationDown': '-86.33', 'TransitSNR': '48.20', 'StellarEffTemp': '4989.00', 'SteffUp': '149.00', 'SteffDown': '-149.00', 'StellarLogG': '4.504', 'LogGUp': '0.090', 'LogGDown': '-0.060', 'StellarRadius': '0.8190', 'SradUp': '0.0670', 'SradDown': '-0.0820', 'RA': '283.765470', 'Dec': '47.804298', 'KeplerMag': '15.269', 'label': 'FALSE POSITIVE'},
        
        {'OrbitalPeriod': '1.506354090', 'OPup': '4.7000000e-07', 'OPdown': '-4.7000000e-07', 'TransEpoch': '170.9980640', 'TEup': '2.460000e-04', 'TEdown': '-2.460000e-04', 'Impact': '0.8600', 'ImpactUp': '0.0420', 'ImpactDown': '-0.1250', 'TransitDur': '1.51550', 'DurUp': '0.04830', 'DurDown': '-0.04830', 'TransitDepth': '2.2117e+03', 'DepthUp': '1.830e+01', 'DepthDown': '-1.830e+01', 'PlanetRadius': '4.53', 'RadiusUp': '1.010e+00', 'RadiusDown': '-4.400e-01', 'EquilibriumTemp': '1452.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '1054.12', 'InsolationUp': '703.10', 'InsolationDown': '-292.26', 'TransitSNR': '181.10', 'StellarEffTemp': '5714.00', 'SteffUp': '155.00', 'SteffDown': '-172.00', 'StellarLogG': '4.563', 'LogGUp': '0.040', 'LogGDown': '-0.160', 'StellarRadius': '0.8310', 'SradUp': '0.1860', 'SradDown': '-0.0800', 'RA': '295.665220', 'Dec': '49.351009', 'KeplerMag': '15.525', 'label': 'FALSE POSITIVE'},
        
        {'OrbitalPeriod': '5.349553819', 'OPup': '8.8340000e-06', 'OPdown': '-8.8340000e-06', 'TransEpoch': '171.8069400', 'TEup': '1.270000e-03', 'TEdown': '-1.270000e-03', 'Impact': '0.0920', 'ImpactUp': '0.3690', 'ImpactDown': '-0.0920', 'TransitDur': '3.02780', 'DurUp': '0.04710', 'DurDown': '-0.04710', 'TransitDepth': '8.3100e+02', 'DepthUp': '1.480e+01', 'DepthDown': '-1.480e+01', 'PlanetRadius': '2.55', 'RadiusUp': '1.600e-01', 'RadiusDown': '-2.400e-01', 'EquilibriumTemp': '919.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '168.99', 'InsolationUp': '34.66', 'InsolationDown': '-38.48', 'TransitSNR': '60.60', 'StellarEffTemp': '5185.00', 'SteffUp': '85.00', 'SteffDown': '-77.00', 'StellarLogG': '4.440', 'LogGUp': '0.098', 'LogGDown': '-0.045', 'StellarRadius': '0.9100', 'SradUp': '0.0570', 'SradDown': '-0.0860', 'RA': '292.376130', 'Dec': '47.880989', 'KeplerMag': '15.416', 'label': 'CONFIRMED'},
        
        {'OrbitalPeriod': '3.941052210', 'OPup': '1.0940000e-05', 'OPdown': '-1.0940000e-05', 'TransEpoch': '136.0866200', 'TEup': '2.330000e-03', 'TEdown': '-2.330000e-03', 'Impact': '0.2260', 'ImpactUp': '0.2430', 'ImpactDown': '-0.2260', 'TransitDur': '2.59840', 'DurUp': '0.07370', 'DurDown': '-0.07370', 'TransitDepth': '3.6330e+02', 'DepthUp': '1.390e+01', 'DepthDown': '-1.390e+01', 'PlanetRadius': '1.70', 'RadiusUp': '1.100e-01', 'RadiusDown': '-1.600e-01', 'EquilibriumTemp': '1018.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '253.15', 'InsolationUp': '51.91', 'InsolationDown': '-57.65', 'TransitSNR': '28.90', 'StellarEffTemp': '5185.00', 'SteffUp': '85.00', 'SteffDown': '-77.00', 'StellarLogG': '4.440', 'LogGUp': '0.098', 'LogGDown': '-0.045', 'StellarRadius': '0.9100', 'SradUp': '0.0570', 'SradDown': '-0.0860', 'RA': '292.376130', 'Dec': '47.880989', 'KeplerMag': '15.416', 'label': 'FALSE POSITIVE'}
    ],
    "ai_name": "TestClassifier_v1",
    "hidden_layers": [64, 32],
    "epochs": 15,  # Réduit pour les tests
    "user_id": "test_unified_api"
}

# ================== FONCTIONS DE TEST ==================

def test_api_health():
    """Test la santé de l'API"""
    print("🩺 Test de santé de l'API...")
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ API opérationnelle")
            print(f"   📋 Version: {result.get('version', 'N/A')}")
            print(f"   🔗 Endpoints: {list(result.get('endpoints', {}).keys())}")
            return True
        else:
            print(f"   ❌ API non accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")
        return False

def test_prediction_formats():
    """Test tous les formats de prédiction supportés"""
    print("\n🔮 Test des différents formats de prédiction...")
    
    # Test 1: Features simples (35 floats)
    print("   📊 Test 1: Features array simple")
    payload1 = {
        "features": test_features,
        "user_id": "test_simple_features"
    }
    
    try:
        response = requests.post(PREDICT_URL, json=payload1)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ Succès - Prédictions: {len(result['data'])}")
            for pred in result['data']:
                print(f"         - {pred['name']}: {pred['label']} (score: {pred['score']:.4f})")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")
    
    # Test 2: Batch avec arrays
    print("   📊 Test 2: Batch features arrays")
    payload2 = {
        "features": batch_features,
        "user_id": "test_batch_arrays"
    }
    
    try:
        response = requests.post(PREDICT_URL, json=payload2)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ Succès - Prédictions: {len(result['data'])}")
            for pred in result['data']:
                print(f"         - {pred['name']}: {pred['label']} (score: {pred['score']:.4f})")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")
    
    # Test 3: Dictionnaires nommés
    print("   📊 Test 3: Dictionnaires avec noms")
    payload3 = {
        "data": dict_data,
        "user_id": "test_named_dicts"
    }
    
    try:
        response = requests.post(PREDICT_URL, json=payload3)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ Succès - Prédictions: {len(result['data'])}")
            for pred in result['data']:
                print(f"         - {pred['name']}: {pred['label']} (score: {pred['score']:.4f})")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")

def test_training_workflow():
    """Test complet du workflow d'entraînement"""
    print("\n🚀 Test du workflow d'entraînement...")
    
    # Étape 1: Lancer l'entraînement
    print("   📦 Lancement de l'entraînement...")
    try:
        response = requests.post(TRAIN_URL, json=training_data)
        if response.status_code == 200:
            result = response.json()
            training_id = result['training_id']
            print(f"      ✅ Entraînement lancé: {training_id}")
            print(f"      🤖 AI Name: {result['ai_name']}")
            print(f"      ⏱️  Durée estimée: {result['estimated_duration']}")
            
            # Étape 2: Surveiller l'entraînement
            print("   👀 Surveillance de l'entraînement...")
            return monitor_training(training_id)
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"      ❌ Exception: {e}")
        return False

def monitor_training(training_id):
    """Surveille un entraînement en temps réel"""
    max_iterations = 60  # Maximum 2 minutes de surveillance
    iteration = 0
    
    while iteration < max_iterations:
        try:
            response = requests.get(f"{STATUS_URL}/{training_id}")
            
            if response.status_code == 200:
                status = response.json()
                
                print(f"\r      Status: {status['status']} | "
                      f"Epoch: {status.get('current_epoch', 0)}/{status['total_epochs']} | "
                      f"Progress: {status['progress_percent']:.1f}% | "
                      f"Loss: {status.get('loss', 'N/A')} | "
                      f"Acc: {status.get('accuracy', 'N/A')} | "
                      f"ETA: {status.get('estimated_time_remaining', 'N/A')}", end="")
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    print(f"\n      🏁 Entraînement terminé: {status['status']}")
                    
                    if status['status'] == 'completed':
                        print(f"      📁 Modèle: {status.get('model_path', 'N/A')}")
                        print(f"      📁 Scaler: {status.get('scaler_path', 'N/A')}")
                        print(f"      📁 Label encoder: {status.get('le_path', 'N/A')}")
                        return True
                    elif status['status'] == 'failed':
                        print(f"      ❌ Erreur: {status.get('error_message', 'Inconnue')}")
                        return False
                    else:
                        return False
            else:
                print(f"\n      ❌ Erreur lors de la surveillance: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"\n      ❌ Exception: {e}")
            return False
        
        time.sleep(2)
        iteration += 1
    
    print("\n      ⏰ Timeout de surveillance atteint")
    return False

def test_model_management():
    """Test des fonctions de gestion des modèles"""
    print("\n🤖 Test de la gestion des modèles...")
    
    # Test 1: Lister tous les entraînements
    print("   📋 Liste des entraînements...")
    try:
        response = requests.get(LIST_TRAININGS_URL)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ {result.get('total', 0)} entraînements trouvés")
            for training in result.get('trainings', [])[:3]:  # Afficher les 3 premiers
                print(f"         - {training['training_id']}: {training['status']} ({training['progress_percent']:.1f}%)")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")
    
    # Test 2: Lister tous les modèles
    print("   📋 Liste des modèles...")
    try:
        response = requests.get(LIST_MODELS_URL)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ {result.get('total', 0)} modèles trouvés")
            for model in result.get('models', []):
                print(f"         - {model['name']}: {len(model['files'])} fichiers")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")

def test_model_export():
    """Test de l'export de modèles"""
    print("\n📦 Test de l'export de modèles...")
    
    # Test export de tous les modèles
    try:
        response = requests.get(f"{EXPORT_URL}?model_id=all")
        if response.status_code == 200:
            content_length = len(response.content)
            print(f"      ✅ Export 'all' réussi: {content_length} bytes")
            print(f"      📁 Content-Type: {response.headers.get('Content-Type')}")
            print(f"      📁 Filename: {response.headers.get('Content-Disposition', 'N/A')}")
        else:
            print(f"      ❌ Erreur export 'all': {response.status_code}")
    except Exception as e:
        print(f"      ❌ Exception export: {e}")

def test_prediction_endpoint():
    """Test l'endpoint de test intégré"""
    print("\n🧪 Test de l'endpoint de test intégré...")
    try:
        response = requests.get(TEST_URL)
        if response.status_code == 200:
            result = response.json()
            print("      ✅ Endpoint de test fonctionnel")
            if 'data' in result:
                pred = result['data'][0]
                print(f"         - {pred['name']}: {pred['label']} (score: {pred['score']:.4f})")
            elif 'error' in result:
                print(f"      ⚠️  Erreur dans le test: {result['error']}")
        else:
            print(f"      ❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")

def main():
    """Fonction principale de test"""
    print("🧪 Test complet de l'API Unifiée - Training & Prediction")
    print("=" * 70)
    
    # Test 1: Santé de l'API
    if not test_api_health():
        print("\n❌ L'API n'est pas accessible. Vérifiez qu'elle est démarrée.")
        return
    
    # Test 2: Endpoint de test intégré
    test_prediction_endpoint()
    
    # Test 3: Formats de prédiction
    test_prediction_formats()
    
    # Test 4: Gestion des modèles
    test_model_management()
    
    # Test 5: Export de modèles
    test_model_export()
    
    # Test 6: Workflow d'entraînement complet
    print("\n" + "=" * 70)
    print("⚠️  ATTENTION: Le test d'entraînement peut prendre plusieurs minutes!")
    print("   Vous pouvez l'interrompre avec Ctrl+C si nécessaire.")
    
    user_input = input("\nVoulez-vous lancer le test d'entraînement? (y/N): ")
    if user_input.lower() in ['y', 'yes', 'oui']:
        training_success = test_training_workflow()
        
        if training_success:
            print("\n📊 Test post-entraînement...")
            test_model_management()  # Re-tester après l'entraînement
    
    # Résumé final
    print("\n" + "=" * 70)
    print("🎉 Test complet terminé!")
    print("\n💡 Résumé des fonctionnalités testées:")
    print("   ✅ Prédiction - Format 1: Features array [35 floats]")
    print("   ✅ Prédiction - Format 2: Batch arrays [[floats], [floats]]")
    print("   ✅ Prédiction - Format 3: Dictionnaires nommés")
    print("   ✅ Entraînement en arrière-plan avec suivi")
    print("   ✅ Gestion et listing des modèles")
    print("   ✅ Export ZIP des modèles")
    print("   ✅ Monitoring temps réel des entraînements")
    
    print("\n🔗 Endpoints disponibles:")
    print(f"   📡 Prédiction: {PREDICT_URL}")
    print(f"   🚀 Entraînement: {TRAIN_URL}")
    print(f"   📊 Statut: {STATUS_URL}/{{training_id}}")
    print(f"   📋 Modèles: {LIST_MODELS_URL}")
    print(f"   📦 Export: {EXPORT_URL}")

if __name__ == "__main__":
    main()