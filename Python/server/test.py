import requests
import json

BASE_URL = "http://localhost:8000"
PREDICT_BATCH_URL = f"{BASE_URL}/predict_batch"
TEST_URL = f"{BASE_URL}/test_prediction"

# Vos données d'exoplanètes (TempUp et TempDown seront ignorés)
values_data = {
    "data": [
        {'OrbitalPeriod': '11.521446064', 'OPup': '1.9800000e-06', 'OPdown': '-1.9800000e-06', 'TransEpoch': '170.8396880', 'TEup': '1.310000e-04', 'TEdown': '-1.310000e-04', 'Impact': '2.4830', 'ImpactUp': '2.8510', 'ImpactDown': '-0.6730', 'TransitDur': '3.63990', 'DurUp': '0.01140', 'DurDown': '-0.01140', 'TransitDepth': '1.7984e+04', 'DepthUp': '3.190e+01', 'DepthDown': '-3.190e+01', 'PlanetRadius': '150.51', 'RadiusUp': '3.976e+01', 'RadiusDown': '-1.331e+01', 'EquilibriumTemp': '753.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '75.88', 'InsolationUp': '58.89', 'InsolationDown': '-19.99', 'TransitSNR': '622.10', 'StellarEffTemp': '5795.00', 'SteffUp': '155.00', 'SteffDown': '-172.00', 'StellarLogG': '4.554', 'LogGUp': '0.033', 'LogGDown': '-0.176', 'StellarRadius': '0.8480', 'SradUp': '0.2240', 'SradDown': '-0.0750', 'RA': '297.079930', 'Dec': '47.597401', 'KeplerMag': '15.472'},
        {'OrbitalPeriod': '19.403937760', 'OPup': '2.0680000e-05', 'OPdown': '-2.0680000e-05', 'TransEpoch': '172.4842530', 'TEup': '8.420000e-04', 'TEdown': '-8.420000e-04', 'Impact': '0.8040', 'ImpactUp': '0.0070', 'ImpactDown': '-0.0050', 'TransitDur': '12.21550', 'DurUp': '0.05980', 'DurDown': '-0.05980', 'TransitDepth': '8.9187e+03', 'DepthUp': '5.330e+01', 'DepthDown': '-5.330e+01', 'PlanetRadius': '7.18', 'RadiusUp': '7.600e-01', 'RadiusDown': '-6.800e-01', 'EquilibriumTemp': '523.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '17.69', 'InsolationUp': '6.66', 'InsolationDown': '-4.88', 'TransitSNR': '214.70', 'StellarEffTemp': '5043.00', 'SteffUp': '151.00', 'SteffDown': '-151.00', 'StellarLogG': '4.591', 'LogGUp': '0.072', 'LogGDown': '-0.048', 'StellarRadius': '0.6800', 'SradUp': '0.0720', 'SradDown': '-0.0650', 'RA': '289.258210', 'Dec': '47.635319', 'KeplerMag': '15.487'},
        {'OrbitalPeriod': '19.221388942', 'OPup': '1.1230000e-06', 'OPdown': '-1.1230000e-06', 'TransEpoch': '184.5521637', 'TEup': '4.500000e-05', 'TEdown': '-4.500000e-05', 'Impact': '1.0650', 'ImpactUp': '0.0310', 'ImpactDown': '-0.0340', 'TransitDur': '4.79843', 'DurUp': '0.00235', 'DurDown': '-0.00235', 'TransitDepth': '7.4284e+04', 'DepthUp': '2.190e+01', 'DepthDown': '-2.190e+01', 'PlanetRadius': '49.29', 'RadiusUp': '1.603e+01', 'RadiusDown': '-5.000e+00', 'EquilibriumTemp': '698.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '55.97', 'InsolationUp': '54.55', 'InsolationDown': '-16.40', 'TransitSNR': '2317.00', 'StellarEffTemp': '6117.00', 'SteffUp': '182.00', 'SteffDown': '-200.00', 'StellarLogG': '4.496', 'LogGUp': '0.052', 'LogGDown': '-0.208', 'StellarRadius': '0.9470', 'SradUp': '0.3080', 'SradDown': '-0.0960', 'RA': '295.814540', 'Dec': '47.690350', 'KeplerMag': '15.341'}
    ],
    "user_id": "test_sans_tempup_tempdown"
}

def test_api_complet():
    print("🚀 Test API avec 35 features (sans TempUp/TempDown)")
    print("=" * 60)
    
    # Test 1: Endpoint de test simple
    print("\n🧪 Test endpoint simple...")
    try:
        resp = requests.get(TEST_URL)
        if resp.status_code == 200:
            result = resp.json()
            print(f"   ✅ Test simple OK")
            print(f"   📊 Résultat: {result}")
        else:
            print(f"   ❌ Test simple failed: {resp.text}")
            return False
    except Exception as e:
        print(f"   ❌ Erreur connexion: {e}")
        return False
    
    # Test 2: Prédiction batch avec vos données
    print(f"\n📦 Test batch prediction...")
    print(f"   📊 {len(values_data['data'])} exoplanètes à traiter")
    print(f"   🎯 Note: TempUp et TempDown sont ignorés (étaient None de toute façon)")
    
    try:
        resp = requests.post(PREDICT_BATCH_URL, json=values_data)
        
        if resp.status_code == 200:
            result = resp.json()
            print("   ✅ Batch prediction réussie!")
            print(f"   📈 Nombre de résultats: {len(result.get('data', []))}")
            
            # Afficher chaque résultat
            for i, pred in enumerate(result.get("data", [])):
                print(f"      🌟 Exoplanète {i+1}:")
                print(f"         - Name: {pred.get('name', 'N/A')}")
                print(f"         - Score: {pred.get('score', 'N/A')}")
                print(f"         - Label: {pred.get('label', 'N/A')}")
            
            return True
            
        else:
            print(f"   ❌ Erreur {resp.status_code}")
            try:
                error_detail = resp.json()
                print(f"   📝 Détail: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   📝 Détail: {resp.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception lors du test batch: {e}")
        return False

def afficher_mapping_features():
    """Affiche le mapping des 35 features utilisés"""
    print("\n📋 Mapping des 35 features utilisés:")
    print("=" * 60)
    
    features_utilises = [
        'OrbitalPeriod', 'OPup', 'OPdown', 'TransEpoch', 'TEup', 'TEdown',
        'Impact', 'ImpactUp', 'ImpactDown', 'TransitDur', 'DurUp', 'DurDown',
        'TransitDepth', 'DepthUp', 'DepthDown', 'PlanetRadius', 'RadiusUp', 'RadiusDown',
        'EquilibriumTemp', 'InsolationFlux', 'InsolationUp', 'InsolationDown',
        'TransitSNR', 'StellarEffTemp', 'SteffUp', 'SteffDown', 'StellarLogG', 'LogGUp', 'LogGDown',
        'StellarRadius', 'SradUp', 'SradDown', 'RA', 'Dec', 'KeplerMag'
    ]
    
    features_ignores = ['TempUp', 'TempDown']
    
    print("✅ Features utilisés (35):")
    for i, feature in enumerate(features_utilises, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n❌ Features ignorés ({len(features_ignores)}):")
    for feature in features_ignores:
        print(f"   - {feature} (était None dans vos données)")
    
    print(f"\n📊 Total: {len(features_utilises)} features → Compatible avec StandardScaler")

def main():
    """Fonction principale"""
    # Afficher le mapping
    afficher_mapping_features()
    
    # Tester l'API
    success = test_api_complet()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Tous les tests ont réussi!")
        print("   Votre API fonctionne avec 35 features comme attendu par le StandardScaler.")
    else:
        print("⚠️  Des problèmes persistent, vérifiez les logs de l'API.")

if __name__ == "__main__":
    main()