import requests

BASE_URL = "http://localhost:8000/predict"

# Votre liste de dictionnaires (mettez ici tout votre values_data)
values_data = [
    {'OrbitalPeriod': '11.521446064', 'OPup': '1.9800000e-06', 'OPdown': '-1.9800000e-06', 'TransEpoch': '170.8396880', 'TEup': '1.310000e-04', 'TEdown': '-1.310000e-04', 'Impact': '2.4830', 'ImpactUp': '2.8510', 'ImpactDown': '-0.6730', 'TransitDur': '3.63990', 'DurUp': '0.01140', 'DurDown': '-0.01140', 'TransitDepth': '1.7984e+04', 'DepthUp': '3.190e+01', 'DepthDown': '-3.190e+01', 'PlanetRadius': '150.51', 'RadiusUp': '3.976e+01', 'RadiusDown': '-1.331e+01', 'EquilibriumTemp': '753.0', 'TempUp': None, 'TempDown': None, 'InsolationFlux': '75.88', 'InsolationUp': '58.89', 'InsolationDown': '-19.99', 'TransitSNR': '622.10', 'StellarEffTemp': '5795.00', 'SteffUp': '155.00', 'SteffDown': '-172.00', 'StellarLogG': '4.554', 'LogGUp': '0.033', 'LogGDown': '-0.176', 'StellarRadius': '0.8480', 'SradUp': '0.2240', 'SradDown': '-0.0750', 'RA': '297.079930', 'Dec': '47.597401', 'KeplerMag': '15.472'},
    # ... ajoutez les autres dictionnaires ici
]

for i, row in enumerate(values_data):
    # Extraire les 4 features du dict en float
    features = [
        float(row['OrbitalPeriod']), 
        float(row['OPup']), 
        float(row['OPdown']), 
        float(row['TransEpoch'])
    ]
    payload = {
        "features": features,
        "user_id": f"exo_{i+1}"
    }
    resp = requests.post(BASE_URL, json=payload)
    print(f"Exoplanet {i+1} status: {resp.status_code}")
    if resp.status_code == 200:
        print("Réponse :", resp.json())
    else:
        print("Erreur :", resp.text)
