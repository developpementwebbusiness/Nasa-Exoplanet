import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class MonModeleIA:
    def __init__(self):
        # Exemple: modèle de classification simple
        self.model = RandomForestClassifier(n_estimators=100)
        # Entraîner avec des données d'exemple
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        self.model.fit(X, y)
    
    def predire(self, donnees):
        """
        Fonction qui fait tourner l'IA sur les données reçues
        """
        # Convertir les données en format numpy
        X = np.array(donnees).reshape(1, -1)
        
        # Prédiction
        prediction = self.model.predict(X)[0]
        probabilites = self.model.predict_proba(X)[0]
        
        return {
            "classe_predite": int(prediction),
            "probabilites": probabilites.tolist(),
            "confiance": float(max(probabilites))
        }
