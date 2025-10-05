from database import KVStore
import json

# Créer une instance de la base de données
db = KVStore("test.db")

# Lire et afficher toutes les clés
print("📝 Clés existantes dans la base de données:")
keys = db.keys()
print(f"Trouvé {len(keys)} clés: {keys}\n")

# Pour chaque clé, lire et afficher la valeur
for key in keys:
    value = db.get(key)
    print(f"🔑 Clé: {key}")
    print("📄 Valeur:")
    print(json.dumps(value, indent=2))
    print("-" * 50 + "\n")