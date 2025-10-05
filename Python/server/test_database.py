from utils.database import KVStore
import json

def test_database():
    # Créer une instance de la base de données
    print("🔵 Création de la base de données...")
    db = KVStore("test_store.db")

    # Test 1: Ajouter des données
    print("\n🔵 Test 1: Ajout de données")
    user1 = {
        "name": "Alice",
        "age": 30,
        "preferences": ["python", "data science"]
    }
    db.put("user:1", user1)
    print(f"✅ Ajouté: {json.dumps(user1, indent=2)}")

    # Test 2: Lire des données
    print("\n🔵 Test 2: Lecture de données")
    retrieved = db.get("user:1")
    print(f"✅ Lu: {json.dumps(retrieved, indent=2)}")

    # Test 3: Vérifier l'existence
    print("\n🔵 Test 3: Vérification d'existence")
    exists = db.exists("user:1")
    print(f"✅ user:1 existe? {exists}")
    print(f"✅ user:2 existe? {db.exists('user:2')}")

    # Test 4: Upsert avec vérification
    print("\n🔵 Test 4: Upsert avec vérification")
    # Premier essai - nouvelle clé
    exists, value = db.upsert_with_check("user:2", {"name": "Bob", "age": 25})
    print(f"✅ Premier essai - Existait déjà? {exists}")
    print(f"✅ Valeur: {json.dumps(value, indent=2)}")
    
    # Deuxième essai - clé existante
    exists, value = db.upsert_with_check("user:2", {"name": "Charlie", "age": 35})
    print(f"✅ Deuxième essai - Existait déjà? {exists}")
    print(f"✅ Valeur: {json.dumps(value, indent=2)}")

    # Test 5: Lecture répétée de la même donnée
    print("\n🔵 Test 5: Lectures multiples de la même donnée")
    test_data = {
        "name": "Eve",
        "score": 95,
        "tags": ["astronomy", "exoplanets"],
        "metadata": {
            "created_at": "2025-10-05",
            "version": "1.0"
        }
    }
    
    # Stocker la donnée
    print("✅ Stockage de la donnée de test...")
    db.put("test:persistent", test_data)
    
    # Lire la même donnée plusieurs fois
    print("✅ Première lecture:")
    first_read = db.get("test:persistent")
    print(json.dumps(first_read, indent=2))
    
    print("\n✅ Deuxième lecture:")
    second_read = db.get("test:persistent")
    print(json.dumps(second_read, indent=2))
    
    print(f"\n✅ Les deux lectures sont identiques? {first_read == second_read}")
    
    # Test 6: Lister les clés
    print("\n🔵 Test 6: Liste des clés")
    keys = db.keys()
    print(f"✅ Clés dans la base: {keys}")

    # Test 7: Supprimer une entrée
    print("\n🔵 Test 6: Suppression")
    deleted = db.delete("user:1")
    print(f"✅ user:1 supprimé? {deleted}")
    print(f"✅ Nouvelle liste des clés: {db.keys()}")

    # Fermer proprement
    db.close()
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    test_database()