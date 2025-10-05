import hashlib

def calculate_hash(data: str, algorithm: str = 'md5') -> str:
    """
    Calcule un hash déterministe d'une chaîne de caractères
    
    Args:
        data (str): La donnée à hasher
        algorithm (str): L'algorithme de hash à utiliser ('sha256', 'sha512', 'md5', etc.)
        
    Returns:
        str: Le hash hexadécimal
    """
    # Convertir la chaîne en bytes
    data_bytes = data.encode('utf-8')
    
    # Créer l'objet hash avec l'algorithme spécifié
    hasher = hashlib.new(algorithm)
    
    # Mettre à jour avec les données
    hasher.update(data_bytes)
    
    # Retourner le hash en hexadécimal
    return hasher.hexdigest()

def verify_hash(data: str, expected_hash: str, algorithm: str = 'md5') -> bool:
    """
    Vérifie si une donnée correspond à un hash
    
    Args:
        data (str): La donnée à vérifier
        expected_hash (str): Le hash attendu
        algorithm (str): L'algorithme utilisé pour le hash
        
    Returns:
        bool: True si le hash correspond, False sinon
    """
    actual_hash = calculate_hash(data, algorithm)
    return actual_hash == expected_hash

# Example usage
if __name__ == "__main__":
    # Example 1: Hasher une chaîne
    data = "donnée_à_hasher_123"
    hashed = calculate_hash(data)
    print(f"\nDonnée: {data}")
    print(f"Hash (SHA-256): {hashed}")
    
    # Example 2: Vérifier le hash
    is_valid = verify_hash(data, hashed)
    print(f"\nHash vérifié? {is_valid}")  # Devrait afficher True
    
    # Example 3: Même donnée = même hash
    print("\nMultiples hashs de la même donnée (doivent être identiques):")
    for _ in range(3):
        print(calculate_hash(data))
        
    # Example 4: Différents algorithmes
    print("\nMême donnée avec différents algorithmes:")
    for algo in ['sha256', 'sha512', 'md5']:
        print(f"{algo}: {calculate_hash(data, algo)}")