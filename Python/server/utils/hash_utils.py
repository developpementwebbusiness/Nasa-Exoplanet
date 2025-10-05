import bcrypt

def hash_password(password: str) -> bytes:
    """
    Hash a password using bcrypt
    
    Args:
        password (str): The plain text password
        
    Returns:
        bytes: The hashed password
    """
    # Convert the password to bytes
    password_bytes = password.encode('utf-8')
    
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    return hashed

def verify_password(plain_password: str, hashed_password: bytes) -> bool:
    """
    Verify a password against its hash
    
    Args:
        plain_password (str): The password to check
        hashed_password (bytes): The stored hash to check against
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password
    )

# Example usage
if __name__ == "__main__":
    # Example 1: Hash a password
    password = "mon_mot_de_passe_123"
    hashed = hash_password(password)
    print(f"\nPassword: {password}")
    print(f"Hashed : {hashed}")
    
    # Example 2: Verify correct password
    is_valid = verify_password(password, hashed)
    print(f"\nCorrect password valid? {is_valid}")  # Should print True
    
    # Example 3: Verify wrong password
    wrong_password = "mauvais_mot_de_passe"
    is_valid = verify_password(wrong_password, hashed)
    print(f"Wrong password valid? {is_valid}")  # Should print False
    
    # Example 4: Hash is different each time (due to random salt)
    print("\nMultiple hashes of same password:")
    for _ in range(3):
        print(hash_password(password))