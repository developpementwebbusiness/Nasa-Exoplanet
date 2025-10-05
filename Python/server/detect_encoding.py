import os
import chardet
from pathlib import Path

def detect_file_encoding(file_path):
    """Détecte l'encodage d'un fichier."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def scan_directory(directory):
    """Scanne un répertoire pour trouver les fichiers Python avec un encodage non-UTF8."""
    print(f"🔍 Scan du répertoire: {directory}")
    problematic_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    encoding = detect_file_encoding(file_path)
                    if encoding and encoding.lower() != 'utf-8':
                        print(f"⚠️  Fichier non-UTF8 détecté: {file_path}")
                        print(f"   Encodage détecté: {encoding}")
                        problematic_files.append((file_path, encoding))
                    else:
                        print(f"✅ {file_path} - UTF-8")
                except Exception as e:
                    print(f"❌ Erreur lors de la lecture de {file_path}: {str(e)}")
    
    return problematic_files

def convert_to_utf8(file_path, source_encoding):
    """Convertit un fichier vers UTF-8."""
    print(f"\n🔄 Conversion de {file_path} vers UTF-8")
    try:
        # Lire le contenu avec l'encodage source
        with open(file_path, 'r', encoding=source_encoding) as file:
            content = file.read()
        
        # Sauvegarder une copie de backup
        backup_path = str(file_path) + '.bak'
        Path(file_path).rename(backup_path)
        print(f"✅ Backup créé: {backup_path}")
        
        # Écrire le nouveau fichier en UTF-8
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"✅ Fichier converti avec succès")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la conversion: {str(e)}")
        # Restaurer le backup en cas d'erreur
        if os.path.exists(backup_path):
            Path(backup_path).rename(file_path)
            print("✅ Fichier original restauré")
        return False

def main():
    directory = os.getcwd()
    print("🚀 Détection des fichiers Python non-UTF8...")
    problematic_files = scan_directory(directory)
    
    if not problematic_files:
        print("\n✅ Tous les fichiers Python sont en UTF-8!")
        return
    
    print(f"\n📝 {len(problematic_files)} fichiers non-UTF8 trouvés:")
    for i, (file_path, encoding) in enumerate(problematic_files, 1):
        print(f"{i}. {file_path} ({encoding})")
    
    response = input("\n❓ Voulez-vous convertir ces fichiers en UTF-8? (o/N): ")
    if response.lower() == 'o':
        for file_path, encoding in problematic_files:
            convert_to_utf8(file_path, encoding)

if __name__ == "__main__":
    main()