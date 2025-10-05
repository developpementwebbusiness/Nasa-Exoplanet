import chardet
import os

file_path = os.path.join('utils', 'STARAI.py')

# Détecter l'encodage
with open(file_path, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(f"Encodage détecté: {result}")

    # Si l'encodage n'est pas UTF-8, convertir le fichier
    if result['encoding'] and result['encoding'].lower() != 'utf-8':
        # Lire avec l'encodage détecté
        content = raw_data.decode(result['encoding'])
        
        # Créer une sauvegarde
        backup_path = file_path + '.bak'
        with open(backup_path, 'wb') as backup:
            backup.write(raw_data)
        print(f"Sauvegarde créée: {backup_path}")
        
        # Réécrire en UTF-8
        with open(file_path, 'w', encoding='utf-8') as out:
            out.write(content)
        print(f"Fichier converti en UTF-8: {file_path}")
    else:
        print("Le fichier est déjà en UTF-8")