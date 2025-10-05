import json

def load_json(json_fname):
    try:
        with open(json_fname,"r",encoding="utf-8") as file:
            file_json = json.load(file)
            file.close()
            return file_json
    except FileNotFoundError:
        print("Fichier non trouvé")
        return None
    except json.JSONDecodeError:
        print("Imossible de décoder le fichier")
        return None
    finally:
        if file:
            file.close()
            print("Fichier fermé avec succés")

#test = load_json("data/data_1.json")
#print(test)

def write_json(json_fname, python_object):
    try:
        with open(json_fname,"w",encoding="utf-8") as file:
            json.dump(python_object,file,indent = 4)
            file.close()
    except FileNotFoundError:
        print("Fichier non trouvé")
        return None
    except PermissionError:
        print("Vous n'avez pas les permissions d'écrire")
        return None

#python_object = {"name":"Axel"}
#write_json("data/data_2.json",python_object)