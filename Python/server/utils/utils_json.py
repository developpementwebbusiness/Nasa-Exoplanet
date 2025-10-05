import json
from utils.hash_utils import hash_password, verify_password
# Pour hacher un mot de passe
password = "mon_mot_de_passe_123"
hashed = hash_password(password)

# Pour vérifier un mot de passe
if verify_password(password, hashed):
    print("Mot de passe correct!")
else:
    print("Mot de passe incorrect!")

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
def convert(data):
    data_output = []
    for element in data["data"]:
        data_output.append(element.values())
    return data_output

def output_json(data_input,data_output):
    data_final_output = {}
    list_hash = [hash_password(str(element)) for element in data_input["data"]]
    for i in range(len(data_output[0])):
        data_final_output[list_hash[i]] = {"score":data_output[0][i],"labels":data_output[1][i]}
    return data_final_output
