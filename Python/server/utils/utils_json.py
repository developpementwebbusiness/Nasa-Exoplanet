from utils.hash_utils import calculate_hash

#python_object = {"name":"Axel"}
def convert(data):
    data_output = []
    list_hash = [calculate_hash(str(element)) for element in data["data"]]
    list_name = [element["name"] for element in data["data"]]
    for element in data["data"]:
        data_output.append(element.values())
    return data_output,list_hash,list_name

def output_json(data_output,list_hash,list_name):
    data_final_output = []
    for i in range(len(data_output[0])):
        name = list_name[i]
        if name == "":
            name = list_hash[i]
        data_final_output.append({"name":name,"score":data_output[0][i],"labels":data_output[1][i]})
    return data_final_output
