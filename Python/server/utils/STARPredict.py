import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #pour installer pytorch avec cuda (gpu) si besoin

#Loading scalers used for AI training

scaler = joblib.load("utils/Data/AI/STAR_AI_v2/scaler.pkl") #to scale data the same way
le = joblib.load("utils/Data/AI/STAR_AI_v2/label_encoder.pkl") #to convert predictions back to True and False


class SimpleMLP(nn.Module): #Multi Layer Perceptron subclass of nn.Module

    def __init__(self, input_dim, hidden=[128,64], num_classes=3, dropout=0.2):
        #amount of columns in input data, list of hidden layer sizes, number of output classes, dropout rate #Adapt these hyperparameters depending on your dataset

        super().__init__() #call parent constructor to get nn.Module functionalities
        
        layers = [] 
        in_dim = input_dim #amount of features in input data = number of neurons in input layer (intially)

        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)] 
            #Linear(X) = xW^Transpose + b, W weights, b bias, ReLU(X) = max(0,X) activation function, dropout to avoid overfitting by randomly setting some activations to 0
            #layers is essentially a cycle of Linear -> ReLU -> Dropout, repeated for each hidden layer size in the hidden list

            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))   #final layer without activation 
        self.net = nn.Sequential(*layers) #automatically inputs data through all layer functions in sequence

    def forward(self, x):
        return self.net(x)

model = SimpleMLP(
    input_dim=35, #number of features in input data
    hidden=[128,64],  #size of hidden layers, can be changed
    num_classes=len(le.classes_) #number of output classes (ex: exoplanet, false positive, candidate)
    )

model.load_state_dict(torch.load("utils/Data/AI/STAR_AI_v2/STAR_AI_v2.pth", map_location=torch.device("cpu"))) #to ensure it works even without cpu
model.eval()  # important for evaluation

def predict_rows(rows):

    """
    rows: list of lists, each inner list = one sample with 8 features
    returns: list of predicted labels and conf_score
    """
    
    X_nu = np.array(rows, dtype=np.float32)
    X_scaled = scaler.transform(X_nu)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor) #unprocessed  output data
        probs = F.softmax(logits,dim=1) #probability convert
        preds = logits.argmax(dim=1).numpy() #gives max
        labels = le.inverse_transform(preds) #transforms the data into True or False
        prob_scores = probs.max(dim=1).values.numpy()

    return labels,prob_scores

rows = [[0.12, 0.45, 0.78, 0.34, 0.91, 0.67, 0.23, 0.56, 0.89, 0.10, 0.33, 0.74, 0.58, 0.29, 0.62, 0.85, 0.41, 0.19, 0.93, 0.77, 0.36, 0.64, 0.28, 0.50, 0.81, 0.07, 0.39, 0.66, 0.94, 0.21, 0.59, 0.73, 0.31, 0.48, 0.86],
    [0.03, 0.26, 0.49, 0.72, 0.95, 0.18, 0.41, 0.64, 0.87, 0.10, 0.33, 0.56, 0.79, 0.02, 0.25, 0.48, 0.71, 0.94, 0.17, 0.40, 0.63, 0.86, 0.09, 0.32, 0.55, 0.78, 0.01, 0.24, 0.47, 0.70, 0.93, 0.16, 0.39, 0.62, 0.85],
    [0.14, 0.37, 0.60, 0.83, 0.06, 0.29, 0.52, 0.75, 0.98, 0.11, 0.34, 0.57, 0.80, 0.03, 0.26, 0.49, 0.72, 0.95, 0.18, 0.41, 0.64, 0.87, 0.10, 0.33, 0.56, 0.79, 0.02, 0.25, 0.48, 0.71, 0.94, 0.17, 0.40, 0.63, 0.86]]



print(predict_rows(rows))