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
    nan_mask = np.isnan(X_nu)
    X_nu[nan_mask] = 0.0
    X_scaled = scaler.transform(X_nu)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    mask_tensor = torch.isnan(X_tensor)

    with torch.no_grad():
        logits = model(X_tensor) #unprocessed  output data
        probs = F.softmax(logits,dim=1) #probability convert
        preds = logits.argmax(dim=1).numpy() #gives max
        labels = le.inverse_transform(preds) #transforms the data into True or F*alse
        prob_scores = probs.max(dim=1).values.numpy()

    return labels,prob_scores

'''
rows = [[
    np.float64(2.701287801),
    np.float64(5.634e-06),
    np.float64(-5.634e-06),
    np.float64(170.69579),
    np.float64(0.00164),
    np.float64(-0.00164),
    np.float64(1.016),
    np.float64(3.967),
    np.float64(-0.059),
    np.float64(3.3827),
    np.float64(0.0892),
    np.float64(-0.0892),
    np.float64(860.4),
    np.float64(20.9),
    np.float64(-20.9),
    np.float64(6.54),
    np.float64(1.64),
    np.float64(None),
    np.float64(1222.0),
    np.float64(527.81),
    np.float64(411.33),
    np.float64(-166.35),
    np.float64(48.3),
    np.float64(None),
    np.float64(182.0),
    np.float64(-182.0),
    np.float64(4.468),
    np.float64(None),
    np.float64(-0.169),
    np.float64(0.91),
    np.float64(0.228),
    np.float64(-0.105),
    np.float64(297.92365),
    np.float64(48.777321),
    np.float64(15.54)
]]



print(predict_rows(rows))
'''