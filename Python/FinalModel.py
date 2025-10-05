import os
import joblib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #pour installer pytorch avec cuda (gpu) si besoin

os.chdir('C:/Coding/PythonScripts/Data') #c pour mon pc ça, change de directoire si besoin

def intializeAI(rows): #rows is a list
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

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
    input_dim=8, #number of features in input data
    hidden=[1028,512,256,128,64],  #size of hidden layers, can be changed
    num_classes=len(le.classes_) #number of output classes (ex: exoplanet, false positive, candidate)
    )

model.load_state_dict(torch.load("STAR_AI.pth", map_location=torch.device("cpu"))) #to ensure it works even without cpu
model.eval()  # important for evaluation