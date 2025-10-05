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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

#-----------------------------------------------------------------------------------------------------------------------
#Data import

columnnames = ['Confirmation','OrbitalPeriod','PlanetRadius','InsolationFlux','EquilibriumTemp','StellarEffectiveTemp','StellarRadius','RA','Dec']

filepath = 'ExoPlanetHarmony.csv' #ex: KOI/TOI/K2 cumulative table data csv file path
fileKOI = 'KOIHarmony.csv'
fileK2 = 'K2Harmony.csv'
fileTOI = 'TOIHarmony.csv'

dfk = pd.read_csv(fileKOI, skiprows=30,usecols=[0,1,4,7,10,13,19,22,23]) #f1=0.79
dfk.columns = columnnames
dfk2 = pd.read_csv(fileK2, skiprows=17,usecols=[0,1,2,3,4,5,6,8,10]) #bad ratio
dfk2.columns = columnnames
dft = pd.read_csv(fileTOI, skiprows=35,usecols=[0,1,5,9,13,17,21,26,28]) #0.57
dft.columns = columnnames

binary_replace = {'CANDIDATE':'True', 
                  'FALSE POSITIVE': 'False', 
                  'NOT DISPOSITIONED': 'False', 
                  'CONFIRMED': 'True',
                  'REFUTED': 'False',
                  'APC': 'False',
                  'CP': 'True',
                  'FP': 'False',
                  'FA': 'False',
                  'KP': 'True',
                  'PC': 'True'}


df = pd.read_csv(filepath)

df = df.applymap(lambda x: binary_replace.get(x, x) if isinstance(x, str) else x)

#-----------------------------------------------------------------------------------------------------------------------
#Data set-up


features = ['OrbitalPeriod','PlanetRadius','InsolationFlux','EquilibriumTemp','StellarEffectiveTemp','StellarRadius','RA','Dec']   # replace with your numeric columns that you want to keep
label_col = 'Confirmation'                   # replace with your target column that you want your model to predict

# encode text labels to integers (0..K-1)
le = LabelEncoder()
Y = le.fit_transform(df[label_col].values)   # keep `le` to invert later

# numeric features -> StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values.astype(np.float32)) #(value-moyenne)/ecart-type to center data and get rid of unit

joblib.dump(scaler, "scaler.pkl") # save the scaler for later use
joblib.dump(le, "label_encoder.pkl") # save the label encoder for later use

#-----------------------------------------------------------------------------------------------------------------------
#Data split

# split data into train (70%), temp (30%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=42,stratify=Y) # 70% XYtrain, 30% XYtemp, 42 for reproducibility (imagine a minecraft seed)


#stratify=Y to keep same class proportions in each split (ex: if 10% of the data is True, and 90% is False, we want the same proportions in train, val and test)

# split temp into val (15%) and test (15%)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42,stratify=Y_temp) # 15% XYval, 15% XYtest (Cuz 50% of 30% is 15%)

#70% training data, 15% validation data, 15% test data

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)   # long for CrossEntropyLoss
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
Y_test  = torch.tensor(Y_test, dtype=torch.long)

#DataLoaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True) #feed data in batches of 64, shuffle to randomize order each epoch
val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=128, shuffle=False)

#-----------------------------------------------------------------------------------------------------------------------
#Model definition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed=42): #setting seed for reproducibility on all libraries
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(67)

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
    input_dim=X.shape[1], #number of features in input data
    hidden=[1028,512,256,128,64],  #size of hidden layers, can be changed
    num_classes=len(le.classes_) #number of output classes (ex: exoplanet, false positive, candidate)
    ).to(DEVICE)

#-----------------------------------------------------------------------------------------------------------------------
#Weights

class_counts = np.bincount(Y)    # numpy counts the occurrences of each class in Y with position indicating the class 
class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(DEVICE) # gives more weight to minority classes to help the model learn them better
class_weights = class_weights / class_weights.sum() * len(class_weights) # normalize the weights around 1

#When computing the loss, misclassifying a minority class will incur a higher penalty than misclassifying a majority class

criterion = nn.CrossEntropyLoss(weight=class_weights)   #multiplies the loss of each class by the weights defined previously
optimizer = optim.Adam(model.parameters(), lr=1e-3)  #Adam optimizer for weight adjustment, lr=learning rate

#-----------------------------------------------------------------------------------------------------------------------
#Training functions

def train_one_epoch(model, loader, optimizer, criterion):
    #model: the neural network,loader: DataLoader for training data (ex: train_loader), optimizer: optimization algorithm, criterion: loss function

    model.train() #set model to training mode (enables dropout, etc)

    running_loss = 0.0 #to accumulate loss over the epoch

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad() #reset gradients from previous step for gradient descent
        logits = model(Xb) #passes the batch through the model to get raw output scores (logits), shape: (batch, num_classes)
        loss = criterion(logits, yb) #compute loss between logits and true labels using CrossEntropyLoss function
        loss.backward() #backpropagation to compute gradients of loss w.r.t. model parameters
        optimizer.step() #update model parameters using computed gradients and Adam optimizer
        running_loss += loss.item() * Xb.size(0) #accumulate loss, scaled by batch size
    return running_loss / len(loader.dataset) #average loss over the epoch

def evaluate(model, loader, criterion):

    model.eval() #set model to evaluation mode (disables dropout, etc)

    running_loss = 0.0 #to accumulate loss over the evaluation

    preds, trues = [], [] #to store predictions and true labels for accuracy calculation

    with torch.no_grad(): #to avoid computing gradients during evaluation 
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE) 
            logits = model(Xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * Xb.size(0) #accumulate loss, scaled by batch size
            pred = logits.argmax(dim=1).cpu().numpy() #get predicted class by taking argmax of logits (dim=1 means that we take the max across rows)
            #cpu because numpy can't handle cuda tensors
            preds.append(pred); trues.append(yb.cpu().numpy()) #store predictions and true labels
    preds = np.concatenate(preds); trues = np.concatenate(trues) #sqish lists into single arrays
    acc = accuracy_score(trues, preds) #compute accuracy
    return running_loss / len(loader.dataset), acc #average loss and accuracy over the evaluation