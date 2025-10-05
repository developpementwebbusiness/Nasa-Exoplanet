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

#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #pour installer pytorch avec cuda (gpu) si besoin

os.chdir('C:/Coding/PythonScripts/Data') #c pour mon pc ça, change de directoire si besoin

#https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=resources lien du challenge avec les datasets

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


'''
imputer = IterativeImputer()
df[features] = imputer.fit_transform(df[features])
'''

print('shape ',df.shape[0])
df = df[features + [label_col]].dropna() # keep only relevant columns without NaN values
print('shape dropped',df.shape[0])

# encode text labels to integers (0..K-1)
le = LabelEncoder()
Y = le.fit_transform(df[label_col].values)   # keep `le` to invert later

# numeric features -> StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values.astype(np.float32)) #(value-moyenne)/ecart-type to center data and get rid of unit

joblib.dump(scaler, "scaler.pkl") # save the scaler for later use
joblib.dump(le, "label_encoder.pkl") # save the label encoder for later use

print("pd serieis",pd.Series(Y).value_counts())


#-----------------------------------------------------------------------------------------------------------------------
#Data split

# split data into train (70%), temp (30%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=42,stratify=Y) # 70% XYtrain, 30% XYtemp, 42 for reproducibility (imagine a minecraft seed)


#stratify=Y to keep same class proportions in each split (ex: if 10% of the data is True, and 90% is False, we want the same proportions in train, val and test)

# split temp into val (15%) and test (15%)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42,stratify=Y_temp) # 15% XYval, 15% XYtest (Cuz 50% of 30% is 15%)

#ONLY KEEP STRATIFY FOR CLASSIFICATION AI

#70% training data, 15% validation data, 15% test data

#training set is used for gradient descent and weight updates (watch 3Blue1Brown video on youtube if you don't know what that means)
#validation set is used to evaluate the model during training (ex: after each epoch) to tune hyperparameters and avoid overfitting (to avoid the model just memorizing patterns in the training data)
#test set is used to evaluate the model after training (to see how well it generalizes to unseen data)

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)   # long for CrossEntropyLoss
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
Y_test  = torch.tensor(Y_test, dtype=torch.long)

#long is a datatype in for integers 

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
#Training functions

# If classes are imbalanced compute weights:
class_counts = np.bincount(Y)    # numpy counts the occurrences of each class in Y
class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(DEVICE) #gives more weight to minority classes to help the model learn them better
class_weights = class_weights / class_weights.sum() * len(class_weights)
#When computing the loss, misclassifying a minority class will incur a higher penalty than misclassifying a majority class

criterion = nn.CrossEntropyLoss(weight=class_weights)   #multiplies the loss of each class by its weight
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Adam optimizer for weight adjustment, lr=learning rate

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


#-----------------------------------------------------------------------------------------------------------------------
#Model training and evaluation

# -----------------------------
# 1) Initialize best validation loss
# -----------------------------

best_val_loss = float("inf")  # start with "infinity" so any real loss will be smaller

# -----------------------------
# 2) Training loop over epochs
# -----------------------------

for epoch in range(1, 201):   # 30 epochs example
    # --- Train on all batches in the training set ---
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion) 
    
    # --- Evaluate on validation set (no weight updates) ---
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    # --- Check if this is the best model so far ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model's weights to disk
        torch.save(model.state_dict(), "STAR_AI.pth")   # checkpoint
    
    # --- Print progress for this epoch ---
    print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# -----------------------------
# 3) Load the best model after training
# -----------------------------

# This ensures we use the model that performed best on validation data
model.load_state_dict(torch.load("STAR_AI.pth", map_location=DEVICE))

# -----------------------------
# 4) Evaluate on test data
# -----------------------------

model.eval()  #same idea as the other times we called our Datakoaders
preds, trues = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        logits = model(Xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred) ; trues.append(yb.cpu().numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)

# -----------------------------
# 5) Classification metrics
# -----------------------------

from sklearn.metrics import classification_report
labels = np.unique(trues)
print(classification_report(trues, preds, labels=labels, target_names=le.inverse_transform(labels)))

# -----------------------------
# 6) Predicting on a new single row
# -----------------------------

exlist = [54.4183827,2.83,443.0,9.11,5455.0,0.927,291.93423,48.141651]

# 1) Prepare new row features in the same order as training features
row = np.array([exlist], dtype=np.float32)

# 2) Scale using the saved StandardScaler (same as training)
row_scaled = scaler.transform(row)

# 3) Convert to PyTorch tensor and move to correct device
t = torch.tensor(row_scaled, dtype=torch.float32).to(DEVICE)

# 4) Forward pass in evaluation mode
model.eval()
with torch.no_grad():
    logits = model(t)                    # get raw outputs
    pred = logits.argmax(dim=1).cpu().item()  # predicted class index
    human_label = le.inverse_transform([pred])[0]  # convert back to original class name

# 5) Print the prediction
print("Predicted class:", human_label)

#----------------------------------------------------------------------
#HOW TO INTERPRET THE DATA

#EX: This is a sickit-learn classification report
'''

                  precision   recall   f1-score   support

     CANDIDATE       0.38      0.55      0.45       282    
     CONFIRMED       0.55      0.71      0.62       411     
FALSE POSITIVE       0.86      0.55      0.67       687

      accuracy                           0.60      1380
     macro avg       0.60      0.60      0.58      1380
  weighted avg       0.67      0.60      0.61      1380

precision = TP/(TP+FP) “When I say it’s positive, how often am I right?”
recall = TP/(TP+FN) “Of all the positives out there, how many did I catch?”
f1 score = f1 score = 2 * (precision * recall) / (precision + recall) (0.7> is good) harmonic mean good for class imbalanced data-sets
support = nb of items
'''

#-----------------------------------------------------------------------
#WHAT IS WEIGHT AND BIAS

'''

For a particular neuron activation function of a neuron on the first hidden layer: (x,y) -> x layer yth element

a1,i = f(w0,0*a0,1+...+w0,n*a0,n+b0)

'''


#-----------------------------------------------------------------------
#WHAT IS THE COST FUNCTION

'''

It adds up the squares of the differences between each of the trash output activations and the values you want them to have

''' 

#-----------------------------------------------------------------------
#WHAT IS A SEED, AND WHY IS IT IMPORTANT??
'''
The seed is responsible for determining where the AI's gradient descent will start.

'''