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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.experimental import enable_iterative_imputer
import utils.cleaning_library as cl


StandardizedColumnNames = ['Confirmation', 
'OrbitalPeriod', 
'TransEpoch', 
'Dec', 
'RA',
'TransitDur', 
'KeplerMag', 
'InsolationFlux', 
'InsolationUp',
'InsolationDown', 
'StellarRadius', 
'StellarEffTemp', 
'TransitDepth',
'Impact', 
'TransitSNR', 
'EquilibriumTemp', 
'PlanetRadius', 
'RadiusUp',
'RadiusDown', 
'StellarLogG', 
'OPup', 
'OPdown', 
'TEup', 
'TEdown',
'DepthDown', 
'DepthUp', 
'ImpactDown', 
'ImpactUp', 
'DurUp', 
'DurDown',
'LogGUp', 
'LogGDown', 
'SradUp', 
'SteffUp', 
'SradDown', 
'SteffDown']


def cleaning(data):
    df = pd.DataFrame(data)
    df = cl.clean_array(df)
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

    df = df.applymap(lambda x: binary_replace.get(x, x) if isinstance(x, str) else x)
    return df

def setup(df,AIname):
    features = StandardizedColumnNames[1:]  # replace with your numeric columns that you want to keep
    label_col = StandardizedColumnNames[0]                  # replace with your target column that you want your model to predict

    scalencode = {f'{AIname}scaler':StandardScaler(),f'{AIname}le' : LabelEncoder()}

    X = scalencode[f'{AIname}scaler'].fit_transform(df[features].values.astype(np.float32)) #(value-moyenne)/ecart-type to center data and get rid of unit

    Y = scalencode[f'{AIname}le'].fit_transform(df[label_col].values)   #transforms labels to integers

    joblib.dump(scalencode[f'{AIname}scaler'], f"utils/Data/AI/{AIname}/scaler.pkl") # save the scaler for later use
    joblib.dump(scalencode[f'{AIname}le'], f"utils/Data/AI/{AIname}/label_encoder.pkl") # save the label encoder for later use

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

    loaders = {'train':(train_loader,X_train,Y_train),'validation':(val_loader,X_val,Y_val),'test':(test_loader,X_test,Y_test)}
    
    return loaders,scalencode,X,Y


def set_seed(seed=42): #setting seed for reproducibility on all libraries
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(67)

class SimpleMLP(nn.Module): #Multi Layer Perceptron subclass of nn.Module

    def __init__(self, hidden=[128,64], dropout=0.2):
        #amount of columns in input data, list of hidden layer sizes, number of output classes, dropout rate #Adapt these hyperparameters depending on your dataset

        super().__init__() #call parent constructor to get nn.Module functionalities
        
        layers = [] 
        in_dim = 35 #amount of features in input data = number of neurons in input layer (intially)

        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)] 
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))   #final layer without activation 
        self.net = nn.Sequential(*layers) #automatically inputs data through all layer functions in sequence

    def forward(self, x):
        return self.net(x)
def devicesel():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE

def definemodel(X,hiddenlayers=[128,64]):
    DEVICE = devicesel()
    model = SimpleMLP(
        hidden=hiddenlayers,  #size of hidden layers, can be changed
        ).to(DEVICE)
    return model

def weights(Y,model):
    cw = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(devicesel())

    #When computing the loss, misclassifying a minority class will incur a higher penalty than misclassifying a majority class

    criterion = nn.CrossEntropyLoss(weight=class_weights)   #multiplies the loss of each class by the weights defined previously
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  #Adam optimizer for weight adjustment, lr=learning rate
    return criterion,optimizer


def train_one_epoch(model, loader, optimizer, criterion):
    #model: the neural network,loader: DataLoader for training data (ex: train_loader), optimizer: optimization algorithm, criterion: loss function

    model.train() #set model to training mode (enables dropout, etc)

    running_loss = 0.0 #to accumulate loss over the epoch

    for Xb, yb in loader:
        Xb, yb = Xb.to(devicesel()), yb.to(devicesel())
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
            Xb, yb = Xb.to(devicesel()), yb.to(devicesel()) 
            logits = model(Xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * Xb.size(0) #accumulate loss, scaled by batch size
            pred = logits.argmax(dim=1).cpu().numpy() #get predicted class by taking argmax of logits (dim=1 means that we take the max across rows)
            #cpu because numpy can't handle cuda tensors
            preds.append(pred); trues.append(yb.cpu().numpy()) #store predictions and true labels
    preds = np.concatenate(preds); trues = np.concatenate(trues) #sqish lists into single arrays
    acc = accuracy_score(trues, preds) #compute accuracy
    return running_loss / len(loader.dataset), acc #average loss and accuracy over the evaluation

def training(epochs,model,train_loader,val_loader,test_loader,AIname,le,optimizer,criterion): 

    DEVICE = devicesel()
    
    best_val_loss = float("inf")  # start with "infinity" so any real loss will be smaller
    AIfilepath = f"Python/server/utils/Data/AI/{AIname}/{AIname}.pth"

    for epoch in range(epochs):   # 30 epochs example
        # --- Train on all batches in the training set ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion) 
        
        # --- Evaluate on validation set (no weight updates) ---
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # --- Check if this is the best model so far ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model's weights to disk
            torch.save(model.state_dict(), AIfilepath)   # checkpoint

    # This ensures we use the model that performed best on validation data
    model.load_state_dict(torch.load(AIfilepath, map_location=DEVICE))

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

    from sklearn.metrics import classification_report
    labels = np.unique(trues)
    target_names = [str(x) for x in le.inverse_transform(labels)]
    classif = classification_report(trues, preds, labels=labels, target_names=target_names)
    return classif

def AITRAIN(data,hiddenlayers=[128,64],epochs=100,AIname='NewAI'):
    df = cleaning(data)  
    loaders, scalencode, X, Y = setup(df,AIname)

    # Define the model
    model = definemodel(X, hiddenlayers=hiddenlayers)

    # Compute loss function and optimizer
    criterion, optimizer = weights(Y, model)

    # Train the model
    classif = training(
        epochs=epochs,
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['validation'],
        test_loader=loaders['test'],
        AIname=AIname,
        device=torch.device('cpu'),
        le=scalencode[f'{AIname}le'],
        optimizer=optimizer,
        criterion=criterion
    )

    model_path = f"utils/Data/AI/{AIname}/{AIname}.pth"
    scaler_path = f"utils/Data/AI/{AIname}/scaler.pkl"
    le_path = f"utils/Data/AI/{AIname}/label_encoder.pkl"

    return classif, model_path, scaler_path, le_path

def AIPredict(model_path, scaler_path, le_path, data, hiddenlayers=[128,64]):
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    model = SimpleMLP(hidden=hiddenlayers)
    model.load_state_dict(torch.load(model_path, map_location=devicesel()))
    model.eval()

    X_nu = np.array(data, dtype=np.float32)
    X_nu[np.isnan(X_nu)] = 0.0  # replace NaNs with 0
    X_scaled = scaler.transform(X_nu)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():

        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).numpy()
        labels = le.inverse_transform(preds)
        prob_scores = probs.max(dim=1).values.numpy()

    return labels, prob_scores

#Function order:
'''
df = cleaning(data) en liste
setup(df) en pandas

loads,scale,label,x,y = setup(df)

model = definemodel(x,hiddenlayers)

cri, opti = weights(y,model)

training(epochs,model,loads['train'],loads['validation'],loads['test'],AIname,device,label,opti,cri)

ce que l'utilisateur doit fournir: data, hiddenlayers,epochs,AIname
'''