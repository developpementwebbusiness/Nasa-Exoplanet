```
# STAR AI – Kepler Exoplanet Classification

**STAR AI** is a Python-based machine learning project for classifying Kepler exoplanet data.  
The system leverages a **Multi-Layer Perceptron (MLP)** neural network built with **PyTorch** and includes tools for data preprocessing, model training, and prediction.

---

## Project Structure

```
Python/
 └─ server/
     └─ utils/
         ├─ Data/
         │   └─ AI/
         │       └─ STAR_AI_v2/
         │           ├─ scaler.pkl
         │           ├─ label_encoder.pkl
         │           └─ STAR_AI_v2.pth
         ├─ STARAI.py          # Script for training and evaluating the model
         ├─ STARPredict.py     # Script for running predictions on new data
         └─ AITrainer.py       # Script to provide functions for user-side AI training
```

---

## Key Scripts

- **STARAI.py**  
  Full training pipeline including preprocessing, model definition, training loop, and evaluation.  
  Can be launched automatically to train the AI and save the scaler and label encoder.

- **STARPredict.py**  
  Uses STARAI to make predictions on new data.  
  Requires a matrix of rows corresponding to the standardized columns.

- **AITrainer.py**  
  Modular functions for cleaning, preprocessing, defining, and training an MLP.  
  Requires the user's own database with a disposition column, hidden layers, and epochs.

---

## Dataset

- **Kepler Object of Interest dataset (`kepler.csv`)**  
- Columns include: orbital period, transit depth, planet radius, equilibrium temperature, insolation flux, stellar properties, etc.  
- **Target column:** `Confirmation`  
  - Class labels are converted to **binary (True/False)**, even if the original data contains more disposition types.

---

## Requirements

- **Python 3.10+**
- Install required libraries:

```bash
pip install numpy pandas scikit-learn torch joblib rich
```

- Optional GPU support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Setup

1. Clone the repository.  
2. Place `kepler.csv` under `Python/server/utils/Data/`.  
3. Install dependencies (see above).  
4. Ensure the folder `Python/server/utils/Data/AI/STAR_AI_v2/` exists for saving scalers, encoders, and model weights.

---

## Custom AI training

Within the AITrainer file you'll find a multitude of functions, with the last two ones being the most important (AITRAIN,AIPredict).
These functions are responsible for handling the user's data and desired training parameters as well has their scaler, encoder and AI file.

---

## Notes

- All preprocessing (scaling, label encoding) is handled automatically.  
- The model is **binary**: only True/False outputs, even if original data has multiple disposition types.  
- Dropout and class weights are applied to handle overfitting and imbalanced classes.
```