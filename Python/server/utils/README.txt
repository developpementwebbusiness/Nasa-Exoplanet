STAR AI is a Python-based machine learning project for classifying Kepler exoplanet data. 
The system leverages a Multi-Layer Perceptron (MLP) neural network built with PyTorch and includes tools for data preprocessing, model training, and prediction.

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