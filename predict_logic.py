import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from model.net import Net

# This ensures the script finds files even when called from Flask
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model = Net()
    model_path = os.path.join(BASE_DIR, "saved_model", "global_model.pth")
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def preprocess(sample):
    # Load CSV to fit the scaler exactly as it was during training
    csv_path = os.path.join(BASE_DIR, "data", "heart_disease.csv")
    df = pd.read_csv(csv_path)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    
    X = df.iloc[:, 0:13].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(sample.reshape(1, -1))

def predict(model, sample):
    x = torch.tensor(sample, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0
    return prob, pred