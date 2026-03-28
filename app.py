import os
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the functions from your prediction script
# Make sure your prediction script is named predict_logic.py
try:
    from predict_logic import load_model, preprocess, predict
    MODEL_AVAILABLE = True
    print("✅ Model logic imported successfully.")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"❌ ERROR LOADING MODEL: {e}") # This will tell you the EXACT problem

app = Flask(__name__)
CORS(app)  # This allows your GitHub Pages/HTML file to talk to this server

# Load the model once when the server starts
model = None
if MODEL_AVAILABLE:
    model = load_model()

# --- ROUTE 1: START THE FEDERATED LEARNING SERVERS ---
@app.route('/run-predict', methods=['POST'])
def run_servers():
    """
    Triggers the .bat file to start the FL Server and Clients.
    """
    try:
        # shell=True is required to run .bat files on Windows
        # We use Popen so it runs in the background and doesn't block Flask
        subprocess.Popen(["run_servers.bat"], shell=True)
        return jsonify({
            "status": "success", 
            "message": "Federated Learning System Initialized. Check your taskbar for new windows."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# --- ROUTE 2: HANDLE LIVE PREDICTIONS FROM DASHBOARD ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Receives 13 features from the dashboard and returns a heart disease risk result.
    """
    if not MODEL_AVAILABLE or model is None:
        return jsonify({"error": "Prediction model not loaded on server."}), 500

    try:
        data = request.json.get('features')
        
        if not data or len(data) != 13:
            return jsonify({"error": "Invalid input: Exactly 13 features required."}), 400

        # Convert list to numpy array
        sample = np.array(data)

        # 1. Preprocess (Scaling)
        processed_sample = preprocess(sample)

        # 2. Run Prediction
        prob, pred = predict(model, processed_sample)

        return jsonify({
            "probability": float(prob),
            "prediction": "HIGH RISK" if pred == 1 else "LOW RISK",
            "is_high_risk": bool(pred)
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


# --- ROUTE 3: HEALTH CHECK ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model_loaded": MODEL_AVAILABLE})


if __name__ == '__main__':
    print("""
    🚀 DASHBOARD CONTROLLER ONLINE
    --------------------------------------
    API URL: http://127.0.0.1:5000
    Endpoints:
    - POST /run-predict (Starts FL training)
    - POST /predict     (Calculates risk)
    --------------------------------------
    """)
    app.run(host='0.0.0.0', port=5000, debug=False)