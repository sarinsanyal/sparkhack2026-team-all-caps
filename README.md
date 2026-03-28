# 🚀 Federated Learning for Privacy-Preserving Healthcare

## 📌 Overview

This project implements a **privacy-preserving federated learning system** for predicting heart disease across multiple hospitals.

Instead of sharing sensitive patient data, each hospital trains a local model and only shares **encrypted and privacy-protected model updates** with a central server.

---

## 🎯 Key Features

* 🧠 **Federated Learning (Flower Framework)**
* 🔐 **Differential Privacy (Gaussian Noise)**
* 🔒 **Encrypted Model Communication (AES-based)**
* 🏥 **Multi-client (Hospital) Simulation**
* 📊 **Live Training Metrics Dashboard**
* 🔍 **Real-time Patient Prediction**

---

## 🏗️ Project Structure

```
project-root/
├── clients/            # Hospital client code
├── server/             # Federated server
├── model/              # Neural network architecture
├── privacy/            # DP + encryption modules
├── data/               # Dataset
├── logs/               # Training logs
├── dashboard/          # UI 
├── predict_logic.py          # CLI prediction
├── run_simulation.*    # Automation scripts
```

---

## ⚙️ How It Works

1. Each hospital trains the model locally on its private data
2. Model weights are:

   * 🟢 Noised using Differential Privacy
   * 🔒 Encrypted before transmission
3. Server aggregates updates using **Federated Averaging (FedAvg)**
4. Global model is updated and redistributed
5. Process repeats for multiple rounds

---

## 🔐 Privacy & Security

* **No raw data leaves hospitals**
* **Differential Privacy** prevents data leakage
* **Encryption** secures communication channels

> 🔑 *“We move the model to the data, not the data to the model.”*

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

### 🔹 Option 1: Manual Run

#### Start Server

```bash
python -m server.server
```

#### Start Clients (in separate terminals)

```bash
python -m clients.client --hospital-id 1
python -m clients.client --hospital-id 2
python -m clients.client --hospital-id 3
```

---

### 🔹 Option 2: One-Command Run

#### Windows:

```bash
run_simulation.bat
```

#### Linux:

```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

---

## 📊 Dashboard

Features:

* 📈 Global accuracy over rounds
* 🏥 Per-hospital performance
* 🔐 Privacy indicators
* 🔍 Live patient prediction

---

## 🔍 Prediction (CLI)

```bash
python predict.py
```

* Guided input (no raw comma input required)
* Outputs probability + classification

---

## 🧠 Model Details

* Input: 13 clinical features
* Output: Binary classification (Heart Disease: Yes/No)
* Activation: Sigmoid
* Loss: Binary Cross-Entropy

---

## 📈 Sample Results

* Global accuracy improves over rounds
* Final accuracy ~ *70-75%**
* Outperforms individual hospital models

---

## ⚖️ Trade-offs

| Factor                   | Impact                  |
| ------------------------ | ----------------------- |
| Higher privacy (noise ↑) | Accuracy ↓              |
| More clients             | Better generalization   |
| Poor data quality        | Can affect global model |

---

## 🌍 Real-World Applications

* Multi-hospital AI collaboration
* Privacy-sensitive domains (healthcare, finance)
* Cross-institution model training

---

## 🏆 Why Federated Learning?

* ✔ No data sharing
* ✔ Regulatory compliance (HIPAA/GDPR)
* ✔ Better generalization
* ✔ Scalable across institutions

---

## 🚧 Future Improvements

* 🔄 Secure aggregation protocols
* 📊 Advanced client weighting
* ☁️ Cloud deployment
* 🌐 API-based inference

---

## 📜 License

This project is for educational and hackathon purposes.

---

## ⭐ Final Note

> **“Privacy is not a feature — it’s a requirement.”**

---