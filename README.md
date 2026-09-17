# 🌍 Seismic-AI
## Real-Time Earthquake Risk Prediction System Using Machine Learning & Deep Learning

---

## 📌 Project Overview

Seismic-AI is an intelligent earthquake risk prediction platform that combines traditional Machine Learning, Deep Learning, Ensemble Learning, Real-Time Earthquake Data, and Explainable AI.

The system predicts earthquake risk levels using historical seismic data and provides real-time predictions through a Streamlit dashboard.

---

# 🚀 Main Features

## 🌐 Real-Time Earthquake Monitoring

- Integration with USGS Earthquake API
- Fetch latest earthquake events
- Display:
  - Location
  - Magnitude
  - Depth
  - Coordinates


---

# 🤖 Machine Learning Models

Implemented models:

- Random Forest
- Tuned ML Models


---

# 🧠 Deep Learning Architectures

The project includes:

## 1. Deep Neural Network (DNN)

Framework:
TensorFlow / Keras

Architecture:

- Dense Layers
- Batch Normalization
- Dropout
- Softmax Output


## 2. Wide & Deep Network

Framework:
TensorFlow / Keras

Architecture:

- Wide Branch
- Deep Dense Layers
- BatchNormalization
- Dropout


## 3. TabNet

Framework:
PyTorch TabNet

Architecture:

- Attention-based Feature Selection
- Sparse Feature Learning


---

# 🔥 Ensemble AI

The final prediction system combines:


Random Forest
+
DNN
+
Wide & Deep
+
TabNet


Using:


Soft Voting Ensemble


Output:

- Risk Level
- Confidence Score
- Model Agreement


---

# 📊 Model Performance

| Model | Accuracy |
|------|----------|
| DNN | ~90% |
| Wide & Deep | 90.41% |
| TabNet | 90.44% |
| Ensemble AI | 92.33% |


---

# 🔍 Explainable AI

Implemented using:

## SHAP

Provides:

- Feature Importance
- Model Explanation
- Understanding of prediction factors


---

# 🖥 Dashboard

Built using:

## Streamlit

Features:

✅ Interactive Maps  
✅ Earthquake Analytics  
✅ Model Comparison  
✅ SHAP Visualization  
✅ Live Earthquake Feed  
✅ Real-Time Ensemble Prediction  


---

# 📂 Project Structure


Seismic-AI/

├── app/
│ └── modules/
│ ├── live_earthquakes.py
│ └── live_prediction.py

├── models/
│ ├── deep_seismic_model.keras
│ ├── wide_deep_seismic_model.keras
│ └── tabnet_seismic_model.zip

├── training/
│ ├── train_deep_learning.py
│ ├── train_tabnet.py
│ ├── train_wide_deep.py
│ └── train_ensemble.py

├── evaluation/

├── reports/

├── dashboard.py

└── requirements.txt



---

# ⚙️ Installation

Create environment:

```bash
python -m venv dl_env

Activate:

Windows:

dl_env\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Run Dashboard
python -m streamlit run dashboard.py
🧪 Live Prediction Example

Example output:

========== Ensemble Live Prediction ==========

Location:
California

Model:
Deep Learning Ensemble

Risk:
Low

Confidence:
97%

Models Used:
DNN
Wide Deep
TabNet
👨‍💻 Technologies
Python
TensorFlow
Keras
PyTorch
PyTorch TabNet
Scikit-Learn
Streamlit
SHAP
Pandas
NumPy
📌 Future Improvements
GPU acceleration
Larger seismic datasets
Time-series earthquake forecasting
Mobile application
Cloud deployment
Project Status

✅ Machine Learning Completed
✅ Deep Learning Completed
✅ Ensemble AI Completed
✅ Real-Time Prediction Completed
✅ Dashboard Completed