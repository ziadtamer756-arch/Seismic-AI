import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import load_model

from pytorch_tabnet.tab_model import TabNetClassifier


# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "risk_training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EVAL_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)


# ==========================
# Load Data
# ==========================

print("Loading data...")

df = pd.read_csv(DATA_PATH)


X = df.drop(
    "risk_level",
    axis=1
)

y = df["risk_level"]


X = pd.get_dummies(
    X,
    columns=[
        "season",
        "depth_category"
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# Scaling
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================
# Random Forest
# ==========================

print("Training RF...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)


rf_pred = rf.predict_proba(
    X_test
)


# ==========================
# Load Deep Models
# ==========================

print("Loading Deep Models...")


dnn = load_model(
    os.path.join(
        MODEL_DIR,
        "deep_seismic_model.keras"
    )
)


dnn_prob = dnn.predict(
    X_test_scaled
)


tabnet = TabNetClassifier()

tabnet.load_model(
    os.path.join(
        MODEL_DIR,
        "tabnet_seismic_model.zip"
    )
)


tabnet_prob = tabnet.predict_proba(
    X_test_scaled
)


wide = load_model(
    os.path.join(
        MODEL_DIR,
        "wide_deep_seismic_model.keras"
    )
)


wide_prob = wide.predict(
    X_test_scaled
)


# ==========================
# Ensemble Voting
# ==========================

print("Creating Ensemble...")


ensemble_prob = (

    rf_pred * 0.25 +

    dnn_prob * 0.25 +

    tabnet_prob * 0.25 +

    wide_prob * 0.25

)


ensemble_pred = np.argmax(
    ensemble_prob,
    axis=1
)


# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(
    y_test,
    ensemble_pred
)


f1 = f1_score(
    y_test,
    ensemble_pred,
    average="weighted"
)


report = classification_report(
    y_test,
    ensemble_pred
)


print(report)


results = {

    "model": "Seismic Ensemble AI",

    "components": [

        "Random Forest",

        "DNN",

        "TabNet",

        "Wide Deep"

    ],

    "accuracy": float(
        accuracy
    ),

    "f1_score_weighted": float(
        f1
    ),

    "strategy": "Soft Voting"

}


with open(
    os.path.join(
        EVAL_DIR,
        "ensemble_results.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(results)

print(
    "✅ Ensemble Training Completed"
)