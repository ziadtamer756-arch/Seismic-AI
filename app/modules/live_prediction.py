import os
import sys
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier


# ======================================
# Project Paths
# ======================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        CURRENT_DIR
    )
)

sys.path.append(PROJECT_DIR)


# Import
try:
    from app.modules.live_earthquakes import fetch_latest_earthquakes
except ModuleNotFoundError:
    from live_earthquakes import fetch_latest_earthquakes



MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)



# ======================================
# Feature Engineering
# ======================================

def prepare_features(row):

    timestamp = pd.to_datetime(
        row["time"]
    )

    depth = float(
        row["depth"]
    )

    month = timestamp.month


    if depth < 70:
        depth_category = 0

    elif depth < 300:
        depth_category = 1

    else:
        depth_category = 2



    if month in [12, 1, 2]:
        season = 1

    elif month in [3, 4, 5]:
        season = 2

    elif month in [6, 7, 8]:
        season = 3

    else:
        season = 4



    features = pd.DataFrame([{

        "depth": depth,

        "latitude": row["latitude"],

        "longitude": row["longitude"],

        "hour": timestamp.hour,

        "month": month,

        "day": timestamp.day,

        "day_of_year": timestamp.dayofyear,

        "season": season,

        "depth_category": depth_category,

        "lat_lon_product":
            row["latitude"] *
            row["longitude"],

        "events_previous": 0,

        "avg_previous_magnitude":
            row["magnitude"]

    }])


    features = pd.get_dummies(
        features,
        columns=[
            "season",
            "depth_category"
        ]
    )


    return features



# ======================================
# Align Features
# ======================================

def align_features(X, scaler):

    expected = scaler.feature_names_in_


    for col in expected:

        if col not in X.columns:

            X[col] = 0


    return X[expected]



# ======================================
# Ensemble Prediction
# ======================================

def predict_live_ensemble():


    print(
        "Fetching live earthquake..."
    )


    live = fetch_latest_earthquakes(
        limit=1
    )


    earthquake = live.iloc[0]



    X = prepare_features(
        earthquake
    )



    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "deep_scaler.pkl"
        )
    )



    X = align_features(
        X,
        scaler
    )



    X_scaled = scaler.transform(
        X
    )



    predictions = []



    # ==========================
    # DNN
    # ==========================

    dnn = load_model(
        os.path.join(
            MODEL_DIR,
            "deep_seismic_model.keras"
        )
    )


    dnn_prob = dnn.predict(
        X_scaled,
        verbose=0
    )[0]


    predictions.append(
        dnn_prob
    )



    # ==========================
    # Wide Deep
    # ==========================

    wide = load_model(
        os.path.join(
            MODEL_DIR,
            "wide_deep_seismic_model.keras"
        )
    )


    wide_prob = wide.predict(
        X_scaled,
        verbose=0
    )[0]


    predictions.append(
        wide_prob
    )



    # ==========================
    # TabNet
    # ==========================

    tabnet = TabNetClassifier()


    tabnet.load_model(
        os.path.join(
            MODEL_DIR,
            "tabnet_seismic_model.zip"
        )
    )


    tabnet_prob = tabnet.predict_proba(
        X_scaled
    )[0]


    predictions.append(
        tabnet_prob
    )



    # ==========================
    # Soft Voting
    # ==========================

    final_probability = np.mean(
        predictions,
        axis=0
    )


    final_prediction = np.argmax(
        final_probability
    )


    confidence = float(
        max(final_probability)
    )



    risk_levels = {

        0: "Low",

        1: "Medium",

        2: "High"

    }



    result = {

        "location":
            earthquake["place"],

        "magnitude":
            float(
                earthquake["magnitude"]
            ),

        "depth":
            float(
                earthquake["depth"]
            ),

        "model":
            "Deep Learning Ensemble",

        "risk":
            risk_levels[
                final_prediction
            ],

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "models_used":
            [
                "DNN",
                "Wide Deep",
                "TabNet"
            ]

    }


    return result



# ======================================
# Test
# ======================================

if __name__ == "__main__":


    result = predict_live_ensemble()


    print(
        "\n========== Ensemble Live Prediction =========="
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )