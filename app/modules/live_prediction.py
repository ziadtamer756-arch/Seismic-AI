import os
import sys
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# ==============================
# Paths
# ==============================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        CURRENT_DIR
    )
)

sys.path.append(PROJECT_DIR)


from live_earthquakes import fetch_latest_earthquakes


MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)


# ==============================
# Feature Engineering
# ==============================

def prepare_features(row):

    timestamp = pd.to_datetime(
        row["time"]
    )

    depth = float(
        row["depth"]
    )

    month = timestamp.month


    # depth category
    if depth < 70:
        depth_category = 0

    elif depth < 300:
        depth_category = 1

    else:
        depth_category = 2



    # season encoding

    if month in [12, 1, 2]:
        season = 1

    elif month in [3, 4, 5]:
        season = 2

    elif month in [6, 7, 8]:
        season = 3

    else:
        season = 4



    data = pd.DataFrame([{

        "depth": depth,

        "latitude":
            row["latitude"],

        "longitude":
            row["longitude"],

        "hour":
            timestamp.hour,

        "month":
            month,

        "day":
            timestamp.day,

        "day_of_year":
            timestamp.dayofyear,

        "season":
            season,

        "depth_category":
            depth_category,

        "lat_lon_product":
            row["latitude"] *
            row["longitude"],

        "events_previous":
            0,

        "avg_previous_magnitude":
            row["magnitude"]

    }])



    # one hot encoding

    data = pd.get_dummies(
        data,
        columns=[
            "season",
            "depth_category"
        ]
    )


    return data



# ==============================
# Live Prediction
# ==============================

def predict_live_risk():

    print(
        "Fetching latest earthquake..."
    )


    live = fetch_latest_earthquakes(
        limit=1
    )


    earthquake = live.iloc[0]


    print(
        "Earthquake:",
        earthquake["place"]
    )



    X = prepare_features(
        earthquake
    )


    # Load scaler

    scaler = joblib.load(

        os.path.join(
            MODEL_DIR,
            "deep_scaler.pkl"
        )

    )


    # Match training features

    expected_features = (
        scaler.feature_names_in_
    )


    for col in expected_features:

        if col not in X.columns:

            X[col] = 0



    X = X[
        expected_features
    ]



    X_scaled = scaler.transform(
        X
    )



    # Load DNN

    model = load_model(

        os.path.join(
            MODEL_DIR,
            "deep_seismic_model.keras"
        )

    )


    probability = model.predict(
        X_scaled
    )[0]



    prediction = np.argmax(
        probability
    )


    confidence = float(
        max(probability)
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


        "risk":
            risk_levels[prediction],


        "confidence":
            round(
                confidence * 100,
                2
            )

    }


    return result



# ==============================
# Test
# ==============================

if __name__ == "__main__":


    result = predict_live_risk()


    print(
        "\n========== Live Prediction =========="
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )