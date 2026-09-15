import sqlite3
import pandas as pd
import joblib
import os


# ==========================
# LOAD MODELS
# ==========================

MODEL_PATH = "risk_model.pkl"
MAG_MODEL_PATH = "magnitude_model.pkl"
SCALER_PATH = "scaler.pkl"


risk_model = joblib.load(
    MODEL_PATH
)

magnitude_model = joblib.load(
    MAG_MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)



# ==========================
# DATABASE
# ==========================

def get_latest_earthquake():

    conn = sqlite3.connect(
        "earthquakes.db"
    )

    df = pd.read_sql(
        """
        SELECT *
        FROM earthquakes
        ORDER BY id DESC
        LIMIT 1
        """,
        conn
    )

    conn.close()

    return df.iloc[0]



# ==========================
# AI PREDICTION
# ==========================

def predict_latest_earthquake():


    earthquake = get_latest_earthquake()


    features = pd.DataFrame(
        [[
            earthquake["depth"],
            earthquake["latitude"],
            earthquake["longitude"],
            pd.to_datetime(
                earthquake["added_date"]
            ).hour,
            pd.to_datetime(
                earthquake["added_date"]
            ).month,
            pd.to_datetime(
                earthquake["added_date"]
            ).day
        ]],
        columns=[
            "depth",
            "latitude",
            "longitude",
            "hour",
            "month",
            "day"
        ]
    )



    scaled_features = scaler.transform(
        features
    )


    # Risk prediction

    risk_prediction = risk_model.predict(
        scaled_features
    )[0]



    risk_labels = {

        0:"Low",

        1:"Medium",

        2:"High"

    }


    risk = risk_labels.get(
        risk_prediction,
        "Unknown"
    )



    # Magnitude prediction

    magnitude = magnitude_model.predict(
        scaled_features
    )[0]



    return {

        "risk": risk,

        "magnitude": round(
            float(magnitude),
            2
        ),

        "location": earthquake["place"],

        "depth": earthquake["depth"]

    }




# TEST

if __name__ == "__main__":


    result = predict_latest_earthquake()


    print(result)