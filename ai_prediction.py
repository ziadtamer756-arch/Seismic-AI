import sqlite3
import pandas as pd
import joblib
import yaml
import streamlit as st

from logger_config import logger
from explanation import explain_prediction



# ==========================
# LOAD CONFIG
# ==========================

with open(
    "config/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
MAG_MODEL_PATH = os.path.join(MODEL_DIR, "magnitude_model.pkl")
RISK_SCALER_PATH = os.path.join(MODEL_DIR, "risk_scaler.pkl")
MAG_SCALER_PATH = os.path.join(MODEL_DIR, "magnitude_scaler.pkl")
DATABASE_PATH = os.path.join(BASE_DIR, "earthquakes.db")




# ==========================
# LOAD MODELS
# ==========================

@st.cache_resource
def load_models():


    risk_model = joblib.load(
        RISK_MODEL_PATH
    )


    magnitude_model = joblib.load(
        MAG_MODEL_PATH
    )


    risk_scaler = joblib.load(
        RISK_SCALER_PATH
    )


    magnitude_scaler = joblib.load(
        MAG_SCALER_PATH
    )


    logger.info(
        "Models loaded successfully"
    )


    return (

        risk_model,

        magnitude_model,

        risk_scaler,

        magnitude_scaler

    )



risk_model, magnitude_model, risk_scaler, magnitude_scaler = load_models()




# ==========================
# DATABASE
# ==========================


def get_latest_earthquake():


    conn = sqlite3.connect(
        DATABASE_PATH
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
# FEATURE ENGINEERING
# ==========================


def prepare_features(earthquake):


    time = pd.to_datetime(

        earthquake["time"],

        unit="ms"

    )



    data = {}



    data["depth"] = earthquake["depth"]

    data["latitude"] = earthquake["latitude"]

    data["longitude"] = earthquake["longitude"]



    data["hour"] = time.hour

    data["month"] = time.month

    data["day"] = time.day



    data["day_of_year"] = (

        time.dayofyear

    )



    data["season"] = (

        time.month % 12 // 3

    )



    data["depth_category"] = (

        0

        if earthquake["depth"] < 70

        else 1

        if earthquake["depth"] < 300

        else 2

    )



    data["lat_lon_product"] = (

        earthquake["latitude"]

        *

        earthquake["longitude"]

    )



    data["events_previous"] = 0



    data["avg_previous_magnitude"] = (

        earthquake["magnitude"]

    )



    return pd.DataFrame(

        [data]

    )




# ==========================
# PREDICTION
# ==========================


def predict_latest_earthquake():


    logger.info(
        "Starting prediction"
    )



    earthquake = get_latest_earthquake()



    features = prepare_features(

        earthquake

    )



    # ======================
    # RISK
    # ======================


    risk_scaled = risk_scaler.transform(

        features

    )



    risk_prediction = risk_model.predict(

        risk_scaled

    )[0]



    probabilities = risk_model.predict_proba(

        risk_scaled

    )[0]



    confidence = max(

        probabilities

    ) * 100




    labels = {

        0:"Low",

        1:"Medium",

        2:"High"

    }



    risk = labels.get(

        risk_prediction,

        "Unknown"

    )




    # ======================
    # MAGNITUDE
    # ======================


    magnitude_features = features.drop(

        columns=[

            "avg_previous_magnitude"

        ]

    )



    mag_scaled = magnitude_scaler.transform(

        magnitude_features

    )



    magnitude = magnitude_model.predict(

        mag_scaled

    )[0]




    # ======================
    # SHAP EXPLANATION
    # ======================


    try:


        explanation_df = explain_prediction(

            features

        )


        explanation = (

            explanation_df["feature"]

            .tolist()

        )


    except Exception as e:


        logger.error(

            f"SHAP explanation error: {e}"

        )


        explanation = []





    logger.info(

        f"Prediction: {risk}, confidence={confidence}"

    )




    return {


        "risk": risk,


        "confidence": round(

            float(confidence),

            2

        ),


        "magnitude": round(

            float(magnitude),

            2

        ),


        "location": earthquake["place"],


        "depth": float(

            earthquake["depth"]

        ),


        "explanation": explanation


    }





# ==========================
# TEST
# ==========================


if __name__ == "__main__":


    result = predict_latest_earthquake()


    print(result)