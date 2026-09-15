import sqlite3
import pandas as pd
import json


def generate_ai_report():

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


    latest = df.iloc[0]


    with open(
        "model_metrics.json",
        "r"
    ) as f:

        metrics = json.load(f)



    report = f"""

SEISMIC AI REPORT
========================


Latest Earthquake

Location:
{latest['place']}


Magnitude:
{latest['magnitude']}


Depth:
{latest['depth']} km


Coordinates:

Latitude:
{latest['latitude']}

Longitude:
{latest['longitude']}



NEURAL NETWORK PERFORMANCE

Risk Model Accuracy:
{metrics['classification']['accuracy']*100:.2f}%


Precision:
{metrics['classification']['precision']*100:.2f}%


Recall:
{metrics['classification']['recall']*100:.2f}%


F1 Score:
{metrics['classification']['f1_score']*100:.2f}%



Magnitude Prediction

MAE:
{metrics['regression']['MAE']}


RMSE:
{metrics['regression']['RMSE']}



AI Interpretation:

The seismic AI system analyzed the latest
earthquake using machine learning models.

Continuous monitoring is recommended.

"""


    return report