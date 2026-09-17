import requests
import pandas as pd
from datetime import datetime


USGS_URL = (
    "https://earthquake.usgs.gov/"
    "earthquakes/feed/v1.0/"
    "summary/all_day.geojson"
)


def fetch_latest_earthquakes(limit=20):
    """
    Fetch latest earthquakes from USGS API
    """

    response = requests.get(
        USGS_URL,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    earthquakes = []

    for event in data["features"][:limit]:

        properties = event["properties"]
        geometry = event["geometry"]

        coords = geometry["coordinates"]

        earthquakes.append({

            "place": properties.get(
                "place",
                "Unknown"
            ),

            "magnitude": properties.get(
                "mag",
                0
            ),

            "longitude": coords[0],

            "latitude": coords[1],

            "depth": coords[2],

            "time": datetime.fromtimestamp(
                properties["time"] / 1000
            )

        })


    return pd.DataFrame(
        earthquakes
    )


def get_live_summary():

    df = fetch_latest_earthquakes()

    summary = {

        "total_events": len(df),

        "max_magnitude": float(
            df["magnitude"].max()
        ),

        "average_depth": float(
            df["depth"].mean()
        )

    }

    return summary


if __name__ == "__main__":

    df = fetch_latest_earthquakes()

    print(df.head())

    print(
        get_live_summary()
    )