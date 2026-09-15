import requests
from datetime import datetime


def get_recent_earthquakes():


    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"


    params = {

        "format": "geojson",

        # 10 years historical data
        "starttime": "2016-01-01",

        "endtime": datetime.utcnow().isoformat(),

        # minimum magnitude
        "minmagnitude": 2.5,

        # maximum records
        "limit": 10000,

        "orderby": "time"

    }



    print("Downloading historical earthquake data...")


    response = requests.get(

        url,

        params=params

    )


    data = response.json()



    earthquakes = []



    for event in data["features"]:


        prop = event["properties"]

        geo = event["geometry"]["coordinates"]



        earthquakes.append({

            "place": prop["place"],

            "magnitude": prop["mag"],

            "time": prop["time"],

            "longitude": geo[0],

            "latitude": geo[1],

            "depth": geo[2]

        })



    return earthquakes



if __name__ == "__main__":


    events = get_recent_earthquakes()


    print(
        f"Found {len(events)} earthquakes"
    )