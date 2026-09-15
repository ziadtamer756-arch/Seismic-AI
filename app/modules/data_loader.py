import sqlite3
import pandas as pd
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


DATABASE = os.path.join(
    BASE_DIR,
    "earthquakes.db"
)



def load_earthquakes():

    try:

        conn = sqlite3.connect(DATABASE)

        df = pd.read_sql_query(
            "SELECT * FROM earthquakes",
            conn
        )

        conn.close()

        return df


    except Exception as e:

        print("Database Error:", e)

        return pd.DataFrame()



def get_statistics(df):

    if df.empty:

        return {
            "total":0,
            "average_magnitude":0,
            "max_magnitude":0,
            "average_depth":0
        }


    return {

        "total": len(df),

        "average_magnitude":
            round(df["magnitude"].mean(),2),

        "max_magnitude":
            round(df["magnitude"].max(),2),

        "average_depth":
            round(df["depth"].mean(),2)

    }