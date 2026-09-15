import sqlite3
from datetime import datetime


DATABASE = "earthquakes.db"


def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS earthquakes (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        place TEXT,

        magnitude REAL,

        time INTEGER,

        latitude REAL,

        longitude REAL,

        depth REAL,

        added_date TEXT

    )
    """)


    conn.commit()
    conn.close()



def save_earthquakes(events):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    for e in events:

        cursor.execute("""

        INSERT INTO earthquakes
        (
        place,
        magnitude,
        time,
        latitude,
        longitude,
        depth,
        added_date
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)

        """,

        (

        e["place"],
        e["magnitude"],
        e["time"],
        e["latitude"],
        e["longitude"],
        e["depth"],
        datetime.now().isoformat()

        ))


    conn.commit()
    conn.close()



def get_statistics():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    cursor.execute(
        """
        SELECT
        COUNT(*),
        AVG(magnitude),
        MAX(magnitude),
        AVG(depth)

        FROM earthquakes
        """
    )


    result = cursor.fetchone()


    conn.close()


    return {

        "total_events": result[0],
        "average_magnitude": result[1],
        "maximum_magnitude": result[2],
        "average_depth": result[3]

    }



if __name__ == "__main__":

    create_database()

    print("✅ Database ready")
