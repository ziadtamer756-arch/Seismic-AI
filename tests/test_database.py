import sqlite3
import os


DATABASE = "earthquakes.db"



def test_database_exists():

    assert os.path.exists(DATABASE)



def test_database_has_data():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM earthquakes"
    )

    count = cursor.fetchone()[0]

    conn.close()


    assert count > 0