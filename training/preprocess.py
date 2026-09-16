import sqlite3
import pandas as pd
import numpy as np


DATABASE = r"F:\Seismic-Agent\earthquakes.db"


print("Loading earthquake data...")


conn = sqlite3.connect(DATABASE)


df = pd.read_sql(
    "SELECT * FROM earthquakes",
    conn
)


conn.close()


print("Original data:")
print(df.shape)



# =========================
# CLEAN DATA
# =========================


df = df.dropna()


df["magnitude"] = pd.to_numeric(
    df["magnitude"]
)


df["depth"] = pd.to_numeric(
    df["depth"]
)



df["latitude"] = pd.to_numeric(
    df["latitude"]
)


df["longitude"] = pd.to_numeric(
    df["longitude"]
)



# =========================
# TIME FEATURES
# =========================


df["time"] = pd.to_datetime(
    df["time"],
    unit="ms"
)


df["hour"] = df["time"].dt.hour

df["month"] = df["time"].dt.month

df["day"] = df["time"].dt.day

df["day_of_year"] = (
    df["time"].dt.dayofyear
)


df["season"] = (
    df["month"] % 12 // 3
)



# =========================
# SEISMIC FEATURES
# =========================


# Energy released by earthquake

df["seismic_energy"] = (
    10 ** (1.5 * df["magnitude"])
)



# Depth classification

def depth_category(depth):

    if depth < 70:
        return 0

    elif depth < 300:
        return 1

    else:
        return 2



df["depth_category"] = (
    df["depth"]
    .apply(depth_category)
)



# =========================
# LOCATION FEATURES
# =========================


df["lat_lon_product"] = (

    df["latitude"] *
    df["longitude"]

)

# =========================
# HISTORICAL ACTIVITY
# =========================


df = df.sort_values(
    "time"
)


# تحويل الوقت إلى index مؤقت

temp = df.set_index(
    "time"
)


# عدد الزلازل خلال آخر 24 ساعة

temp["events_previous"] = (
    temp["magnitude"]
    .rolling("24h")
    .count()
)


# متوسط آخر 10 زلازل

temp["avg_previous_magnitude"] = (
    temp["magnitude"]
    .rolling(
        10,
        min_periods=1
    )
    .mean()
)


# رجوع الوقت كعمود

df = temp.reset_index()


# إزالة القيم الفارغة

df["events_previous"] = (
    df["events_previous"]
    .fillna(0)
)


df["avg_previous_magnitude"] = (
    df["avg_previous_magnitude"]
    .fillna(
        df["magnitude"].mean()
    )
)
# =========================
# MULTI FACTOR RISK SCORE
# =========================


# Normalize magnitude

mag_score = (
    (df["magnitude"] - df["magnitude"].min())
    /
    (df["magnitude"].max() - df["magnitude"].min())
)



# Shallow earthquakes are more critical

depth_score = np.where(

    df["depth"] < 70,

    1,

    np.where(

        df["depth"] < 300,

        0.5,

        0.2

    )

)



# Historical activity normalization

history_score = (

    df["events_previous"]

    /

    (df["events_previous"].max()+1)

)



# Total seismic risk score


df["risk_score"] = (

    0.5 * mag_score +

    0.25 * depth_score +

    0.25 * history_score

)



# Convert score to classes


def create_risk(score):

    if score < 0.35:

        return 0

    elif score < 0.65:

        return 1

    else:

        return 2



df["risk_level"] = (

    df["risk_score"]

    .apply(create_risk)

)

# =========================
# FEATURES FOR RISK MODEL
# بدون magnitude
# =========================


risk_features = [

"depth",

"latitude",

"longitude",

"hour",

"month",

"day",

"day_of_year",

"season",

"depth_category",

"lat_lon_product",

"events_previous",

"avg_previous_magnitude"

]



risk_data = df[
    risk_features
].copy()


risk_data["risk_level"] = (
    df["risk_level"]
)



# =========================
# FEATURES FOR MAGNITUDE MODEL
# =========================


magnitude_features = [

"depth",

"latitude",

"longitude",

"hour",

"month",

"day",

"day_of_year",

"season",

"depth_category",

"lat_lon_product",

"events_previous"

]



magnitude_data = df[
    magnitude_features
].copy()


magnitude_data["magnitude"] = (
    df["magnitude"]
)



# =========================
# SAVE FILES
# =========================


risk_data.to_csv(
    "risk_training_data.csv",
    index=False
)


magnitude_data.to_csv(
    "magnitude_training_data.csv",
    index=False
)



print("✅ Preprocessing completed")


print(
    "Risk data:",
    risk_data.shape
)


print(
    "Magnitude data:",
    magnitude_data.shape
)