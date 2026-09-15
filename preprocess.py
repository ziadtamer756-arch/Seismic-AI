import sqlite3
import pandas as pd


DATABASE = r"F:\Seismic-Agent\earthquakes.db"


print("Loading earthquake data...")


# قراءة البيانات من قاعدة البيانات
conn = sqlite3.connect(DATABASE)

df = pd.read_sql(
    "SELECT * FROM earthquakes",
    conn
)

conn.close()


print("Original data:")
print(df.shape)



# =========================
# تنظيف البيانات
# =========================

df = df.dropna()


df["magnitude"] = pd.to_numeric(
    df["magnitude"]
)


df["depth"] = pd.to_numeric(
    df["depth"]
)



# =========================
# تحويل الوقت
# =========================

df["time"] = pd.to_datetime(
    df["time"],
    unit="ms"
)


df["hour"] = df["time"].dt.hour

df["month"] = df["time"].dt.month

df["day"] = df["time"].dt.day



# =========================
# إنشاء مستوى الخطورة
# =========================

# 0 = Low
# 1 = Medium
# 2 = High

def risk_level(mag):

    if mag < 4:
        return 0

    elif mag < 6:
        return 1

    else:
        return 2



df["risk_level"] = df["magnitude"].apply(
    risk_level
)



# =========================
# اختيار بيانات الذكاء الاصطناعي
# =========================

features = [

    "depth",
    "latitude",
    "longitude",
    "hour",
    "month",
    "day"

]


prepared = df[features].copy()


prepared["risk_level"] = df["risk_level"]

prepared["magnitude"] = df["magnitude"]



# =========================
# حفظ ملف التدريب
# =========================

prepared.to_csv(
    "prepared_earthquake_data.csv",
    index=False
)



print("✅ Preprocessing finished")

print(
    "Prepared data:"
)

print(
    prepared.shape
)


print(
    prepared.head()
)