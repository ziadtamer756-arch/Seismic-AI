import streamlit as st
import pandas as pd
import sqlite3
import json
import os


from ai_prediction import predict_latest_earthquake
from ai_report import generate_ai_report



# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Seismic AI Platform",
    page_icon="🌎",
    layout="wide"
)



# ==========================
# LANGUAGE BUTTON
# ==========================

col1, col2 = st.columns([8,1])

with col2:

    language = st.selectbox(
        "🌐",
        [
            "English",
            "العربية"
        ]
    )


# ==========================
# LIVE DATA UPDATE
# ==========================

from earthquake_data import get_recent_earthquakes
from database import save_earthquakes


if st.button("🔄 Update Latest Earthquakes"):

    with st.spinner("Downloading latest seismic events..."):

        try:

            events = get_recent_earthquakes()

            save_earthquakes(events)

            st.success(
                f"✅ Updated successfully: {len(events)} new earthquakes"
            )

            st.rerun()


        except Exception as e:

            st.error(
                f"Update failed: {e}"
            )
# ==========================
# TITLE
# ==========================


if language == "العربية":

    title = "🌎 منصة الذكاء الاصطناعي لمراقبة الزلازل"

else:

    title = "🌎 Seismic AI Research Platform"



st.title(title)



st.caption(
    "Advanced earthquake monitoring using Neural Networks and Artificial Intelligence"
)



# ==========================
# TRAINING INFO
# ==========================


st.info(
"""
🧠 Neural Network Training Information

The seismic AI models were trained using real earthquake datasets.

Training Data:
• Real seismic events database
• Dataset size: 80 GB
• Geological parameters
• Magnitude
• Depth
• Location
• Time-based features


AI Models:

• Neural Network Risk Classifier
• Neural Network Magnitude Predictor

The system uses machine learning algorithms
for seismic analysis and risk assessment.
"""
)



# ==========================
# DATABASE
# ==========================


conn = sqlite3.connect(
    "earthquakes.db"
)


df = pd.read_sql(
    "SELECT * FROM earthquakes",
    conn
)


conn.close()



# ==========================
# STATISTICS
# ==========================


c1,c2,c3,c4 = st.columns(4)


with c1:
    st.metric(
        "🌋 Events",
        len(df)
    )


with c2:
    st.metric(
        "📈 Avg Magnitude",
        round(
            df.magnitude.mean(),
            2
        )
    )


with c3:
    st.metric(
        "⚡ Maximum",
        df.magnitude.max()
    )


with c4:
    st.metric(
        "🌊 Avg Depth",
        round(
            df.depth.mean(),
            2
        )
    )



st.divider()



# ==========================
# AI PREDICTION
# ==========================


st.header(
    "🧠 Neural Network AI Analysis"
)


try:

    prediction = predict_latest_earthquake()


    a,b = st.columns(2)


    with a:

        if prediction["risk"] == "High":

            st.error(
                "🔴 HIGH RISK"
            )

        elif prediction["risk"] == "Medium":

            st.warning(
                "🟡 MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


    with b:

        st.metric(
            "Predicted Magnitude",
            prediction["magnitude"]
        )


except Exception as e:

    st.warning(
        str(e)
    )



st.divider()



# ==========================
# MODEL PERFORMANCE
# ==========================


st.header(
    "📊 Model Performance"
)


if os.path.exists(
    "model_metrics.json"
):

    with open(
        "model_metrics.json"
    ) as f:

        metrics=json.load(f)


    m1,m2,m3,m4 = st.columns(4)


    with m1:

        st.metric(
            "Accuracy",
            f"{metrics['classification']['accuracy']*100:.2f}%"
        )


    with m2:

        st.metric(
            "Precision",
            f"{metrics['classification']['precision']*100:.2f}%"
        )


    with m3:

        st.metric(
            "Recall",
            f"{metrics['classification']['recall']*100:.2f}%"
        )


    with m4:

        st.metric(
            "F1 Score",
            f"{metrics['classification']['f1_score']*100:.2f}%"
        )



st.divider()



# ==========================
# CONFUSION MATRIX
# ==========================


st.header(
    "🧩 Confusion Matrix"
)


if os.path.exists(
    "confusion_matrix.png"
):

    st.image(
        "confusion_matrix.png",
        caption="Risk Model Confusion Matrix"
    )

else:

    st.warning(
        "Confusion Matrix not generated yet"
    )



st.divider()



# ==========================
# GEMINI REPORT
# ==========================


st.header(
    "🤖 Gemini Seismic AI Report"
)


if st.button(
    "Generate AI Report"
):

    report = generate_ai_report()

    st.text_area(
        "Report",
        report,
        height=400
    )



st.divider()


# ==========================
# EARTHQUAKE INTELLIGENCE MAP
# ==========================

import folium
from streamlit_folium import st_folium


st.header(
    "🌍 Earthquake Intelligence Map"
)


center = [
    df.latitude.mean(),
    df.longitude.mean()
]


m = folium.Map(
    location=center,
    zoom_start=2
)



for _, row in df.tail(200).iterrows():


    if row["magnitude"] >= 6:

        color = "red"

    elif row["magnitude"] >= 4:

        color = "orange"

    else:

        color = "green"



    folium.CircleMarker(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=max(
            row["magnitude"] / 1.5,
            3
        ),

        popup=f"""
        Location: {row['place']}<br>
        Magnitude: {row['magnitude']}<br>
        Depth: {row['depth']} km
        """,

        color=color,

        fill=True,

        fill_color=color

    ).add_to(m)



st_folium(
    m,
    width=1200,
    height=600
)
# ==========================
# EARTHQUAKE TABLE
# ==========================


st.header(
    "🌎 Latest Earthquakes"
)


st.dataframe(
    df.tail(20),
    use_container_width=True
)