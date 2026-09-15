import streamlit as st
import pandas as pd

from modules.data_loader import load_earthquakes
from modules.map import create_map
from modules.model_router import ModelRouter


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Seismic Research Dashboard",
    page_icon="🌎",
    layout="wide"
)



# =========================
# Title
# =========================

st.title("🌎 Seismic Research Dashboard")

st.caption(
    "Real-time earthquake monitoring and research analysis"
)



# =========================
# Load Data
# =========================

try:

    df = load_earthquakes()


except Exception as e:

    st.error(
        f"Database loading error: {e}"
    )

    st.stop()



# =========================
# Statistics
# =========================

if len(df) > 0:


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Events",
        len(df)
    )


    col2.metric(
        "Average Magnitude",
        round(
            df["magnitude"].mean(),
            2
        )
    )


    col3.metric(
        "Maximum Magnitude",
        round(
            df["magnitude"].max(),
            2
        )
    )


    col4.metric(
        "Average Depth km",
        round(
            df["depth"].mean(),
            2
        )
    )



else:

    st.warning(
        "No earthquake data available"
    )



st.divider()



# =========================
# Map
# =========================

st.subheader(
    "🌎 Earthquake Map"
)


try:

    earthquake_map = create_map(df)

    st.components.v1.html(
        earthquake_map._repr_html_(),
        height=600
    )


except Exception as e:

    st.error(
        f"Map error: {e}"
    )



st.divider()



# =========================
# AI Seismic Analyst
# =========================

st.header(
    "🤖 AI Seismic Research Analyst"
)



if st.button(
    "Analyze Current Activity"
):


    with st.spinner(
        "AI is analyzing seismic activity..."
    ):


        try:


            ai = ModelRouter()


            data = df.to_string()



            report = ai.run(

                "Analyze current earthquake activity scientifically. "
                "Identify patterns, magnitude distribution, "
                "depth behavior and research observations.",

                data

            )


            st.markdown(
                report
            )



        except Exception as e:


            st.error(
                f"AI Error: {e}"
            )



# =========================
# Data Table
# =========================

st.divider()


st.subheader(
    "📊 Earthquake Data"
)


st.dataframe(
    df,
    use_container_width=True
)