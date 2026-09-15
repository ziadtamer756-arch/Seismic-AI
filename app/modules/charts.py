import plotly.express as px
import streamlit as st



def magnitude_distribution(df):

    fig = px.histogram(

        df,

        x="magnitude",

        nbins=20,

        title="Magnitude Distribution"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



def depth_analysis(df):

    fig = px.scatter(

        df,

        x="magnitude",

        y="depth",

        size="magnitude",

        hover_name="place",

        title="Magnitude vs Depth"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



def location_activity(df):


    locations = (

        df["place"]
        .value_counts()
        .head(10)
        .reset_index()

    )


    locations.columns=[
        "Location",
        "Events"
    ]


    fig = px.bar(

        locations,

        x="Events",

        y="Location",

        orientation="h",

        title="Most Active Locations"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



def magnitude_time(df):


    fig = px.scatter(

        df,

        x="time",

        y="magnitude",

        title="Magnitude Timeline"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )