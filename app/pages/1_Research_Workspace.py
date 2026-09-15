import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from obspy import read



st.set_page_config(

    page_title="Research Workspace",

    page_icon="🌋",

    layout="wide"

)



st.title(
    "🌋 Seismic Research Workspace"
)


st.caption(
    "Waveform analysis and seismic signal processing environment"
)



# ==========================
# Upload File
# ==========================


uploaded_file = st.file_uploader(

    "Upload seismic file",

    type=[
        "mseed",
        "miniseed",
        "sac"
    ]

)



if uploaded_file:


    with open(

        "temp_seismic_file",

        "wb"

    ) as f:


        f.write(
            uploaded_file.read()
        )



    try:


        stream = read(
            "temp_seismic_file"
        )


        trace = stream[0]


        data = trace.data



        st.success(
            "Seismic file loaded successfully"
        )



        # ==========================
        # Metadata
        # ==========================


        col1,col2,col3,col4 = st.columns(4)



        col1.metric(

            "Station",

            trace.stats.station

        )


        col2.metric(

            "Sampling Rate",

            f"{trace.stats.sampling_rate} Hz"

        )


        col3.metric(

            "Duration",

            f"{trace.stats.duration:.2f} sec"

        )


        col4.metric(

            "Samples",

            len(data)

        )



        st.divider()



        # ==========================
        # Waveform
        # ==========================


        st.subheader(
            "📈 Waveform"
        )



        fig, ax = plt.subplots(

            figsize=(12,4)

        )


        ax.plot(
            data
        )


        ax.set_xlabel(
            "Samples"
        )


        ax.set_ylabel(
            "Amplitude"
        )


        ax.grid()



        st.pyplot(
            fig
        )



        # ==========================
        # Analysis
        # ==========================


        st.subheader(
            "📊 Signal Analysis"
        )



        a,b,c = st.columns(3)



        a.metric(

            "Maximum",

            round(
                float(np.max(data)),
                4
            )

        )


        b.metric(

            "Minimum",

            round(
                float(np.min(data)),
                4
            )

        )


        c.metric(

            "Mean",

            round(
                float(np.mean(data)),
                4
            )

        )



        # ==========================
        # AI Ready Section
        # ==========================


        st.info(

        """
        AI analysis pipeline ready.

        Future modules:

        - P Wave Detection
        - S Wave Detection
        - Earthquake Phase Picking
        - Magnitude Estimation
        - Machine Learning Classification

        """

        )



    except Exception as e:


        st.error(

            f"Error reading seismic file: {e}"

        )



else:


    st.warning(

        "Upload a seismic file to start analysis"

    )