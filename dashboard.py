import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import yaml

import folium
from streamlit_folium import st_folium


from ai_prediction import predict_latest_earthquake
from ai_report import generate_ai_report

from earthquake_data import get_recent_earthquakes
from database import save_earthquakes

tabs = st.tabs(
[
"🌎 Overview",
"🧠 AI Prediction",
"🏆 Baselines",
"🔁 Validation",
"🧪 Ablation",
"🔍 Explainability",
"🌍 Map",
"🤖 AI Report"
]
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(

    page_title="Seismic AI Platform",

    page_icon="🌎",

    layout="wide"

)



# ==========================
# CONFIG
# ==========================

with open(
    "config/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)



DATABASE_PATH = config["paths"]["database"]



# ==========================
# LANGUAGE
# ==========================

col1, col2 = st.columns([9,1])


with col2:

    language = st.selectbox(

        "🌐",

        [
            "English",
            "العربية"
        ]

    )



# ==========================
# TITLE
# ==========================


if language == "العربية":

    st.title(
        "🌎 منصة الذكاء الاصطناعي لمراقبة الزلازل"
    )


else:

    st.title(
        "🌎 Seismic AI Research Platform"
    )



st.caption(

"AI-powered earthquake monitoring, prediction and explainability system"

)



# ==========================
# UPDATE DATA
# ==========================


if st.button(
    "🔄 Update Latest Earthquakes"
):

    with st.spinner(
        "Updating seismic data..."
    ):


        try:


            events = get_recent_earthquakes()


            save_earthquakes(events)


            st.success(

                f"Updated {len(events)} events"

            )


            st.rerun()



        except Exception as e:


            st.error(e)




# ==========================
# LOAD DATABASE
# ==========================


@st.cache_data
def load_database():


    conn = sqlite3.connect(

        DATABASE_PATH

    )


    df = pd.read_sql(

        "SELECT * FROM earthquakes",

        conn

    )


    conn.close()


    return df




df = load_database()



# ==========================
# MODEL INFORMATION
# ==========================


st.info(

"""
🧠 Seismic AI Research Model


Training:

• Real earthquake event database
• Geological parameters
• Geographic coordinates
• Temporal patterns


Models:

• Random Forest Classifier
• XGBoost Classifier
• MLP Neural Network
• Magnitude Regression Model


Validation:

• Train/Test evaluation
• 5 Fold Cross Validation
• SMOTE balancing


Explainability:

• SHAP Feature Importance

"""

)




# ==========================
# STATISTICS
# ==========================


st.subheader(

"📊 Earthquake Statistics"

)



c1,c2,c3,c4 = st.columns(4)



with c1:

    st.metric(

        "🌋 Events",

        len(df)

    )


with c2:

    st.metric(

        "📈 Average Magnitude",

        round(

            df.magnitude.mean(),

            2

        )

    )



with c3:

    st.metric(

        "⚡ Maximum Magnitude",

        df.magnitude.max()

    )



with c4:

    st.metric(

        "🌊 Average Depth",

        round(

            df.depth.mean(),

            2

        )

    )



st.divider()
# ==========================
# AI PREDICTION
# ==========================


st.subheader(

    "🧠 AI Seismic Intelligence"

)


try:


    prediction = predict_latest_earthquake()



    c1, c2, c3 = st.columns(3)



    # Risk

    with c1:


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



        if prediction.get("confidence"):


            st.metric(

                "🤖 AI Confidence",

                f"{prediction['confidence']}%"

            )




    # Magnitude

    with c2:


        st.metric(

            "🌋 Predicted Magnitude",

            prediction["magnitude"]

        )




    # Depth

    with c3:


        st.metric(

            "🌊 Depth",

            f"{prediction['depth']} km"

        )





    st.info(

f"""

📍 Location:

{prediction['location']}


AI Analysis:

• Geographic position

• Depth information

• Historical seismic features

• Temporal patterns


"""

    )



except Exception as e:


    st.error(

        f"Prediction Error: {e}"

    )




st.divider()





# ==========================
# MODEL PERFORMANCE
# ==========================


st.subheader(

    "📊 Model Performance"

)



if os.path.exists(

    "evaluation/model_metrics.json"

):


    with open(

        "evaluation/model_metrics.json",

        "r"

    ) as file:


        metrics = json.load(file)




    a,b,c,d = st.columns(4)



    with a:

        st.metric(

            "Accuracy",

            f"{metrics['classification']['accuracy']*100:.2f}%"

        )


    with b:

        st.metric(

            "Precision",

            f"{metrics['classification']['precision']*100:.2f}%"

        )


    with c:

        st.metric(

            "Recall",

            f"{metrics['classification']['recall']*100:.2f}%"

        )


    with d:

        st.metric(

            "F1 Score",

            f"{metrics['classification']['f1_score']*100:.2f}%"

        )



st.divider()
# ==========================
# STATISTICAL VALIDATION
# ==========================

st.subheader(
    "🔁 Statistical Validation"
)


if os.path.exists(
    "evaluation/confidence_interval.json"
):

    with open(
        "evaluation/confidence_interval.json",
        "r"
    ) as f:

        validation = json.load(f)


    a,b,c = st.columns(3)


    with a:

        st.metric(
            "Accuracy",
            f"{validation['accuracy']*100:.2f}%"
        )


    with b:

        st.metric(
            "Confidence Level",
            validation["confidence_level"]
        )


    with c:

        st.metric(
            "95% Confidence Interval",
            f"{validation['lower_bound']*100:.2f}% - {validation['upper_bound']*100:.2f}%"
        )


else:

    st.info(
        "Confidence interval results not available"
    )

# ==========================
# ABLATION STUDY
# ==========================


st.subheader(
    "🧪 Feature Ablation Study"
)



if os.path.exists(

    "evaluation/ablation_results.json"

):


    with open(

        "evaluation/ablation_results.json"

    ) as f:


        ablation=json.load(f)



    ablation_df = pd.DataFrame(

        ablation

    ).T



    st.dataframe(

        ablation_df,

        use_container_width=True

    )


    st.bar_chart(

        ablation_df["accuracy"]

    )

# ==========================
# HYPERPARAMETER OPTIMIZATION
# ==========================

st.subheader(
    "⚙️ Hyperparameter Optimization"
)

if os.path.exists("evaluation/hyperparameter_results.json"):

    with open("evaluation/hyperparameter_results.json", "r") as f:
        hp = json.load(f)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "CV Accuracy",
            f"{hp.get('cv_accuracy', 0) * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Test Accuracy",
            f"{hp.get('test_accuracy', 0) * 100:.2f}%"
        )

    with c3:
        st.metric(
            "Test F1",
            f"{hp.get('test_f1', 0) * 100:.2f}%"
        )

    st.write("Best Parameters")
    st.json(hp.get("best_parameters", {}))

else:
    st.info("Hyperparameter results not available")


# ==========================
# AI SUMMARY
# ==========================


st.subheader(
    "🤖 AI Summary"
)



summary = f"""

The AI model classified this earthquake event as:

**{prediction['risk']} Risk**


Predicted magnitude:

**{prediction['magnitude']}**


Confidence level:

**{prediction['confidence']}%**


The decision was generated using:

- Geographic information
- Depth characteristics
- Temporal patterns
- Historical seismic features


"""


st.success(
    summary
)

# ==========================
# CONFUSION MATRIX
# ==========================


st.subheader(

    "🧩 Confusion Matrix"

)



if os.path.exists(

    "evaluation/confusion_matrix.png"

):


    st.image(

        "evaluation/confusion_matrix.png",

        caption="Risk Model Confusion Matrix"

    )


else:


    st.warning(

        "Confusion matrix not found"

    )




st.divider()
# ==========================
# GEMINI AI REPORT
# ==========================


st.subheader(

    "🤖 Gemini Seismic AI Report"

)



if st.button(

    "Generate AI Report"

):


    with st.spinner(

        "Generating report..."

    ):


        try:


            report = generate_ai_report()


            st.text_area(

                "AI Report",

                report,

                height=400

            )


        except Exception as e:


            st.error(e)




st.divider()





# ==========================
# EARTHQUAKE MAP
# ==========================


st.subheader(

    "🌍 Earthquake Intelligence Map"

)



center = [

    df.latitude.mean(),

    df.longitude.mean()

]



earthquake_map = folium.Map(

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

Location: {row['place']}

Magnitude: {row['magnitude']}

Depth: {row['depth']} km

""",


        color=color,


        fill=True,


        fill_color=color


    ).add_to(earthquake_map)




st_folium(

    earthquake_map,

    width=1200,

    height=600

)




st.divider()



# ==========================
# RESEARCH RESULTS
# ==========================

st.divider()

st.header(
    "🔬 Research Evaluation"
)


# Baseline

if os.path.exists("evaluation/model_comparison.json"):

    st.subheader(
        "🏆 Baseline Comparison"
    )


    with open(
        "evaluation/model_comparison.json"
    ) as f:

        baseline = json.load(f)


    baseline_df = pd.DataFrame(baseline).T


    st.dataframe(
        baseline_df,
        use_container_width=True
    )



# Ablation

if os.path.exists("evaluation/ablation_results.json"):

    st.subheader(
        "🧪 Ablation Study"
    )


    with open(
        "evaluation/ablation_results.json"
    ) as f:

        ablation = json.load(f)


    ablation_df = pd.DataFrame(ablation).T


    st.dataframe(
        ablation_df,
        use_container_width=True
    )



# SHAP

if os.path.exists(
    "evaluation/shap_importance_report.json"
):

    st.subheader(
        "🔍 Feature Importance"
    )


    with open(
        "evaluation/shap_importance_report.json"
    ) as f:

        shap_data = json.load(f)


    shap_df = pd.DataFrame(
        shap_data
    )


    st.dataframe(
        shap_df,
        use_container_width=True
    )



# ==========================
# ADDITIONAL VALIDATION REPORTS
# ==========================

st.subheader("📈 Confidence Interval Validation")

if os.path.exists("evaluation/confidence_interval.json"):
    with open("evaluation/confidence_interval.json", "r") as f:
        ci = json.load(f)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", f"{ci.get('accuracy',0)*100:.2f}%")
    with c2:
        st.metric("Confidence Level", ci.get("confidence_level","95%"))
    with c3:
        st.metric(
            "Interval",
            f"{ci.get('lower_bound',0)*100:.2f}% - {ci.get('upper_bound',0)*100:.2f}%"
        )


st.subheader("🧠 MLP Regularization Study")

if os.path.exists("evaluation/mlp_tuning_results.json"):
    with open("evaluation/mlp_tuning_results.json", "r") as f:
        mlp = json.load(f)

    st.json(mlp)


# ==========================
# EARTHQUAKE TABLE
# ==========================


st.subheader(

    "🌎 Latest Earthquakes"

)



st.dataframe(

    df.tail(20),

    use_container_width=True

)