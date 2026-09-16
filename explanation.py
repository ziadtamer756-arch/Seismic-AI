import pandas as pd
import joblib
import shap
import yaml



# ==========================
# LOAD CONFIG
# ==========================

with open(
    "config/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)



MODEL_PATH = config["paths"]["risk_model"]



# ==========================
# LOAD MODEL
# ==========================

model = joblib.load(
    MODEL_PATH
)



# ==========================
# LOAD TRAINING DATA
# ==========================

data = pd.read_csv(
    "risk_training_data.csv"
)


X_train = data.drop(
    "risk_level",
    axis=1
)



# أخذ عينة للخلفية

background = X_train.sample(
    200,
    random_state=42
)



# ==========================
# EXPLAIN
# ==========================

def explain_prediction(features):


    explainer = shap.Explainer(

        model,

        background

    )


    shap_values = explainer(
        features
    )


    values = shap_values.values


    print(
        "SHAP shape:",
        values.shape
    )


    # Multi class

    if len(values.shape) == 3:


        values = abs(values).mean(
            axis=2
        )


    importance = abs(
        values[0]
    )



    result = pd.DataFrame({

        "feature":
            features.columns,


        "impact":
            importance

    })



    result = result.sort_values(

        "impact",

        ascending=False

    )


    return result.head(5)




# ==========================
# TEST
# ==========================

if __name__ == "__main__":


    sample = X_train.sample(
        1,
        random_state=10
    )


    print(
        explain_prediction(sample)
    )