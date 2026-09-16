import pandas as pd
import numpy as np
import joblib
import json

import shap
import matplotlib.pyplot as plt


print("Loading data...")


# =========================
# LOAD DATA
# =========================


df = pd.read_csv(
    "risk_training_data.csv"
)


X = df.drop(
    "risk_level",
    axis=1
)



# =========================
# LOAD MODEL
# =========================


print("Loading model...")


model = joblib.load(
    "risk_model.pkl"
)



scaler = joblib.load(
    "risk_scaler.pkl"
)



# =========================
# PREPARE DATA
# =========================


X_scaled = scaler.transform(
    X
)



# أخذ عينة صغيرة للـ SHAP
# لأن البيانات الكبيرة بطيئة

sample = pd.DataFrame(

    X_scaled[:1000],

    columns=X.columns

)



# =========================
# SHAP EXPLAINER
# =========================


print("Running SHAP...")


explainer = shap.Explainer(

    model,

    sample

)



shap_values = explainer(

    sample

)



# =========================
# FEATURE IMPORTANCE
# =========================


# =========================
# HANDLE MULTI CLASS SHAP
# =========================


shap_array = shap_values.values


print(
    "SHAP shape:",
    shap_array.shape
)


if len(shap_array.shape) == 3:

    # samples, features, classes

    importance = np.abs(
        shap_array
    ).mean(
        axis=(0,2)
    )


else:

    importance = np.abs(
        shap_array
    ).mean(
        axis=0
    )



importance_df = pd.DataFrame(

{

"feature": X.columns,

"importance": importance

}

)



importance_df = importance_df.sort_values(

"importance",

ascending=False

)



print("\nFeature Importance")

print(
    importance_df
)



# =========================
# SAVE JSON
# =========================


importance_dict = dict(

zip(

importance_df["feature"],

importance_df["importance"].round(4)

)

)



with open(

"feature_importance.json",

"w"

) as f:


    json.dump(

        importance_dict,

        f,

        indent=4

    )



# =========================
# PLOT
# =========================


plt.figure(

figsize=(8,6)

)


plt.barh(

importance_df["feature"],

importance_df["importance"]

)


plt.xlabel(

"SHAP Importance"

)


plt.title(

"Earthquake Risk Feature Importance"

)


plt.gca().invert_yaxis()



plt.tight_layout()



plt.savefig(

"shap_feature_importance.png",

dpi=300

)



plt.close()



print("\n✅ Explainability completed")

print(
"Saved:"
)

print(
"shap_feature_importance.png"
)

print(
"feature_importance.json"
)