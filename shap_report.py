import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap



print("Loading dataset...")


df = pd.read_csv(
    "risk_training_data.csv"
)



features = [

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


X = df[features]

y = df["risk_level"]




X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



print("Training explainability model...")


model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"

)


model.fit(

    X_train,

    y_train

)




print("Running SHAP...")


explainer = shap.TreeExplainer(

    model

)


shap_values = explainer.shap_values(

    X_test.iloc[:500]

)



# ==========================
# FEATURE IMPORTANCE
# ==========================


importance = pd.DataFrame(

    {

        "feature": features,

        "importance":

        model.feature_importances_

    }

)



importance = importance.sort_values(

    "importance",

    ascending=False

)



print(importance)




importance.to_json(

    "shap_importance_report.json",

    orient="records",

    indent=4

)



# ==========================
# PLOT
# ==========================


plt.figure(

    figsize=(10,6)

)



plt.barh(

    importance["feature"],

    importance["importance"]

)



plt.gca().invert_yaxis()



plt.title(

    "Seismic AI Feature Importance"

)



plt.xlabel(

    "Importance"

)



plt.tight_layout()



plt.savefig(

    "shap_feature_importance_final.png",

    dpi=300

)



plt.close()



print("\n✅ SHAP Report Completed")