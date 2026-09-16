import pandas as pd
import numpy as np
import json

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



print("Loading dataset...")


df = pd.read_csv(
    "risk_training_data.csv"
)



# ==========================
# FEATURES
# ==========================


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




# ==========================
# MODEL
# ==========================


model = Pipeline(

    [

        (
            "scaler",

            StandardScaler()

        ),


        (

            "classifier",

            RandomForestClassifier(

                n_estimators=300,

                random_state=42,

                class_weight="balanced"

            )

        )

    ]

)




# ==========================
# CROSS VALIDATION
# ==========================


cv = StratifiedKFold(

    n_splits=10,

    shuffle=True,

    random_state=42

)



print("Running 10 Fold Cross Validation...")



scores = cross_val_score(

    model,

    X,

    y,

    cv=cv,

    scoring="accuracy"

)



mean_accuracy = np.mean(scores)


std_accuracy = np.std(scores)



# 95% confidence interval

confidence_low = (

    mean_accuracy

    -

    1.96 *

    std_accuracy

    /

    np.sqrt(len(scores))

)



confidence_high = (

    mean_accuracy

    +

    1.96 *

    std_accuracy

    /

    np.sqrt(len(scores))

)




results = {


    "fold_scores":

    [

        round(float(x),4)

        for x in scores

    ],


    "mean_accuracy":

    round(float(mean_accuracy),4),


    "standard_deviation":

    round(float(std_accuracy),4),



    "95_confidence_interval":

    {

        "lower":

        round(float(confidence_low),4),


        "upper":

        round(float(confidence_high),4)

    }


}




print("\n====================")

print("VALIDATION RESULTS")

print(results)




with open(

    "validation_results.json",

    "w"

) as file:


    json.dump(

        results,

        file,

        indent=4

    )



print("\n✅ Validation Analysis Completed")