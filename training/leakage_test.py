import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE



print("Loading dataset...")


df = pd.read_csv(
    "risk_training_data.csv"
)


y = df["risk_level"]



# =========================
# Experiments
# =========================


experiments = {


"Full Features":

[

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

],



"Without Historical Magnitude":

[

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
"events_previous"

],



"Only Geographic + Time":

[

"depth",
"latitude",
"longitude",

"hour",
"month",
"day"

]

}



results = {}



for name, features in experiments.items():


    print("\n====================")

    print(name)


    X = df[features]



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    scaler = StandardScaler()



    X_train = scaler.fit_transform(
        X_train
    )


    X_test = scaler.transform(
        X_test
    )



    smote = SMOTE(
        random_state=42
    )


    X_train, y_train = smote.fit_resample(

        X_train,

        y_train

    )



    model = RandomForestClassifier(

        n_estimators=300,

        random_state=42

    )


    model.fit(

        X_train,

        y_train

    )


    pred = model.predict(

        X_test

    )



    acc = accuracy_score(

        y_test,

        pred

    )


    f1 = f1_score(

        y_test,

        pred,

        average="weighted"

    )


    results[name]={

        "accuracy":
        round(float(acc),4),

        "f1":
        round(float(f1),4)

    }



    print(results[name])




print("\n====================")

print("FINAL LEAKAGE RESULTS")

print(results)



import json


with open(

"leakage_results.json",

"w"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )


print("\nSaved leakage_results.json")