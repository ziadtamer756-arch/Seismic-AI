import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score
)



print("Loading dataset...")


df = pd.read_csv(
    "risk_training_data.csv"
)



# ==========================
# TARGET
# ==========================

y = df["risk_level"]



# ==========================
# FEATURE GROUPS
# ==========================


all_features = [

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



without_history = [

    "depth",
    "latitude",
    "longitude",
    "hour",
    "month",
    "day",
    "day_of_year",
    "season",
    "depth_category",
    "lat_lon_product"

]



without_time = [

    "depth",
    "latitude",
    "longitude",
    "depth_category",
    "lat_lon_product",
    "events_previous",
    "avg_previous_magnitude"

]



geographic_only = [

    "depth",
    "latitude",
    "longitude"

]




experiments = {


    "Full Features":

    all_features,


    "Without Historical Features":

    without_history,


    "Without Time Features":

    without_time,


    "Only Geographic Features":

    geographic_only

}





results = {}



# ==========================
# RUN EXPERIMENTS
# ==========================


for name, features in experiments.items():


    print("\nRunning:" , name)



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




    model = RandomForestClassifier(

        n_estimators=300,

        random_state=42,

        class_weight="balanced"

    )



    model.fit(

        X_train,

        y_train

    )



    prediction = model.predict(

        X_test

    )



    accuracy = accuracy_score(

        y_test,

        prediction

    )


    f1 = f1_score(

        y_test,

        prediction,

        average="weighted"

    )



    results[name] = {


        "features_count":

        len(features),


        "accuracy":

        round(float(accuracy),4),


        "f1_score":

        round(float(f1),4)


    }



    print(results[name])





# ==========================
# SAVE
# ==========================


with open(

    "ablation_results.json",

    "w"

) as file:


    json.dump(

        results,

        file,

        indent=4

    )



print("\n====================")

print("FINAL ABLATION RESULTS")

print(results)


print("\n✅ Ablation Study Completed")