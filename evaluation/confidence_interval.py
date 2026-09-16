import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("Loading data...")


df = pd.read_csv(
    "risk_training_data.csv"
)


X = df.drop(
    "risk_level",
    axis=1
)


y = df["risk_level"]



# Load tuned model

model = joblib.load(
    "models/best_tuned_risk_model.pkl"
)


# Load saved scaler

scaler = joblib.load(
    "tuned_scaler.pkl"
)



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


X_test = scaler.transform(
    X_test
)



pred = model.predict(
    X_test
)



accuracy = accuracy_score(
    y_test,
    pred
)


print(
    "Accuracy:",
    accuracy
)


# ==========================
# Bootstrap Confidence Interval
# ==========================

print("Calculating confidence interval...")


scores = []


rng = np.random.default_rng(
    42
)


for i in range(1000):

    indexes = rng.choice(

        len(y_test),

        len(y_test),

        replace=True

    )


    score = accuracy_score(

        y_test.iloc[indexes],

        pred[indexes]

    )


    scores.append(score)



lower = np.percentile(
    scores,
    2.5
)


upper = np.percentile(
    scores,
    97.5
)



result = {


    "accuracy":

        float(accuracy),


    "confidence_level":

        "95%",


    "lower_bound":

        float(lower),


    "upper_bound":

        float(upper)

}



print(result)



with open(

    "evaluation/confidence_interval.json",

    "w"

) as f:


    json.dump(

        result,

        f,

        indent=4

    )



print(
    "✅ Confidence Interval Generated"
)
with open(
    "evaluation/confidence_interval.json",
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


print(
    "Saved confidence_interval.json"
)