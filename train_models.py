import pandas as pd
import numpy as np
import json
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from imblearn.over_sampling import SMOTE



print("Loading prepared data...")


df = pd.read_csv(
    "prepared_earthquake_data.csv"
)


print(df.shape)



# =========================
# FEATURES
# =========================


features = [
    "depth",
    "latitude",
    "longitude",
    "hour",
    "month",
    "day"
]


X = df[features]


y_class = df["risk_level"]


y_reg = df["magnitude"]




# =========================
# RISK CLASSIFICATION
# =========================


print("\nPreparing Risk Model")



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_class,

    test_size=0.2,

    random_state=42,

    stratify=y_class

)



scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



print("\nBefore SMOTE")

print(
    y_train.value_counts()
)



smote = SMOTE(
    random_state=42
)



X_train_scaled, y_train = smote.fit_resample(

    X_train_scaled,

    y_train

)



print("\nAfter SMOTE")

print(
    y_train.value_counts()
)




print("\nTraining Neural Network")



risk_model = MLPClassifier(

    hidden_layer_sizes=(128,64,32),

    activation="relu",

    max_iter=700,

    random_state=42

)



risk_model.fit(

    X_train_scaled,

    y_train

)




risk_prediction = risk_model.predict(

    X_test_scaled

)



accuracy = accuracy_score(

    y_test,

    risk_prediction

)


precision = precision_score(

    y_test,

    risk_prediction,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    risk_prediction,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    y_test,

    risk_prediction,

    average="weighted",

    zero_division=0

)



print("\n===== Risk Results =====")


print("Accuracy:",accuracy)

print("Precision:",precision)

print("Recall:",recall)

print("F1:",f1)



report = classification_report(

    y_test,

    risk_prediction,

    zero_division=0

)



print(report)



# =========================
# CONFUSION MATRIX
# =========================


cm = confusion_matrix(

    y_test,

    risk_prediction

)



plt.figure(

    figsize=(6,5)

)


plt.imshow(cm)



plt.title(
    "Risk Classification Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.colorbar()



for i in range(len(cm)):

    for j in range(len(cm[i])):

        plt.text(

            j,

            i,

            cm[i][j],

            ha="center",

            va="center"

        )



plt.savefig(

    "confusion_matrix.png",

    bbox_inches="tight"

)



plt.close()



joblib.dump(

    risk_model,

    "risk_model.pkl"

)





# =========================
# MAGNITUDE REGRESSION
# =========================



print("\nTraining Magnitude Model")



X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(

    X,

    y_reg,

    test_size=0.2,

    random_state=42

)



X_train_r = scaler.fit_transform(

    X_train_r

)


X_test_r = scaler.transform(

    X_test_r

)




magnitude_model = MLPRegressor(

    hidden_layer_sizes=(128,64,32),

    activation="relu",

    max_iter=700,

    random_state=42

)



magnitude_model.fit(

    X_train_r,

    y_train_r

)




mag_prediction = magnitude_model.predict(

    X_test_r

)



mae = mean_absolute_error(

    y_test_r,

    mag_prediction

)



rmse = np.sqrt(

    mean_squared_error(

        y_test_r,

        mag_prediction

    )

)



print("\n===== Magnitude Results =====")


print("MAE:",mae)

print("RMSE:",rmse)



joblib.dump(

    magnitude_model,

    "magnitude_model.pkl"

)



joblib.dump(

    scaler,

    "scaler.pkl"

)




# =========================
# SAVE RESULTS
# =========================


metrics = {


"classification":{


"accuracy":round(float(accuracy),4),

"precision":round(float(precision),4),

"recall":round(float(recall),4),

"f1_score":round(float(f1),4)

},


"regression":{

"MAE":round(float(mae),4),

"RMSE":round(float(rmse),4)

}


}




with open(

"model_metrics.json",

"w"

) as f:

    json.dump(

        metrics,

        f,

        indent=4

    )




with open(

"classification_report.txt",

"w"

) as f:

    f.write(report)




print("\n✅ ALL DONE")

print(metrics)