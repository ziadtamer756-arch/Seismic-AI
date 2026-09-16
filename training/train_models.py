import pandas as pd
import numpy as np
import json
import joblib

import matplotlib.pyplot as plt


from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

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


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier, MLPRegressor

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE



# =====================================================
# RISK CLASSIFICATION
# =====================================================


print("Loading risk dataset...")


risk_df = pd.read_csv(
    "risk_training_data.csv"
)


print(
    risk_df.shape
)



X = risk_df.drop(
    "risk_level",
    axis=1
)


y = risk_df["risk_level"]



# Train/Test

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# Scaling

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



print("\nClass distribution before SMOTE")

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



print("\nClass distribution after SMOTE")

print(
    y_train.value_counts()
)



# =====================================================
# BASELINE MODELS
# =====================================================


models = {


"Logistic Regression":

LogisticRegression(
    max_iter=2000
),



"Random Forest":

RandomForestClassifier(

    n_estimators=300,

    random_state=42

),



"XGBoost":

XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    random_state=42

),



"MLP Neural Network":

MLPClassifier(

    hidden_layer_sizes=(128,64,32),

    activation="relu",

    max_iter=700,

    random_state=42

)

}



results = {}

best_model = None

best_f1 = 0



for name, model in models.items():


    print("\nTraining:", name)


    model.fit(

        X_train_scaled,

        y_train

    )


    pred = model.predict(

        X_test_scaled

    )



    accuracy = accuracy_score(

        y_test,

        pred

    )


    precision = precision_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )


    results[name] = {


        "accuracy":

        round(float(accuracy),4),


        "precision":

        round(float(precision),4),


        "recall":

        round(float(recall),4),


        "f1":

        round(float(f1),4)

    }



    print(results[name])



    if f1 > best_f1:

        best_f1 = f1

        best_model = model





# =====================================================
# CROSS VALIDATION
# =====================================================


print("\nRunning 5 Fold Cross Validation...")


cv_scores = cross_val_score(

    best_model,

    X_train_scaled,

    y_train,

    cv=5,

    scoring="accuracy"

)



results["Cross Validation"] = {


"mean_accuracy":

round(float(cv_scores.mean()),4),


"std":

round(float(cv_scores.std()),4)


}




# =====================================================
# SAVE BEST RISK MODEL
# =====================================================


joblib.dump(

    best_model,

    "risk_model.pkl"

)



joblib.dump(

    scaler,

    "risk_scaler.pkl"

)



# =====================================================
# CONFUSION MATRIX
# =====================================================


prediction = best_model.predict(

    X_test_scaled

)



cm = confusion_matrix(

    y_test,

    prediction

)



plt.figure(

    figsize=(6,5)

)


plt.imshow(cm)


plt.title(
    "Risk Model Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.colorbar()


plt.savefig(

    "confusion_matrix.png",

    bbox_inches="tight"

)


plt.close()




# =====================================================
# MAGNITUDE REGRESSION
# =====================================================


print("\nLoading magnitude dataset...")


mag_df = pd.read_csv(

    "magnitude_training_data.csv"

)



X_mag = mag_df.drop(

    "magnitude",

    axis=1

)


y_mag = mag_df["magnitude"]




X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(

    X_mag,

    y_mag,

    test_size=0.2,

    random_state=42

)




mag_scaler = StandardScaler()



X_train_m = mag_scaler.fit_transform(

    X_train_m

)


X_test_m = mag_scaler.transform(

    X_test_m

)



mag_model = MLPRegressor(

    hidden_layer_sizes=(128,64,32),

    max_iter=700,

    random_state=42

)



mag_model.fit(

    X_train_m,

    y_train_m

)




mag_prediction = mag_model.predict(

    X_test_m

)



mae = mean_absolute_error(

    y_test_m,

    mag_prediction

)


rmse = np.sqrt(

    mean_squared_error(

        y_test_m,

        mag_prediction

    )

)



joblib.dump(

    mag_model,

    "magnitude_model.pkl"

)



joblib.dump(

    mag_scaler,

    "magnitude_scaler.pkl"

)




results["Magnitude Regression"] = {


"MAE":

round(float(mae),4),


"RMSE":

round(float(rmse),4)

}




# =====================================================
# SAVE RESULTS
# =====================================================


with open(

"model_comparison.json",

"w"

) as f:


    json.dump(

        results,

        f,

        indent=4

    )




print("\n========================")

print("FINAL RESULTS")

print(results)


print("\n✅ Research Training Completed")