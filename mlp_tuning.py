import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


print("Loading dataset...")


df = pd.read_csv(
    "risk_training_data.csv"
)


X = df.drop(
    "risk_level",
    axis=1
)

y = df["risk_level"]



# ==========================
# Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ==========================
# Scaling
# ==========================

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



# ==========================
# MLP MODEL
# ==========================

mlp = MLPClassifier(

    random_state=42,

    max_iter=500

)



# ==========================
# Hyperparameters
# ==========================

parameters = {


    "hidden_layer_sizes":[

        (50,),

        (100,),

        (100,50)

    ],


    "alpha":[

        0.0001,

        0.001,

        0.01

    ],


    "early_stopping":[

        True,

        False

    ]

}



grid = GridSearchCV(

    mlp,

    parameters,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)



print(
    "Starting MLP tuning..."
)


grid.fit(

    X_train,

    y_train

)



best_model = grid.best_estimator_



prediction = best_model.predict(

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



result = {


    "best_parameters":

        grid.best_params_,


    "cv_accuracy":

        float(grid.best_score_),


    "test_accuracy":

        float(accuracy),


    "test_f1":

        float(f1),


    "regularization":

        {

            "alpha":

                grid.best_params_["alpha"],


            "early_stopping":

                grid.best_params_["early_stopping"]

        }

}



print(result)



with open(

    "evaluation/mlp_tuning_results.json",

    "w"

) as f:


    json.dump(

        result,

        f,

        indent=4

    )



joblib.dump(

    best_model,

    "models/best_mlp_model.pkl"

)


joblib.dump(

    scaler,

    "models/mlp_scaler.pkl"

)



print(
    "✅ MLP Regularization Tuning Completed"
)