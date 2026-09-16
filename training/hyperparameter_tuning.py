import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE


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
# SMOTE
# ==========================

print("Applying SMOTE...")


smote = SMOTE(
    random_state=42
)


X_resampled, y_resampled = smote.fit_resample(
    X,
    y
)



# ==========================
# SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X_resampled,

    y_resampled,

    test_size=0.2,

    random_state=42,

    stratify=y_resampled
)



# ==========================
# SCALING
# ==========================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



# ==========================
# RANDOM FOREST
# ==========================

model = RandomForestClassifier(
    random_state=42
)



parameters = {

    "n_estimators":[100,200],

    "max_depth":[10,20,None],

    "min_samples_split":[2,5],

    "min_samples_leaf":[1,2]

}



grid = GridSearchCV(

    model,

    parameters,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)



print("Starting tuning...")


grid.fit(
    X_train_scaled,
    y_train
)



# ==========================
# TEST
# ==========================

best_model = grid.best_estimator_


pred = best_model.predict(
    X_test_scaled
)



accuracy = accuracy_score(
    y_test,
    pred
)


f1 = f1_score(
    y_test,
    pred,
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
        float(f1)

}



print(result)



with open(
    "hyperparameter_results.json",
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )



joblib.dump(
    best_model,
    "best_tuned_risk_model.pkl"
)


joblib.dump(
    scaler,
    "tuned_scaler.pkl"
)



print(
    "✅ Correct Hyperparameter Optimization Completed"
)