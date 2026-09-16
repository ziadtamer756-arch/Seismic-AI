import os
import json
import joblib
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "risk_training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EVAL_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)


# ==========================
# Load Data
# ==========================

print("Loading data...")

df = pd.read_csv(
    DATA_PATH
)


print("Dataset shape:")
print(df.shape)


# ==========================
# Features / Target
# ==========================

X = df.drop(
    "risk_level",
    axis=1
)

y = df["risk_level"]


# Convert categorical columns

X = pd.get_dummies(
    X,
    columns=[
        "season",
        "depth_category"
    ]
)


print("Features:")
print(X.columns.tolist())


# Number of classes

num_classes = len(
    y.unique()
)


# ==========================
# Train / Test Split
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

print("Scaling data...")


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)


joblib.dump(

    scaler,

    os.path.join(
        MODEL_DIR,
        "deep_scaler.pkl"
    )

)


# ==========================
# Deep Neural Network
# ==========================

print("Building Deep Neural Network...")


model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(
            X_train.shape[1],
        )
    ),

    BatchNormalization(),

    Dropout(
        0.3
    ),


    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.25
    ),


    Dense(
        32,
        activation="relu"
    ),


    Dense(
        num_classes,
        activation="softmax"
    )

])


model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


# ==========================
# Training
# ==========================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True

)


print("Training DNN...")


history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=64,

    callbacks=[
        early_stop
    ],

    verbose=1

)


# ==========================
# Evaluation
# ==========================

print("Evaluating...")


pred_prob = model.predict(
    X_test
)


pred = pred_prob.argmax(
    axis=1
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


report = classification_report(

    y_test,

    pred

)


print(report)


# ==========================
# Save Results
# ==========================

results = {

    "model": "Deep Neural Network",

    "framework": "TensorFlow/Keras",

    "task": "Multiclass Classification",

    "accuracy": float(
        accuracy
    ),

    "f1_score_weighted": float(
        f1
    ),

    "architecture": [

        "Dense(128)",

        "BatchNormalization",

        "Dropout(0.3)",

        "Dense(64)",

        "Dropout(0.25)",

        "Dense(32)",

        "Softmax Output"

    ],

    "regularization": [

        "Dropout",

        "BatchNormalization",

        "EarlyStopping"

    ]

}


with open(

    os.path.join(
        EVAL_DIR,
        "deep_learning_results.json"
    ),

    "w"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )


with open(

    os.path.join(
        EVAL_DIR,
        "dnn_classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)



# ==========================
# Save Model
# ==========================

model.save(

    os.path.join(
        MODEL_DIR,
        "deep_seismic_model.keras"
    )

)


print("\nFinal Results:")
print(results)


print("\n✅ DNN Deep Learning Training Completed")