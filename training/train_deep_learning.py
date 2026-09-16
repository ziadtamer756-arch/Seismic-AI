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
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


print("Loading data...")

df = pd.read_csv(DATA_PATH)


# Features and target

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


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scaling

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


print("Building Deep Neural Network...")


model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    BatchNormalization(),

    Dropout(0.3),


    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.25),


    Dense(
        32,
        activation="relu"
    ),


    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


print("Training...")


history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


print("Evaluating...")


pred_prob = model.predict(
    X_test
)

pred = (
    pred_prob > 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)


results = {

    "model": "TensorFlow Deep Neural Network",

    "framework": "TensorFlow/Keras",

    "accuracy": float(accuracy),

    "f1_score": float(f1),

    "architecture": [
        "Dense(128)",
        "BatchNormalization",
        "Dropout(0.3)",
        "Dense(64)",
        "Dropout(0.25)",
        "Dense(32)",
        "Sigmoid Output"
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


model.save(
    os.path.join(
        MODEL_DIR,
        "deep_seismic_model.keras"
    )
)


print(results)

print("Deep Learning Training Completed")
