import os
import json
import joblib
import torch
import pandas as pd

from pytorch_tabnet.tab_model import TabNetClassifier

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
# Load Dataset
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


# Convert categorical features

X = pd.get_dummies(
    X,
    columns=[
        "season",
        "depth_category"
    ]
)


print("Features:")
print(X.columns.tolist())


# ==========================
# Train Test Split
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
        "tabnet_scaler.pkl"
    )

)


# ==========================
# Build TabNet
# ==========================

print("Building TabNet...")


model = TabNetClassifier(

    n_d=32,

    n_a=32,

    n_steps=5,

    gamma=1.5,

    lambda_sparse=1e-4,

    optimizer_fn=torch.optim.Adam,

    optimizer_params={
        "lr": 0.02
    },

    mask_type="entmax"

)


# ==========================
# Training
# ==========================

print("Training TabNet...")


model.fit(

    X_train,

    y_train.values,

    eval_set=[
        (
            X_test,
            y_test.values
        )
    ],

    eval_name=[
        "test"
    ],

    eval_metric=[
        "accuracy"
    ],

    max_epochs=100,

    patience=15,

    batch_size=256,

    virtual_batch_size=128,

    num_workers=0,

    drop_last=False

)


# ==========================
# Evaluation
# ==========================

print("Evaluating...")


pred = model.predict(
    X_test
)


accuracy = accuracy_score(

    y_test,

    pred

)


# Multiclass F1

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

    "model": "TabNet",

    "framework": "PyTorch TabNet",

    "task": "Multiclass Classification",

    "accuracy": float(
        accuracy
    ),

    "f1_score_weighted": float(
        f1
    ),

    "architecture": {

        "n_d": 32,

        "n_a": 32,

        "n_steps": 5,

        "optimizer": "Adam",

        "learning_rate": 0.02,

        "attention": "Sparse Feature Attention"

    },

    "regularization": [

        "Sparse Attention",

        "Early Stopping"

    ]

}


with open(

    os.path.join(
        EVAL_DIR,
        "tabnet_results.json"
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
        "tabnet_classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)



# Save model

model.save_model(

    os.path.join(

        MODEL_DIR,

        "tabnet_seismic_model"

    )

)


print("\nFinal Results:")
print(results)

print("\n✅ TabNet Training Completed")