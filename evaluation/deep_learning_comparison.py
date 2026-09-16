import os
import json


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EVAL_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)


results = {}


# ==========================
# Load JSON safely
# ==========================

def load_result(filename, model_name):

    path = os.path.join(
        EVAL_DIR,
        filename
    )

    if os.path.exists(path):

        with open(path, "r") as f:
            data = json.load(f)

        results[model_name] = {

            "accuracy": data.get(
                "accuracy",
                None
            ),

            "f1_score": data.get(
                "f1_score_weighted",
                data.get(
                    "f1_score",
                    None
                )
            ),

            "framework": data.get(
                "framework",
                ""
            )

        }

    else:

        print(
            f"Missing: {filename}"
        )


# ==========================
# Deep Learning Models
# ==========================

load_result(
    "deep_learning_results.json",
    "DNN"
)


load_result(
    "tabnet_results.json",
    "TabNet"
)


load_result(
    "wide_deep_results.json",
    "Wide_Deep"
)


# ==========================
# Save Comparison
# ==========================

output = os.path.join(
    EVAL_DIR,
    "deep_learning_comparison.json"
)


with open(
    output,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nDeep Learning Comparison")
print("========================")

for model, data in results.items():

    print(
        f"\n{model}"
    )

    print(
        f"Accuracy: {data['accuracy']}"
    )

    print(
        f"F1: {data['f1_score']}"
    )

    print(
        f"Framework: {data['framework']}"
    )


print(
    "\n✅ Deep Learning Comparison Generated"
)