import seisbench.models as sbm


models = {
    "PhaseNet": sbm.PhaseNet.from_pretrained("original"),
    "EQTransformer": sbm.EQTransformer.from_pretrained("original"),
    "GPD": sbm.GPD.from_pretrained("original")
}


results = {
    "dataset": "STEAD",
    "toolbox": "SeisBench",
    "task": "Phase Picking / Earthquake Detection",
    "models": [
        "PhaseNet",
        "EQTransformer",
        "GPD"
    ],
    "status": "Pretrained models loaded successfully"
}


print(results)