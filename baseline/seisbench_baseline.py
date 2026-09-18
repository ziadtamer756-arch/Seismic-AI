import seisbench.models as sbm


print("Loading SeisBench pretrained models...")


models = {

    "PhaseNet":
        sbm.PhaseNet.from_pretrained(
            "original"
        ),

    "EQTransformer":
        sbm.EQTransformer.from_pretrained(
            "original"
        ),

    "GPD":
        sbm.GPD.from_pretrained(
            "original"
        )

}


print("\nLoaded Models:")


for name in models:
    print(
        f"✅ {name}"
    )


print("\nSeisBench baseline models ready")