import seisbench.data as sbd
import seisbench.models as sbm


print("Loading STEAD dataset...")


dataset = sbd.STEAD(
    root="./data/stead",
    download=True
)


print(dataset)


models = {

    "PhaseNet":
        sbm.PhaseNet.from_pretrained("original"),

    "EQTransformer":
        sbm.EQTransformer.from_pretrained("original"),

    "GPD":
        sbm.GPD.from_pretrained("original")

}


results = {}


for name, model in models.items():

    print("\nRunning:", name)

    sample = dataset[0]

    output = model.classify(
        sample
    )

    results[name] = str(output)

    print(output)



print("\n===== Baseline Results =====")

for k,v in results.items():
    print(k, ":", v)