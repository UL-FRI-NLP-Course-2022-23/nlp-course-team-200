import os, json

def get_fable(dataset, fable):
    """Returns fable and characters in it."""

    fable_path = os.path.join("corpora", dataset, "original", f"{fable}.txt")

    annotation_path = os.path.join("corpora", dataset, "annotations", f"{fable}.json")

    with open(fable_path, "r", encoding="utf-8") as f:
        fable = f.read()

    characters = json.load(open(annotation_path))["characters"]

    return fable, characters

def get_fables(dataset):
    """Returns a list of all fables in a dataset."""

    fables_path = os.path.join("corpora", dataset, "original")

    fables = [f.replace(".txt", "") for f in os.listdir(fables_path)]

    return fables