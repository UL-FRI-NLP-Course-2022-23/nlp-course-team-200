import os, json, argparse
from utils import get_fables

def evaluate(args):

    fables = get_fables(args.dataset)

    protagonists = 0

    for fable in fables:

        # Annotation
        results_file = open(os.path.join("corpora", f"{args.dataset}", "annotations", f"{fable}.json"))
        results_annotations = json.load(results_file)

        # Detected
        results_file = open(os.path.join("results", f"{args.dataset}", f"{fable}.json"))
        results_detected = json.load(results_file)

        if args.protagonist:

            if results_detected["protagonist"] == results_annotations["protagonist"]:
                protagonists += 1
            else:
                print(f"{fable}\n{results_annotations['protagonist']}\t{results_detected['protagonist']}\n")
    
    if args.protagonist:
        print(f"Protagonist: {protagonists}/{len(fables)}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aesop")
    parser.add_argument("--detected_entities", type=str, default=False)
    parser.add_argument("--protagonist", type=str, default=True)
    parser.add_argument("--antagonist", type=str, default=False)
    parser.add_argument("--sentiments", type=str, default=False)

    args = parser.parse_args()

    evaluate(args)