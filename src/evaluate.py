import os, json, argparse
from utils import get_fables

def evaluate(args):

    fables = get_fables(args.dataset)

    protagonists = 0
    antagonists = 0
    sentiments = 0

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
        
        if args.antagonist:
            if results_detected["antagonist"] == results_annotations["antagonist"]:
                antagonists += 1
        
        if args.sentiments:
            if results_detected["sentiments"] == results_annotations["sentiments"]:
                sentiments += 1
                print(fable)

    if args.protagonist:
        print(f"Protagonist: {protagonists}/{len(fables)}")
    
    if args.antagonist:
        print(f"Antagonist: {antagonists}/{len(fables)}")
    
    if args.sentiments:
        print(f"Sentiments: {sentiments}/{len(fables)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aesop")
    parser.add_argument("--detected_entities", type=str, default=False)
    parser.add_argument("--protagonist", type=str, default=True)
    parser.add_argument("--antagonist", type=str, default=True)
    parser.add_argument("--sentiments", type=str, default=True)

    args = parser.parse_args()

    evaluate(args)