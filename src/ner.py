import argparse
from NER.NerNltk import NerNltk
from NER.NerStanza import NerStanza
from NER.NerSpacy import NerSpacy

# python ner.py --ner stanza --coreference True --save_results True
# python ner.py --ner spacy --coreference True --save_results True
# python ner.py --ner nltk --coreference True --save_results True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aesop")
    parser.add_argument("--ner", type=str, required=True, default="stanza")
    parser.add_argument("--coreference", type=bool, default=False)
    parser.add_argument("--save_results", type=bool, default=False)

    args = parser.parse_args()

    if args.ner == "stanza":
        ner = NerStanza()
    elif args.ner == "spacy":
        ner = NerSpacy()
    elif args.ner == "nltk":
        ner = NerNltk()
    
    ner.initialize_tool()
    ner.run()
    precision, recall, f_measure = ner.corpora_performance()

    print(f"Dataset:\t{args.dataset}")
    print(f"NER:\t{args.ner}")
    print(f"CR:\t{args.coreference}")

    print(f"Pr:\t{precision:.2f}")
    print(f"Re:\t{recall:.2f}")
    print(f"F1:\t{f_measure:.2f}")    

    if args.save_results:
        ner.save_results()