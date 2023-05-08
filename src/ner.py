import argparse
from NER.NerNltk import NerNltk
from NER.NerStanza import NerStanza
from NER.NerSpacy import NerSpacy

from sentiment_analysis import sentiment_analysis

# python src\\ner.py --ner stanza --coreference True --save_results True
# python src\\ner.py --ner spacy --coreference False --save_results True
# python src\\ner.py --ner nltk --coreference True --save_results True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aesop")
    parser.add_argument("--ner", type=str, default="spacy")
    parser.add_argument("--coreference", type=bool, default=True)
    parser.add_argument("--save_results", type=bool, default=False)
    parser.add_argument("--sentiment_analysis", type=bool, default=True)
    parser.add_argument("--load_ner_results", type=bool, default=True)

    args = parser.parse_args()

    if args.ner == "stanza":
        ner = NerStanza()
    elif args.ner == "spacy":
        ner = NerSpacy()
    elif args.ner == "nltk":
        ner = NerNltk()

    ner.coreference_resolution = args.coreference
    
    ner.initialize_tool()

    if args.load_ner_results:
        ner.load_results()
    else:
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

    if args.sentiment_analysis:
        sentiment_analysis(ner)