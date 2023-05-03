from src.NER.NerNltk import NerNltk
from src.NER.NerStanza import NerStanza
from src.NER.NerSpacy import NerSpacy

def ners_runner():
    ner = NerSpacy()
    ner.initialize_tool()
    ner.run()
    precision, recall, f_measure = ner.corpora_performance()
    print("Spacy results")
    print(f"Precision, recall, f_measure:\t{precision} \t{recall} \t{f_measure}")

    ner = NerStanza()
    ner.initialize_tool()
    ner.run()
    precision, recall, f_measure = ner.corpora_performance()
    print("Stanza results")
    print(f"Precision, recall, f_measure:\t{precision} \t{recall} \t{f_measure}")


if __name__ == "__main__":
    ner = NerNltk()
    ner.initialize_tool()
    ner.run()
    precision, recall, f_measure = ner.corpora_performance()
    print("NLTK results")
    print(f"Precision, recall, f_measure:\t{precision} \t{recall} \t{f_measure}")



