from src.NER.NER import NER
import spacy


class NerSpacy(NER):
    def __init__(self, tool='stanza'):
        super().__init__(tool=tool)

    def initialize_tool(self):
        self.NER = spacy.load("en_core_web_sm")

    def detect_entities(self, doc):
        return list(set(ent.text.lower() for ent in doc.ents if ent.label_ == 'PERSON'))
