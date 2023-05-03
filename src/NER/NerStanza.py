from src.NER.NER import NER
import stanza


class NerStanza(NER):
    def __init__(self, tool='stanza'):
        super().__init__(tool=tool)

    def initialize_tool(self):
        self.NER = stanza.Pipeline("en", processors="tokenize,ner")

    def detect_entities(self, doc):
        return list(set(ent.text.lower() for ent in doc.ents if ent.type == 'PERSON'))

