from allennlp.predictors.predictor import Predictor
from NER.NER import NER
import stanza


class NerStanza(NER):
    def __init__(self, tool='stanza'):
        super().__init__(tool=tool)

    def initialize_tool(self):
        self.NER = stanza.Pipeline("en", processors="tokenize,ner")

        if self.coreference_resolution:
            model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
            self.cr_predictor = Predictor.from_path(model_url)

    def detect_entities(self, document):

        if self.coreference_resolution:
            document = self.cr_predictor.coref_resolved(document)

        doc = self.NER(document)
        # return list(set(ent.text.lower() for ent in doc.ents if ent.type == 'PERSON'))
        detected_characters = list(set(ent.text.lower() for ent in doc.ents if ent.type in ['PERSON', 'ORG']))

        positions = [[(ent.start_char, ent.end_char) for ent in doc.ents if ent.text.lower() == c] for c in
                     detected_characters]

        return detected_characters, positions

