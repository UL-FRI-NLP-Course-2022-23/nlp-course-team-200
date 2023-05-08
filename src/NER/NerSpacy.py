from allennlp.predictors.predictor import Predictor
import spacy
from NER.NER import NER


class NerSpacy(NER):
    def __init__(self, tool='spacy'):
        super().__init__(tool=tool)

    def initialize_tool(self):
        self.NER = spacy.load("en_core_web_sm")
        
        if self.coreference_resolution:
            model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
            self.cr_predictor = Predictor.from_path(model_url)

    def detect_entities(self, document):

        if self.coreference_resolution:
            document = self.cr_predictor.coref_resolved(document)

        doc = self.NER(document)

        detected_characters = list(set(ent.root.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']))

        positions = [[(ent.start_char, ent.end_char) for ent in doc.ents if ent.root.text.lower() == c] for c in detected_characters]

        return detected_characters, positions
