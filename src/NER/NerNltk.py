from allennlp.predictors.predictor import Predictor
from NER.NER import NER
import nltk


class NerNltk(NER):
    def __init__(self, tool='nltk'):
        super().__init__(tool=tool)
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

    def initialize_tool(self):
        self.NER = None

        if self.coreference_resolution:
            model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
            self.cr_predictor = Predictor.from_path(model_url)

    def detect_entities(self, document):

        if self.coreference_resolution:
            document = self.cr_predictor.coref_resolved(document)

        detected_entities = set()
        for sent in nltk.sent_tokenize(document):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    if chunk.label() == 'PERSON':
                        detected_entities.add(chunk[0][0].lower())
        return list(detected_entities), []
