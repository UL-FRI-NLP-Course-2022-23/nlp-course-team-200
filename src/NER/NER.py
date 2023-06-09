import time, json, os

from tqdm import tqdm

from utils import get_fables, get_fable


class NER:
    def __init__(self, tool='stanza', dataset="aesop"):
        self.tool = tool
        self.NER = None
        self.corpora = None
        self.dataset = dataset
        self.num_of_documents = 0
        self.ner_res = None
        self.computation_time = 0.0
        self.coreference_resolution = True
        self.get_corpora()

    def tool_defined(self):
        if self.NER is None:
            return False
        else:
            return True

    def initialize_tool(self):
        raise Exception("Not overriden!")

    def detect_entities(self, doc):
        raise Exception("Not overriden!")

    def get_corpora(self):
        self.corpora = get_fables(dataset=self.dataset)
        self.num_of_documents = len(self.corpora)
        self.ner_res = [None for _ in range(self.num_of_documents)]

    def get_document(self, n):
        return get_fable(self.dataset, self.corpora[n])

    def run(self):
        start_time = time.time()
        for i in tqdm(range(self.num_of_documents)):
            document, annotated_entities = get_fable(self.dataset, self.corpora[i])
            detected_entities, positions = self.detect_entities(document)
            self.ner_res[i] = NerRes(self.corpora[i], document, annotated_entities, detected_entities, positions)
        self.computation_time = time.time() - start_time

    def sum_outcomes(self):
        tp = 0
        fp = 0
        fn = 0
        for res in self.ner_res:
            tp += res.tp
            fp += res.fp
            fn += res.fn
        return tp, fp, fn

    def corpora_performance(self):
        tp, fp, fn = self.sum_outcomes()
        precision = NER.calculate_precision(tp, fp)
        recall = NER.calculate_recall(tp, fn)
        f_measure = NER.calculate_f_measure(precision, recall)
        return precision, recall, f_measure

    def save_results(self):
        results_file = os.path.join("results", f"{self.dataset}_{self.tool}{'_CR' if self.coreference_resolution else ''}.json")

        results = {ner_res.title: {"annotations": ner_res.annotations,
                                   "detections": ner_res.detected_annotations,
                                   "positions": ner_res.detected_annotations_positions,
                                   "tp": ner_res.tp,
                                   "fp": ner_res.fp,
                                   "fn": ner_res.fn} for ner_res in self.ner_res}

        # Serializing json
        json_object = json.dumps(results, indent=4)
        
        # Writing to sample.json
        with open(results_file, "w") as out_file:
            out_file.write(json_object)
        
        print(f"Saved results to {results_file}.")

    def load_results(self):
        results_file = open(os.path.join("results", f"{self.dataset}_{self.tool}{'_CR' if self.coreference_resolution else ''}.json"))
        results = json.load(results_file)

        for i, r in enumerate(results):
            document, annotated_entities = get_fable(self.dataset, r)
            self.ner_res[i] = NerRes(r, document, annotated_entities, results[r]["detections"], results[r]["positions"] if "positions" in results[r] else [])

    def visualize_result(self):
        # TODO
        pass

    @staticmethod
    def get_num_of_correctly_identified_entities(detected_entities, annotated_entities):
        return sum([c in detected_entities for c in annotated_entities])

    @staticmethod
    def calculate_precision(tp, fp):
        return tp / (tp + fp)

    @staticmethod
    def calculate_recall(tp, fn):
        return tp / (tp + fn)

    @staticmethod
    def calculate_f_measure(precision, recall):
        return 2 * ((precision * recall) / (precision + recall))


class NerRes:
    def __init__(self, title=None, document=None, annotations=None, detected_annotations=None, detected_annotations_positions=None):
        self.title = title
        self.document = document
        self.annotations = annotations
        self.detected_annotations = detected_annotations
        self.detected_annotations_positions = detected_annotations_positions
        self.num_of_detected_annotations = len(detected_annotations)
        self.num_of_annotations = len(annotations)
        self.tp = sum([c in self.detected_annotations for c in self.annotations])
        self.fp = sum([c not in self.annotations for c in self.detected_annotations])
        self.fn = sum([c not in self.detected_annotations for c in self.annotations])
        #self.precision = NER.calculate_precision(self.tp, self.fp)
        #self.recall = NER.calculate_recall(self.tp, self.fn)
        #self.f_measure = NER.calculate_f_measure(self.precision, self.recall)


'''
    def perform_ner_on_document(self, n):
        document, annotated_entities = get_fable(self.dataset, n)
        num_of_annotated_entities = len(annotated_entities)
        doc = NER(document)
        detected_entities = self.detect_entities(doc)
        correctly_detected_entities = self.get_num_of_correctly_identified_entities\
            (detected_entities, annotated_entities)
        return correctly_detected_entities, num_of_annotated_entities
'''