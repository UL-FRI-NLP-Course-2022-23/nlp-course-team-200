import os

import numpy as np
import stanza
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import json
from src.utils import get_fables, get_fable


sentiments_pipeline = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_pretokenized=True)



class SentRes:
    def __init__(self, dataset, title, document, characters, coocurrence_matrix, sentiment_matrix, index_to_character):
        self.dataset = dataset
        self.title = title
        self.document = document
        self.characters = characters
        self.coocurrence_matrix = coocurrence_matrix
        self.sentiment_matrix = sentiment_matrix
        self.index_to_character = index_to_character
        self.sentiment_matrix_normalized = None
        self.negative_threshold = -0.4
        self.positive_threshold = 0.4

    def normalize_sentiment_matrix(self):
        temp = np.copy(self.sentiment_matrix)
        divisor = np.max(temp) - np.min(temp)
        self.sentiment_matrix_normalized = temp / divisor

    def get_sentiment(self, character_1, character_2):
        if self.sentiment_matrix_normalized is None:
            raise Exception("Sentiment matrix is not normalized.")
        value = self.sentiment_matrix_normalized[character_1, character_2]
        if value >= self.positive_threshold:
            return 1.0
        elif value <= self.negative_threshold:
            return -1.0
        else:
            return 0.0

    def get_dictionary(self):
        res_dict = {}
        res_dict["characters"] = self.characters
        res_dict["protagonist"] = ""
        res_dict["antagonist"] = ""
        sent_dict = {}
        for i, character_1 in self.index_to_character.items():
            sent_character_dict = {}
            for j, character_2 in self.index_to_character.items():
                sent_character_dict[character_2] = self.get_sentiment(i, j)
            sent_dict[character_1] = sent_character_dict
        res_dict["sentiments"] = sent_dict
        print(json.dumps(res_dict))
        return res_dict


def preprocess_document(document):
    return document.replace('\n', ' ')


def compute_sentiment(document, characters):
    #
    index_to_character = {}
    for i, character in enumerate(characters):
        index_to_character[i] = character
    # preprocess document
    document = preprocess_document(document)
    # split document to sentences
    sentences_array = sent_tokenize(document)
    # initialize sentiment score array
    sentences_sentiments_array = [0.0 for _ in range(len(sentences_array))]
    # compute sentiment for each sentence
    for i, sentence in enumerate(sentences_array):
        res = sentiments_pipeline(sentence)
        if len(res.sentences) > 1:
            print("Problem")
        for tmp in res.sentences:
            sentences_sentiments_array[i] = tmp.sentiment
    # find character occurrences in document
    character_occurrences_counter = CountVectorizer(vocabulary=characters, binary=True)
    character_occurrences_in_sentences = character_occurrences_counter.fit_transform(sentences_array).toarray()
    cooccurrence_matrix = np.dot(character_occurrences_in_sentences.T, character_occurrences_in_sentences)
    sentiment_matrix = np.dot(character_occurrences_in_sentences.T,
                              (character_occurrences_in_sentences.T * sentences_sentiments_array).T)
    # same character sentiment is not useful
    np.fill_diagonal(cooccurrence_matrix, 0)
    np.fill_diagonal(sentiment_matrix, 0)
    # we have not computed from char1 to char2 and from char2 to char1, but only mutual sentiment
    # cooccurrence_matrix = np.tril(cooccurrence_matrix)
    # sentiment_matrix = np.tril(sentiment_matrix)
    return cooccurrence_matrix, sentiment_matrix, index_to_character


def save_results(sr):
    pass


if __name__ == '__main__':
    # document
    dataset = "aesop"
    fables = get_fables(dataset)
    results = []
    for fable in fables:
        title = fable
        fable_1, characters_1 = get_fable(dataset, title)
        coocurrence_matrix, sentiment_matrix, index_to_character = compute_sentiment(fable_1, characters_1)
        sr = SentRes(dataset, title, fable_1, characters_1, coocurrence_matrix, sentiment_matrix, index_to_character)
        sr.normalize_sentiment_matrix()
        results.append(sr)
        results_file = os.path.join("results_sentiment_2", dataset, f"{title}.json")
        json_object = json.dumps(sr.get_dictionary(), indent=4)

        # Writing to sample.json
        with open(results_file, "w") as out_file:
            out_file.write(json_object)



