import os, json
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from visualization import visualize

def get_sentences(document):
    """
    Returns sentences with their indexes in original document.
    """
    sentences = []
    i = 0
    for sentence in sent_tokenize(document):
        sentences.append((sentence, (i, i + len(sentence))))
        i += len(sentence) + 1
    
    return sentences

def save_results(ner, fable_sentiments, protagonist, dataset):
    sentiments = dict()
    final_sentiments = dict()
    result_sentiments = dict()

    # Merge scores to labels
    for characters, sentiment, _ in fable_sentiments:
        if characters not in sentiments:
            sentiments[characters] = {"NEGATIVE": [], "POSITIVE": []}
        
        sentiment_label, sentiment_score = sentiment

        sentiments[characters][sentiment_label].append(sentiment_score)

    # Average both labels, choose bigger one
    for characters, sentiment in sentiments.items():
        negative = (sum(sentiment["NEGATIVE"]) / len(sentiment["NEGATIVE"])) if sentiment["NEGATIVE"] else 0
        positive = (sum(sentiment["POSITIVE"]) / len(sentiment["POSITIVE"])) if sentiment["POSITIVE"] else 0

        final_sentiments[characters] = 1 if positive > negative else -1
    
    # Build the results dict
    for character in ner.detected_annotations:
        result_sentiments[character] = {character: 0}

        for other_character in ner.detected_annotations:
            
            if character != other_character:

                result_sentiments[character][other_character] = 0

                for both_characters, sentiment in final_sentiments.items():
                    if character in both_characters and other_character in both_characters:
                        result_sentiments[character][other_character] = sentiment

    results = {"characters": ner.detected_annotations,
               "protagonist": protagonist,
               "antagonist": "",
               "sentiments": result_sentiments}

    results_file = os.path.join("results", dataset, f"{ner.title}.json")

    # Serializing json
    json_object = json.dumps(results, indent=4)
    
    # Writing to sample.json
    with open(results_file, "w") as out_file:
        out_file.write(json_object)

def get_protagonist(r):
    return r.detected_annotations[[len(l) for l in r.detected_annotations_positions].index(max([len(l) for l in r.detected_annotations_positions]))] if r.detected_annotations else ""

def sentiment_analysis(ner):

    sentiment = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

    for r in ner.ner_res:
        
        sentences = get_sentences(r.document)

        protagonist = get_protagonist(r)
        
        sentences_sentiment = [sentiment(s[0])[0] for s in sentences]

        fable_sentiments = []

        for i in range(len(sentences)):
            sentence = sentences[i][0]
            sentiment_label, sentiment_score = sentences_sentiment[i]["label"], sentences_sentiment[i]["score"]
            start, end = sentences[i][1]

            entities_in_sentence = []

            for i in range(len(r.detected_annotations_positions)):
                if any([s >= start and e <= end for s, e in r.detected_annotations_positions[i]]):
                    entities_in_sentence.append(r.detected_annotations[i])
            
            if len(entities_in_sentence) == 2:
                fable_sentiments.append((tuple(entities_in_sentence), (sentiment_label, sentiment_score), len(sentence)))
        
        save_results(r, fable_sentiments, protagonist, ner.dataset)
        visualize(r, fable_sentiments)