[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/39lKl_D-)
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/39lKl_D-)
# Natural language processing course 2022/23: `Challenges in Creating a Knowledge Base for Literacy Situations: Character Extraction and Analysis`

Team members:
 * `Matic Šuc`, `63180290`, `ms0181@student.uni-lj.si`
 * `Loris Štrosar Grmek`, `63180289`, `ls3453@student.uni-lj.si`
 
Group public acronym/name: `Literacy situation models knowledge base creation, Team 200`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Dataset
[55 Aesop fables](https://github.com/anzemur/literacy-knowledge-base/tree/main/data/aesop) from the Project Gutenberg website. Collected and annotated by previous year's team.

## Models used
NER
- [Spacy](https://spacy.io/models/en#en_core_web_sm)
- [NLTK](https://www.nltk.org/api/nltk.tokenize.punkt.html)
- [Stanza](https://stanfordnlp.github.io/stanza/ner.html)

Coreference Resolution: [AllenNLP](https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz)

Sentiment analysis: [BERT](https://huggingface.co/siebert/sentiment-roberta-large-english)