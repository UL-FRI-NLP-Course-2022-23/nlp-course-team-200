import stanza
from utils import get_fable, get_fables

# Get fables
dataset = "aesop"
fables = get_fables(dataset)

# Download English models for the neural pipeline
stanza.download("en")

# Default neural pipeline in English
nlp = stanza.Pipeline("en", processors="tokenize,ner")

correctly_detected_entities_sum = 0
annotated_entities_sum = 0

for f in fables:
    fable, annotated_entities = get_fable(dataset, f)

    # NER on fable
    doc = nlp(fable)

    # Extract characters
    detected_entities = list(set(ent.text.lower() for ent in doc.ents if ent.type == 'PERSON'))

    # Count correctly detected entities
    correctly_detected_entities = sum([c in detected_entities for c in annotated_entities])

    correctly_detected_entities_sum += correctly_detected_entities
    annotated_entities_sum += len(annotated_entities)

    # Print
    print(f"{f}\t{correctly_detected_entities}/{len(annotated_entities)}")
    print(f"Annotated:\t{', '.join(annotated_entities)}")
    print(f"Detected:\t{', '.join(detected_entities)}")

print(f"Correctly detected:\t{correctly_detected_entities_sum}")
print(f"Annotated:\t{annotated_entities_sum}")
print(correctly_detected_entities_sum/annotated_entities_sum)