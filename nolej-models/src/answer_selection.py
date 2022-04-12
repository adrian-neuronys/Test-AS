import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from .utils.s3_tools import download_s3_folder


nlp = spacy.load('en_core_web_trf')

def get_nounchunks(text, min_size=None, max_size=None, max_n=50):
    """
    Get spacy nounchunks from a list of sentences
    """

    DEFAULT_MIN_SIZE = 2
    DEFAULT_MAX_SIZE = 5

    min_size = min_size or DEFAULT_MIN_SIZE
    max_size = max_size or DEFAULT_MAX_SIZE

    result = {}

    doc = nlp(text)
    nounchunks = list(doc.noun_chunks)
    for nounchunk in nounchunks:
        len_nounchunk = len(nounchunk.text.split())
        if (len_nounchunk >= min_size) and (len_nounchunk <= max_size):
            # filter some non interesting nounchunk with part of speech parsing
            if doc[nounchunk.start].pos_ not in ["DET","PRON","CCONJ","PUNCT","INTJ"] and \
                    ("JJ" not in doc[nounchunk.start].tag_) and\
                    ("RB" not in doc[nounchunk.start].tag_) and\
                    nounchunk.text not in result:
                result[nounchunk.text] = {"text":nounchunk.text.lower()}
            else:
                # remove first word of the nounchunk
                _, _, rest = nounchunk.text.partition(" ")
                if len(rest.split()) > 1 and\
                    doc[nounchunk.start + 1].pos_ not in ["DET", "PRON", "CCONJ", "PUNCT", "INTJ"] != "DET" and \
                    ("JJ" not in doc[nounchunk.start + 1].tag_) and\
                    ("RB" not in doc[nounchunk.start + 1].tag_) and\
                    rest.lower() not in result:
                        result[nounchunk.text] = {"text":nounchunk.text.lower()}
    result = list(result.values())

    return result

def get_word_scores(text, words, limit=50, model_path="s3://neuronys-datascience/models/sentence_transformers/all-MiniLM-L6-v2", CACHE={}):
    if model_path not in CACHE:
        if model_path.startswith("s3://"):
            model_path = download_s3_folder(model_path)
        kw_model = KeyBERT(model=SentenceTransformer(model_path))
        CACHE["kw_model:"+model_path] = kw_model
    else:
        kw_model = CACHE["kw_model:"+model_path]

    return kw_model.extract_keywords(docs=text, stop_words='english', top_n=min(len(words), limit), candidates=words)

def get_ner(text, max_size=None):
    """
    Get spacy ner from a list of sentences
    """
    DEFAULT_MAX_SIZE = 5

    max_size = max_size or DEFAULT_MAX_SIZE

    doc = nlp(text)
    ners = list(doc.ents)
    
    result = {}
    for ner in ners:
        if (len(ner.text.split()) <= max_size):
            ALLOWED_NER_TYPES = ["GPE", "PERSON", "ORG", "PRODUCT", "EVENT", "PERCENT", "FAC", "MONEY", "WORK_OF_ART"]

            if ner.label_ in ALLOWED_NER_TYPES or ((ner.label_ == "DATE") and (doc[ner.start].is_digit)):

                if (doc[ner.start].pos_ != "DET"):
                    ner_text = ner.text.strip().lower()
                    result[ner_text] = {"text":ner_text}
                else: # Splitting out the DET ?
                    _, _, rest = ner.text.partition(" ")
                    if (doc[ner.start + 1].pos_ != "DET"):
                        ner_text = rest.strip().lower()
                        if rest.strip().lower() not in result:
                                result[ner_text] = {"text":ner_text}
    result = list(result.values())
    return result


def filter_result(result, return_indices_to_pop=False):
    """
    Removes items already include included other items
    Args:
        result:

    Returns:

    """
    to_pop = set({})
    for i1, tag1 in enumerate(result):
        for i2, tag2 in enumerate(result):
            if i1 != i2:
                if tag1 in tag2:
                    to_pop.add(i1)

    result = [item for i, item in enumerate(result) if i not in to_pop]
    if return_indices_to_pop:
        return result, to_pop
    return result