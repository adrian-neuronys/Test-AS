import logging

import spacy
import textrazor
from collections import Counter


logger = logging.getLogger(__name__)

TEXTRAZOR_API_KEY = 'e4984b1d0fbcc84d92fc81e6f7e132fd4e39b77cbac39153f86154e5'

def get_ner_spacy(text, CACHE={}):
    """
    NER spacy extractor
    Args:
        text:
        CACHE:

    Returns: {"text":str,
            "ner_label":str,
            "start":int,
            "end":int}

    """
    if "spacy_ner" not in CACHE:
        model = spacy.load("en_core_web_trf")
        CACHE['spacy_ner'] = model
    else:
        model = CACHE['spacy_ner']

    doc = model(text)

    result = {}

    for entity in doc.ents:
        positions = {'start': entity.start_char,
                     'end': entity.end_char,
                     'text': entity.text}
        if entity.text not in result:
            result[entity.text] = {"text": entity.text, "ner_label": entity.label_,"positions":[positions]}

        else:
            if entity.label_ == result[entity.text]["ner_label"]:
                result[entity.text]["positions"].append(positions)
    result = list(result.values())
        # Removing entities being subentities of others
    to_pop = []
    for k1, v1 in enumerate(result):
        for k2, v2 in enumerate(result):
            if k1 != k2:
                if v2['text'].lower() in v1['text'].lower():
                    to_pop.append(k2)
    logger.info("Found {} subentities to remove".format(len(to_pop)))
    result = [item for i, item in enumerate(result) if i not in to_pop]
    return result

def get_entities_textrazor(text, relevance_threshold=0., confidence_threshold=0., freebase_types_cardinality=3):
    """
    Get entities from a text using textrazor library
    :param text:
    :return:
    """


    logger.info("Extracting links in text")

    textrazor.api_key = TEXTRAZOR_API_KEY
    client = textrazor.TextRazor(extractors=["entities"])
    client.set_entity_allow_overlap(False)
    analyzed_text = client.analyze(text)

    # 1/ Entities. Looping over entities to extract positions and kb_id
    entities = {}
    all_freebase_types = []
    for entity in analyzed_text.entities():
        if entity.wikidata_id and \
                entity.wikidata_id[0] == 'Q' and\
                (len(entity.freebase_types) > 0) and\
                entity.relevance_score >= relevance_threshold and\
                entity.confidence_score >= confidence_threshold:

            positions = {'start': entity.starting_position,
                         'end': entity.ending_position,
                         'text': entity.matched_text}

            if (entity.id not in entities):
                entities[entity.id] = {"entity": entity.id,
                                       "kb_id": entity.wikidata_id,
                                       "relevance_score": [entity.relevance_score],
                                       "confidence_score": [entity.confidence_score],
                                       "dbpediatype": entity.dbpedia_types,
                                       "occurrences": 1,
                                       "positions": [positions],
                                       "freebase_types": entity.freebase_types,
                                       "enable": True}
            else:
                fb_types = set(entities[entity.id]["freebase_types"])
                [fb_types.add(k) for k in entity.freebase_types]
                entities[entity.id]["freebase_types"] = list(fb_types)
                entities[entity.id]["positions"].append(positions)
                entities[entity.id]["relevance_score"].append(entity.relevance_score)
                entities[entity.id]["confidence_score"].append(entity.confidence_score)
                entities[entity.id]["occurrences"] += 1

            all_freebase_types.extend(entity.freebase_types)

    logger.info("Found {} entities".format(len(entities)))

    # Removing entities being subentities of others
    to_pop = set({})
    for k1, v1 in entities.items():
        for k2, v2 in entities.items():
            if k1 != k2:
                if v2['entity'].lower() in v1['entity'].lower():
                    to_pop.add(k2)

    freebase_types_count = dict(Counter(all_freebase_types))
    freebase_types_to_keep = [k for k, count in freebase_types_count.items() if count >= freebase_types_cardinality]

    for k, v in entities.items():
        if all([fbt not in freebase_types_to_keep for fbt in v['freebase_types']]):
            to_pop.add(k)

    logger.debug("Found {} subentities to remove".format(len(to_pop)))
    for k in to_pop:
        entities.pop(k)

    # Taking max of relevance and confidence
    for k in entities.keys():
        entities[k]['relevance'] = float(max(entities[k].pop('relevance_score')))
        entities[k]['confidence'] = float(max(entities[k].pop('confidence_score')))

    # Keeping ony desired fieds and getting rid of dict format (just values)
    result = []
    for item in entities.values():
        result.append({k:item[k] for k in ["entity", "kb_id", "positions", "enable", "relevance", "confidence"]})
    
    return result

def get_tags_text_razor(text, max_size=2):
    """
    Get tags from text_razor topic's field
    Args:
        text:
        max_size:

    Returns:

    """
    textrazor.api_key = TEXTRAZOR_API_KEY
    client = textrazor.TextRazor(extractors=["topics"])
    analyzed_text = client.analyze(text)

    tags = []
    for topic in analyzed_text.topics():
        # keep only the tags with full confidence with 1 word max
        if (topic.score == 1) and (len(topic.label.split()) <= max_size):
            if topic.wikidata_id and topic.wikidata_id[0] == 'Q':
                tags.append({"text": topic.label.lower().strip(),
                             "kb_id": topic.wikidata_id})

    # Removing duplicates
    to_pop = set({})
    for i1, tag1 in enumerate(tags):
        for i2, tag2 in enumerate(tags):
            if i1 != i2:
                if tag1["text"] in tag2["text"]:
                    to_pop.add(i1)

    tags = [{"text": item["text"].lower(),
             "kb_id": item["kb_id"]} for i, item in enumerate(tags) if i not in to_pop]

    return tags

def disable_concepts(concepts):
    MIN_RELEVANCE = 0.3
    MIN_CONFIDENCE = 4.5
    result = []
    for concept in concepts:
        if (concept["confidence"] > MIN_CONFIDENCE):
            if (concept["relevance"] > MIN_RELEVANCE):
                concept["enable"] = True
                result.append(concept)
            else:
                if (concept["confidence"]> 10):
                    concept["enable"] = True
                    result.append(concept)
        else:
            if (concept["relevance"] > 0.5):
                concept["enable"] = False
                result.append(concept)
    return result