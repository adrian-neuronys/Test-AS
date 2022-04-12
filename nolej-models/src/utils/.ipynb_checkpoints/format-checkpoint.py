import numpy as np

import logging

logger = logging.getLogger(__name__)

def get_n_words(sentence, seperator=" "):
    """
    Returns the length of words in a sentence
    """
    return len(sentence.strip().split(seperator))


def split_big_wagon(sentences_lens, limit):
    """
    Splitting homogeneously a group of sentences regarding their length
    """
    n_split = sum(sentences_lens) // limit + 1
    return [list(item) for item in np.array_split(range(len(sentences_lens)), n_split)]


def group_sentences(sentences, paragraphs, sections, max_group_size=300):
    """
    Group sentences into groups taking into account paragraphs, sections and a max_number of words per group.
    """
    texts = []
    groups = []
    for section_id in set(sections):
        ids = [i for i, item in enumerate(sections) if item == section_id]
        section_texts, section_groups  =  group_sentences_per_paragraph([sentences[id] for id in ids],
                                                        [paragraphs[id] for id in ids],
                                                        max_group_size=max_group_size)

        section_groups = [[ids[item] for item in group] for group in section_groups]

        texts += section_texts
        groups += section_groups

    return texts, groups


def group_sentences_per_paragraph(sentences, paragraphs, max_group_size=300):
    """
    Grouping sentences in paragraphs to split homogeneously big paragraphs, and to group small paragraph
    together.
    """
    assert len(sentences)==len(paragraphs), "Sentences and paragraphs should be the same length"
    if len(sentences)==0:
        logger.warning("Grouping empty list of sentences and paragraphs, returning empty results")
        return [], []
    final_wagons = []
    current_wagon = []
    wagon_space_left = max_group_size

    for paragraph_id in sorted(list(set(paragraphs))):
        sentence_ids = [i for i, paragraph in enumerate(paragraphs) if paragraph == paragraph_id]
        sentence_lens = [get_n_words(sentences[item]) for item in sentence_ids]
        paragraph_len = sum(sentence_lens) + len(sentence_lens)

        if paragraph_len > max_group_size:
            if len(current_wagon) > 0:
                final_wagons.append(current_wagon)
                wagon_space_left = max_group_size
                current_wagon = []
            wagon_ids = split_big_wagon(sentence_lens, max_group_size)
            [final_wagons.append([sentence_ids[index] for index in ids]) for ids in wagon_ids]
        else:
            if paragraph_len > wagon_space_left:
                final_wagons.append(current_wagon)
                wagon_space_left = max_group_size
                current_wagon = []
            current_wagon.extend(sentence_ids)
            wagon_space_left = wagon_space_left - paragraph_len

    if len(current_wagon) > 0:
        final_wagons.append(current_wagon)
    texts = [' '.join([sentences[index] for index in indices]) for indices in final_wagons]
    groups = final_wagons
    return texts, groups