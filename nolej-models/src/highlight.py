import logging

import numpy as np

from .utils import split_text_in_sentences, get_sentences_weights

logger = logging.getLogger(__name__)


def get_highlights(sentences=None, language="en", raw_text=None, cached_models=None, highlight_ratios=(0.25, 0.4)):
    """"
        Provides highlights
    """

    LANGUAGE_DICT = {'english': 'en', 'french': 'fr', 'german': 'de', 'spanish': 'es', 'italian': 'it',
                     'portuguese': 'pt',
                     'dutch': 'nl', 'polish': 'pl', 'russian': 'ru', 'chinese': 'zh', 'arabic': 'ar', 'korean': 'ko'}
    language = LANGUAGE_DICT.get(language, language)

    if sentences is None:
        if raw_text is None:
            raise Exception("You should provide raw_text if sentences are not provided")
        sentences = split_text_in_sentences(raw_text, language=language)

    logger.info("Extracting highlights for {} sentence(s)".format(len(sentences)))

    weights = get_sentences_weights(sentences)

    # We argsort so that the first element is the sentence with the highest score
    sorted_weights = np.argsort(-weights)

    result = {"sentences": sentences}

    for highlight_ratio in highlight_ratios:
        result["highlight_{}".format(int(highlight_ratio * 100))] = get_highlight(sorted_weights,
                                                                                  ratio=highlight_ratio)

    return result

def get_highlight(ordered_ids, ratio):
    highlited_ids = []
    l = round(len(ordered_ids) * ratio)
    highlited_ids.extend([int(item) for item in ordered_ids[:l]])
    highlited_ids = sorted(highlited_ids)

    return highlited_ids
