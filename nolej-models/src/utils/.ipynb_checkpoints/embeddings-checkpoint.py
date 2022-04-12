import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import minmax_scale
from .lexrank import degree_centrality_scores


def get_sentences_embeddings(sentences, model_name="paraphrase-mpnet-base-v2", CACHE={}, device="cpu"):
    """
    Get sentences embeddings for a list a sentences, regarding the SentenceTransformer model provided by model_name
    Args:
        sentences:
        model_name:
        CACHE:

    Returns:

    """
    if model_name not in CACHE:
        embedder = SentenceTransformer(model_name, device=device)
        CACHE[model_name] = embedder
    else:
        embedder = CACHE[model_name]

    return embedder.encode(sentences, convert_to_tensor=False)


def get_sentences_weights(sentences=None, sentences_embeddings=None, model_name="paraphrase-mpnet-base-v2",
                          normalize=True,
                          threshold=0.2,
                          device="cpu",
                          CACHE={}):
    """
    Get LEXRANK sentences weights regarding provided embeddings, or computed from the model_name
    Args:
        sentences:
        sentences_embeddings:
        model_name:
        normalize:
        threshold:
        CACHE:

    Returns:

    """
    if sentences_embeddings is None:
        if sentences is None:
            raise Exception("You should provide sentences if embeddings are not provided")
        sentences_embeddings = get_sentences_embeddings(sentences, model_name=model_name, CACHE=CACHE, device=device)

    cos_scores = util.pytorch_cos_sim(sentences_embeddings, sentences_embeddings).numpy()
    centrality_scores = degree_centrality_scores(
        cos_scores, threshold=threshold)
    if normalize:
        centrality_scores = minmax_scale(centrality_scores)
    return centrality_scores


def reduce_cluster(sentences, limit=300):
    """
    Remove sentences to have the total number of words below a fixed limit (300 by default)
    Args:
        sentences:
        limit:

    Returns:

    """
    number_of_words = 0
    result = []
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if number_of_words + sentence_len < limit:
            number_of_words = number_of_words + sentence_len
            result.append(sentence)
        else:
            break
    return result


def get_lexrank_cut_indices(sentences, sentences_weights=None):
    """
    Cut array of sentences with lexrank weights, adapting rules regarding text size
    Args:
        sentences:
        sentences_weights:

    Returns:

    """
    joined_sentences = ''.join(sentences)
    # total number of words
    total_number_of_words = len(joined_sentences.split())

    # filters automatically adjust accord to document's length
    if total_number_of_words < 900:
        # remove 10%
        lexrank_cut = 0.1
    elif total_number_of_words < 1200:
        # remove 20%
        lexrank_cut = 0.2
    elif total_number_of_words < 2700:
        # remove 25%
        lexrank_cut = 0.25
    else:
        # remove 33%
        lexrank_cut = 0.33

    if sentences_weights is None:
        sentences_weights = get_sentences_weights(sentences)
    sorted_weights_indices = np.argsort(sentences_weights)[::-1]

    to_keep = sorted(sorted_weights_indices[:int(len(sentences) * (1 - lexrank_cut))])

    return to_keep