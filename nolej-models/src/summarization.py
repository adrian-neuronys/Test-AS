from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils.embeddings import get_sentences_embeddings, get_sentences_weights, reduce_cluster, get_lexrank_cut_indices
from .utils import split_text_in_sentences

def summarize_text(text, model_name="flax-community/t5-base-cnn-dm", CACHE={}):
    """
    Summarize text from a transformer corresponding to the model_name
    Args:
        text:
        model_name:
        CACHE:

    Returns:

    """
    if model_name in CACHE:
        model = CACHE[model_name]["model"]
        tokenizer = CACHE[model_name]["tokenizer"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        CACHE[model_name] = {"model": model,
                             "tokenizer": tokenizer}

    maximum_size = len(text.split())
    minimum_size = int(maximum_size / 3)

    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, min_length=minimum_size, max_length=maximum_size, num_beams=5,
                                   repetition_penalty=2.5, length_penalty=1, early_stopping=True,
                                   num_return_sequences=1)

    preds = [tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in
             generated_ids]
    return preds[0]


def cluster_sentences_for_bullet_points(sentences, sentences_weights=None, sentences_embeddings=None):
    """
    Cluster sentences for bullet_points summarization
    Args:
        sentences:
        sentences_weights:
        sentences_embeddings:

    Returns:

    """
    if sentences_embeddings is None:
        sentences_embeddings = get_sentences_embeddings(sentences)

    total_number_of_words = len(''.join(sentences).split())
    number_of_clusters = int(total_number_of_words / 300)

    if number_of_clusters < 3:
        minimum_cluster_size = 50
        number_of_clusters += 2
    elif number_of_clusters < 6:
        minimum_cluster_size = 65
        number_of_clusters += 3
    elif number_of_clusters < 9:
        minimum_cluster_size = 80
        number_of_clusters += 4
    else:
        minimum_cluster_size = 95
        number_of_clusters += 5

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters)
    try:
        clustering_model.fit(sentences_embeddings)
    except Exception as e:
        print(sentences_embeddings)
        raise e

    clustered_sentences_ids = {}
    for sentence_id, cluster_id in enumerate(clustering_model.labels_):
        clustered_sentences_ids.setdefault(cluster_id, []).append(sentence_id)

    if sentences_weights is None:
        sentences_weights = get_sentences_weights(sentences, sentences_embeddings=sentences_embeddings)

    sorted_weights_indices = np.argsort(sentences_weights)[::-1]

    # Creating clusters, and rearranging them internally regarding sentences weights
    clusters = []
    for cluster_id, sentence_ids in clustered_sentences_ids.items():
        sorted_sentence_ids = [i for i in sorted_weights_indices if i in sentence_ids]
        cluster = [sentences[i] for i in sorted_sentence_ids]

        if len(''.join(cluster).split()) > minimum_cluster_size:
            clusters.append(cluster)

    return clusters


def get_bullet_points(text=None, sentences=None, sentences_embeddings=None, sentences_weights=None,
                      max_cluster_size=300, model_name="flax-community/t5-base-cnn-dm", CACHE={}):
    """
    Generates bulllet_points summary from a text or list of sentences
    Args:
        text:
        sentences:
        max_cluster_size:
        CACHE:

    Returns:

    """
    if sentences is None:
        if text is None:
            raise Exception("text or sentences should be provided")
        sentences = split_text_in_sentences(text)

    if sentences_embeddings is None:
        sentences_embeddings = get_sentences_embeddings(sentences, CACHE=CACHE)
    if sentences_weights is None:
        sentences_weights = get_sentences_weights(sentences_embeddings=sentences_embeddings)

    new_indices = get_lexrank_cut_indices(sentences, sentences_weights=sentences_weights)

    clusters = cluster_sentences_for_bullet_points([sentences[i].strip() for i in new_indices],
                                             sentences_weights=[sentences_weights[i] for i in new_indices],
                                             sentences_embeddings=[sentences_embeddings[i] for i in new_indices])

    result = []
    for cluster in clusters:
        sentences_len = 0
        batch = []
        cluster_summaries = []
        for sentence in cluster:
            sentences_len += len(sentence.split())
            if sentences_len > max_cluster_size:
                summarie = summarize_text(' '.join(batch), model_name=model_name, CACHE=CACHE)
                cluster_summaries.append(summarie)
                batch = [sentence]
                sentences_len = len(sentence.split())
            else:
                batch.append(sentence)

        if len(batch) > 0:
            if len(batch) < 2 or (len(" ".join(batch).split()) < 10):  # Only one sentence or last batch is too short.
                print("End of batch have been ignored because too short")
            else:
                summarie = summarize_text(''.join(batch), model_name=model_name, CACHE=CACHE)
                cluster_summaries.append(summarie)

        to_join = []
        for item in cluster_summaries:
            item = item.strip()
            if not item.endswith("."):
                item = item + "."
            to_join.append(item)
        result.append(" ".join(to_join))


    return result