import pysbd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger(__name__)


def format_answers_with_context_for_t5(contexts, answers):
    """
    Formatting when providing answers for t5
    """
    if isinstance(answers, str):
        answers = [answers]
    if isinstance(contexts, str):
        contexts = [contexts]
    formated_texts = []
    for text, answer in zip(contexts, answers):
        formated_texts.append("answer : {} context : {}".format(answer, text))


def load_transformers_model(path, onnx = False):
    """
    Load transformers model from path with onnx option
    Args:
        path:
        onnx:

    Returns:

    """
    tokenizer = AutoTokenizer.from_pretrained(path)

    if onnx:
        raise Exception("Not Implemented")
       #model = OnnxT5(path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return model, tokenizer



def get_list_dimension(l):
    """
    Returns dimension of a list
    Args:
        l: List to test

    Returns:

    """
    if type(l) == list:
        if len(l)>0:
            return 1 + get_list_dimension(l[0])
        else:
            return 1
    else:
        return 0


def find_answers_in_sentences(answers, sentences, keep_only_one=True):
    """
    Search list answers in list of sentences
    Args:
        answers:
        sentences:
        keep_only_one:

    Returns:

    """
    result = [[] for _ in sentences]

    for answer in answers:
        if isinstance(answer, str):
            answer = {"text":answer}
        found = False
        for i, sent in enumerate(sentences):
            if answer["text"].lower() in sent.lower():
                result[i].append(answer)
                found=True
                if keep_only_one:
                    break
        if not found:
            logger.warning("Answer '{}' was not found in sentences".format(answer['text']))
    return result

def split_text_in_sentences(text, language="en"):
    """
    Split a text into sentences, regarding a specified language
    Args:
        text:
        language:

    Returns:

    """
    seg = pysbd.Segmenter(language=language, clean=True)
    return seg.segment(text)