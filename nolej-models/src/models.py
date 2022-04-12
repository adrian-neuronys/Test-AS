"""Main module."""
import logging

from .dl_models.dl_pipelines import QuestionGenerationPipeline, AnswerSelectionPipeline, SummarizationPipeline,\
    DistractorGenerationPipeline
from .highlight import get_highlights
from .link import get_entities_textrazor, get_ner_spacy, get_tags_text_razor, disable_concepts
from .answer_selection import get_nounchunks, get_ner, filter_result,get_word_scores
from .utils import load_transformers_model, split_text_in_sentences
from .utils.tools import get_list_dimension
from .utils.s3_tools import download_s3_folder
from collections.abc import Iterable
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATHS = {"SyntaxEvaluator":"salesken/query_wellformedness_score",
                       "QuestionGenerator":'s3://neuronys-datascience/models/final/en/qg',
                       "AnswerSelectorDL":'s3://neuronys-datascience/models/final/en/as',
                       "Summarizer":'s3://neuronys-datascience/models/final/en/sumup',
                       "DistractorGenerator":None}

class BaseModel():
    """
    Parent class returning the model results
    """
    def __init__(self, *args, **kwargs):
        self.version = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

# SIMPLE MODELS
class Highlighter(BaseModel):
    """
    Highlighter model returning highlights in a text or set of sentences
    """

    def predict(self, text=None, sentences=None, language="en", cached_models={}, **kwargs):
        if text is not None:
            if isinstance(text, str):
                result = get_highlights(sentences=sentences,
                                        raw_text=text,
                                        language=language,
                                        cached_models=cached_models)
            elif isinstance(text, Iterable):
                result = [self.predict(text=subtext, **kwargs) for subtext in text]
            else:
                raise Exception("Text should be str or Iterable")
        else:
            if sentences is None:
                raise Exception("You should provide text or sentences")
            return self.predict(sentences=sentences, **kwargs)

        return result

class Linker(BaseModel):
    """
    Linker searching for keywords in a texts or set of sentences
    """

    def __init__(self, **kwargs):
        self.MODEL_CACHE = {}

        super().__init__(**kwargs)

    def predict(self, text=None, sentences=None, language="en", **kwargs):
        assert (text is not None) or (sentences is not None), "'text' or 'sentences' should be provided"
        if text is None:
            text = " ".join(sentences)

        concepts = get_entities_textrazor(text=text)
        tags = get_tags_text_razor(text=text)
        nounchunks = get_nounchunks(text=text)
        nounchunks_scores = get_word_scores(text, [item['text'] for item in nounchunks], CACHE=self.MODEL_CACHE)
        nounchunks = [{"text": nounchunk["text"],
                       "score": nounchunk_score[1]} for nounchunk, nounchunk_score in zip(nounchunks, nounchunks_scores)]
        ner = []#get_ner(text=text)

        if (len(concepts)>0):
            concepts = disable_concepts(concepts=concepts)

        result = {"concepts": concepts,
                "tags": tags,
                "nounchunks":nounchunks,
                "ner":ner}

        return result

class SpacyEntityExtractor(BaseModel):
    """
    Model giving NER info
    """
    CACHE = {}
    def predict(self, text=None, sentences=None, language="en", **kwargs):
        assert text is not None or sentences is not None, "'text' or 'sentences' should be provided"
        if text is None:
            text = " ".join(sentences)
        result = get_ner_spacy(text=text, CACHE= self.CACHE)
        return result

class SpacyExtractor(BaseModel):

    def predict(self, text=None, sentences=None, language="en", **kwargs):
        assert text is not None or sentences is not None, "'text' or 'sentences' should be provided"
        if sentences is None:
            sentences = split_text_in_sentences(text)
        nounchunks = filter_result(get_nounchunks(sentences=sentences))
        ner = filter_result(get_ner(sentences))

        return {"nounchunks":nounchunks,
                "ner": ner}
        #answer_verification_types= ['semantic' for _ in nounchunks]
        #answer_verification_types.extend(['orthographic' for _ in ner])
        #answers, to_pop = filter_result(nounchunks + ner, return_indices_to_pop=True)

        #answer_verification_types = [item for i, item in enumerate(answer_verification_types) if i not in to_pop]

        #result = []
        #for answer, verification_type in zip(answers, answer_verification_types):
        #    result.append({"text":answer,
        #                   "verification_type":verification_type})
        #return result

# DEEP LEARNING MODELS
class DLModel(BaseModel):
    """
    A deeplearning predicton model based en transformers DeepLearning pipelines
    """
    def __init__(self, model_path=None, sagemaker_endpoint=None, **kwargs):
        self.loaded = False
        self.pipeline = None
        self.sagemaker_endpoint= sagemaker_endpoint

        if self.sagemaker_endpoint is None:
            if model_path is not None:
                self.model_path = model_path
                if model_path.startswith("s3://"):
                    logger.info(f"Downloading files from s3 {model_path}")
                    self.model_path = download_s3_folder(model_path)
                    logger.info(f"Files succesfully downloaded from s3 into {self.model_path}")
            else:
                raise Exception("A model_path or sagemaker_endpoint name should be provided")

        self.load(**kwargs)

        super().__init__(**kwargs)

    def load(self, **kwargs):
        raise NotImplementedError()

class SyntaxEvaluator(DLModel):
    def __init__(self, model_path=DEFAULT_MODEL_PATHS["SyntaxEvaluator"], **kwargs):
        super().__init__(model_path=model_path, **kwargs)

    def load(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, sentences, *args, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        features = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**features).logits
            return [item[0] for item in scores.tolist()]

class QuestionAnswerer(DLModel):
    def __init__(self, **kwargs):
        super().__init__(model_path=None, **kwargs)

    def load(self, *args, **kwargs):
      self.pipeline = pipeline('question-answering')

    def predict(self, question, context, *args, **kwargs):
        if isinstance(question, str):
            return self.pipeline({"question": question, "context": context})
        else:
            if isinstance(question, list):
                if isinstance(context, str):
                    context = [context for _ in question]
                assert len(context) == len(question)

                return [self.predict(question=q, context=c) for q, c in zip(question, context)]

# CUSTOM PIPELINES DLMODELS

class DLT5Model(DLModel):
    """
    A DeepLearning prediction model for T5
    """
    PIPELINE_CLASS = None

    def __init__(self, model_path, batch_size=8, sagemaker_endpoint=None, **kwargs):
        self.batch_size = batch_size
        super().__init__(model_path=model_path, sagemaker_endpoint=sagemaker_endpoint, **kwargs)

    def load(self, **kwargs):
        logger.info("Loading model")

        if self.sagemaker_endpoint is None:
            self.model, self.tokenizer = load_transformers_model(self.model_path)
        else:
            self.model, self.tokenizer = None, None

        self.pipeline = self.PIPELINE_CLASS(self.model, self.tokenizer,sagemaker_endpoint=self.sagemaker_endpoint, **kwargs)
        self.loaded = True

        logger.info(f"Model of type {str(self.PIPELINE_CLASS)} loaded")

    def predict(self, text=None, sentences=None, *args, **kwargs):
        if not self.loaded:
            self.load()
        if text is None:
            if sentences is None:
                raise Exception("text or sentences should be provided")
            else:
                sentences_dim = get_list_dimension(sentences)
                if sentences_dim==0:
                    raise Exception("sentences should be a list.")
                return self.pipeline(sentences=sentences, *args, **kwargs)
        else:
            return self.pipeline(context=text, *args, **kwargs)

class QuestionGenerator(DLT5Model):
    """
    Model returning automatic generated questions
    """
    PIPELINE_CLASS = QuestionGenerationPipeline

    def __init__(self, model_path=DEFAULT_MODEL_PATHS["QuestionGenerator"], **kwargs):
        super().__init__(model_path=model_path, **kwargs)

class AnswerSelectorDL(DLT5Model):
    """
    Model returning automatic generated answers
    """
    PIPELINE_CLASS = AnswerSelectionPipeline

    def __init__(self, model_path=DEFAULT_MODEL_PATHS["AnswerSelectorDL"], **kwargs):
        super().__init__(model_path=model_path, **kwargs)

class Summarizer(DLT5Model):
    """
    Model returning automatic generated summaries
    """
    PIPELINE_CLASS = SummarizationPipeline

    def __init__(self, model_path=DEFAULT_MODEL_PATHS["Summarizer"], **kwargs):
        super().__init__(model_path=model_path, **kwargs)

class DistractorGenerator(DLT5Model):
    """
    Model returning automatic generated distractors
    """
    PIPELINE_CLASS = DistractorGenerationPipeline

    def __init__(self, model_path=DEFAULT_MODEL_PATHS["DistractorGenerator"], **kwargs):
        super().__init__(model_path=model_path, **kwargs)