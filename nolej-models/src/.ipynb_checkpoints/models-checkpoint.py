"""Main module."""
import logging

from .dl_models.dl_pipelines import QuestionGenerationPipeline, AnswerSelectionPipeline, SummarizationPipeline
from .highlight import get_highlights
from .link import get_entities_textrazor, get_ner_spacy, get_tags_text_razor, disable_concepts
from .answer_selection import get_nounchunks, get_ner, filter_result
from .utils import load_transformers_model, split_text_in_sentences
from .utils.tools import get_list_dimension
from .s3_tools import download_s3_folder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

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
        result = get_highlights(sentences=sentences,
                                raw_text=text,
                                language=language,
                                cached_models=cached_models)
        return result

    def __call__(self, texts=None, sentences=None, **kwargs):
        if texts is not None:
            if not isinstance(texts, str):
                return [self.predict(text=text, **kwargs) for text in texts]
            else:
                return self.predict(text=texts, **kwargs)
        else:
            if sentences is None:
                raise Exception("You should provide texts or sentences")
            else:
                return self.predict(sentences=sentences, **kwargs)

class Linker(BaseModel):
    """
    Linker searching for keywords in a texts or set of sentences
    """
    def predict(self, text=None, sentences=None, language="en", cached_models={}, **kwargs):
        assert text is not None or sentences is not None, "'text' or 'sentences' should be provided"
        if text is None:
            text = " ".join(sentences)
        entities = get_entities_textrazor(text=text)
        tags = get_tags_text_razor(text=text, max_size=2)

        if len(entities)>0:
            entities = disable_concepts(concepts=entities)

        return {'entities': entities,
                'tags': tags}

class EntityExtractor(BaseModel):
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

class AnswerSelector(BaseModel):
    def predict(self, text=None, sentences=None, language="en", **kwargs):
        assert text is not None or sentences is not None, "'text' or 'sentences' should be provided"
        if sentences is None:
            sentences = split_text_in_sentences(text)
        nounchunks = get_nounchunks(sentences=sentences)
        ner = get_ner(sentences)

        answer_verification_types= ['semantic' for _ in nounchunks]
        answer_verification_types.extend(['orthographic' for _ in ner])
        answers, to_pop = filter_result(nounchunks + ner, return_indices_to_pop=True)

        answer_verification_types = [item for i, item in enumerate(answer_verification_types) if i not in to_pop]

        result = []
        for answer, verification_type in zip(answers, answer_verification_types):
            result.append({"text":answer,
                           "verification_type":verification_type})
        return result

# DEEP LEARNING MODELS
class DLModel(BaseModel):
    """
    A deeplearning predicton model based en transformers DeepLearning pipelines
    """
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path.startswith("s3://"):
                logger.info("Downloading files from s3")
                self.pretrained_model_name_or_path = download_s3_folder(pretrained_model_name_or_path)
                logger.info("Files succesfully downloaded from s3")
            else:
                self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.pipeline = None

        self.load(**kwargs)
        super().__init__(**kwargs)

    def load(self, **kwargs):
        raise NotImplementedError()

class SyntaxEvaluator(DLModel):
    def __init__(self, pretrained_model_name_or_path="salesken/query_wellformedness_score", **kwargs):
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def load(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path)
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
        super().__init__(pretrained_model_name_or_path=None, **kwargs)

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

class DLT5Model(DLModel):
    """
    A DeepLearning prediction model for T5
    """
    PIPELINE_CLASS = None

    def __init__(self, pretrained_model_name_or_path, batch_size=8, **kwargs):
        self.batch_size = batch_size

        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def load(self, **kwargs):
        logger.info("Loading model")
        self.model, self.tokenizer = load_transformers_model(self.pretrained_model_name_or_path)
        self.pipeline = self.PIPELINE_CLASS(self.model, self.tokenizer, **kwargs)
        self.loaded = True
        logger.info("Model of type {} loaded".format(str(self.PIPELINE_CLASS)))

    def predict(self, context=None, sentences=None, *args, **kwargs):
        if not self.loaded:
            self.load()
        if context is None:
            if sentences is None:
                raise Exception("texts or sentences should be provided")
            else:
                sentences_dim = get_list_dimension(sentences)
                if sentences_dim==0:
                    raise Exception("sentences should be a list.")
                return self.pipeline(sentences=sentences, *args, **kwargs)

        else:
            return self.pipeline(context=context, *args, **kwargs)

class QuestionGenerator(DLT5Model):
    """
    Model returning answers and questions
    """
    PIPELINE_CLASS = QuestionGenerationPipeline

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path or 's3://neuronys-datascience/models/final/en/qg',
                         **kwargs)

class AnswerSelectorDL(DLT5Model):
    """
    Model returning answers and questions
    """
    PIPELINE_CLASS = AnswerSelectionPipeline

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path or 's3://neuronys-datascience/models/final/en/as',
                         **kwargs)

class Summarizer(DLT5Model):
    """
    Model returning answers and questions
    """
    PIPELINE_CLASS = SummarizationPipeline

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path or "s3://neuronys-datascience/models/final/en/sumup",
            **kwargs)