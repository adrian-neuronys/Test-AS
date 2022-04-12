import itertools
import logging

from collections.abc import Iterable
from ..utils import split_text_in_sentences
from ..utils.tools import get_list_dimension, find_answers_in_sentences, get_explanation_from_position, get_substext_starts

import torch
from transformers import(PreTrainedModel, PreTrainedTokenizer)
from sagemaker.huggingface import HuggingFacePredictor

logger = logging.getLogger(__name__)

MODEL_CLASSES  = ["T5ForConditionalGeneration", "MT5ForConditionalGeneration", "PegasusForConditionalGeneration"]

class DLPipeline():
    def __init__(self, model=None, tokenizer=None, use_cuda: bool=True, sagemaker_endpoint=None, model_type="t5", **kwargs):


        self.model = model
        self.tokenizer = tokenizer
        self.sagemaker_endpoint = sagemaker_endpoint


        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        if self.model is not None:
            self.model.to(self.device)

        if self.sagemaker_endpoint:
            if model_type is None:
                raise Exception("model_type should be provided with sagemaker_endpoint")
            self.model_type = model_type
        else:
            self.model_type = self.model.config_class.model_type

    def _model_predict(self, **kwargs):
        return self.model.generate(**kwargs)

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        """
        Tokenization of inputs
        """
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs

    def model_predict_sagemaker(self, inputs, params : {}):
        if self.sagemaker_endpoint is None:
            raise Exception("Sagemaker endpoint name is not specified")
        predictor = HuggingFacePredictor(endpoint_name = self.sagemaker_endpoint)
        outs = predictor.predict(data={"inputs":inputs,"parameters":params})
        return outs


class AnswerSelectionPipeline(DLPipeline):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def __call__(self, context=None, sentences=None, max_length=32, mask=None, **kwargs):
        if sentences is None:
            if context is None:
                raise Exception("Context or sentences should be provided")
            sentences = split_text_in_sentences(context)

        inputs, analyzed_sentence_ids = self._prepare_inputs(sentences, mask=mask)

        if self.sagemaker_endpoint is not None:
            outs = self.model_predict_sagemaker(inputs,params={})
            raw_answers = [out["generated_text"] for out in outs]

        else:
            inputs = self._tokenize(inputs, padding=True, truncation=True)

            outs = self._model_predict(input_ids=inputs['input_ids'].to(self.device),
                                   attention_mask=inputs['attention_mask'].to(self.device),
                                   max_length=max_length,
                                   use_cache=False,
                                   **kwargs)

            raw_answers = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        # Cleaning generated answers
        all_answers = [[] for _ in sentences]

        for answers_set, sentence_id in zip(raw_answers, analyzed_sentence_ids):
            explanation = sentences[sentence_id]
            content = answers_set.strip()
            if content.strip().endswith("<sep>"):
                content = content[:-5]
            content = content.split('<sep>')

            answers = []
            for answer in content:
                answer = answer.strip()
                if answer.lower() in explanation.lower():
                    answers.append(answer)
                else:
                    logger.warning("Answer '{}' could not be find in : '{}'".format(answer, explanation))
            all_answers[sentence_id] = answers

        return all_answers

    def _prepare_inputs(self, sentences=None, mask=None, t5_prefix="extract answers:"):
        inputs = []
        sentence_ids = []

        if mask is not None:
            assert len(sentences) == len(mask)

        for i in range(len(sentences)):

            if mask is not None:
                if not mask[i]:
                    continue

            source_text = t5_prefix
            for j, sent in enumerate(sentences):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()

            source_text = source_text + " </s>"

            inputs.append(source_text)
            sentence_ids.append(i)

        return inputs, sentence_ids

class QuestionGenerationPipeline(DLPipeline):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, as_model=None,
                 as_tokenizer=None, use_for_as=False, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.as_model = as_model
        self.as_tokenizer = as_tokenizer
        self.as_pipeline = None

        if (as_model is not None) and (as_tokenizer is not None):
            self.as_pipeline = AnswerSelectionPipeline(model=as_model, tokenizer=as_tokenizer, **kwargs)
        else:
            if use_for_as:
                logger.info("Using same qg model for answer selection. If the model was not trained for this, set use_for_as to False")
                self.as_pipeline = AnswerSelectionPipeline(model=model, tokenizer=tokenizer, **kwargs)
            else:
                logger.info("No as_model has been set, you should provide 'answers' at prediction time.")

    def __call__(self, context=None, sentences=None, answers=None, predict_max_length=32, num_beans=4,
                 early_stopping=True, **kwargs):
        if sentences is None and context is None:
            raise Exception("Context or sentences should be provided")
        if sentences is None:
            sentences = split_text_in_sentences(context)
        if context is None:
            context = " ".join(sentences)

        if answers is None:
            #if self.as_pipeline is None:
            #    raise NotImplementedError("You have to provide answers if no as_model has been set")
            #else:
            #    logger.info("Generating answers")
            #    answers = self.as_pipeline(context=context, sentences=sentences)
            raise NotImplementedError("answers should be provided")

        if len(list(itertools.chain(*answers))) == 0:
            return []

        corrected_answers = []
        if not isinstance(answers[0],dict) or "start" not in answers[0]:
            for answer in answers :
                starts = get_substext_starts(answer["text"], context)
                for start in starts:
                    corrected_answers.append({**answer, "start":start, "end":start+len(answer['text'])})
        else:
            corrected_answers = answers


        examples = self._prepare_inputs(context, corrected_answers, **kwargs)
        inputs = [example.pop("source_text") for example in examples]

        if len(inputs)==0:
            questions = []
        else:
            if self.sagemaker_endpoint is not None:
                outs = self.model_predict_sagemaker(inputs,params={})
                questions = [out["generated_text"] for out in outs]

            else:
                inputs = self._tokenize(inputs, padding=True, truncation=True)
                outs = self._model_predict(input_ids=inputs['input_ids'].to(self.device),
                                           attention_mask=inputs['attention_mask'].to(self.device),
                                           num_beams=num_beans,
                                           early_stopping=early_stopping,
                                           max_length=predict_max_length,
                                           use_cache=True)
                questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        output = [{**example, 'question': question} for example, question in zip(examples, questions)]
        return output

    def _prepare_inputs(self, text, answers, answers_max_len=5, t5_prefix="generate question :", **kwargs):
        inputs = []
        for answer in answers:

            answer_text = answer['text'].strip()
            answer_start = answer["start"]
            answer_end = answer.get("end") or answer_start+len(answer_text)
            explanation, explanation_id = get_explanation_from_position(answer_start, answer_end, text=text)
            if explanation is None or explanation_id is None:
                logger.warning(f"Explanation could not be found for answer {answer_text} with start {answer_start}")
                continue

            start_in_explanation = explanation.lower().find(answer_text.lower())
            if start_in_explanation == -1:
                continue
            end_in_explanation = start_in_explanation+len(answer_text)

            if answers_max_len:
                if len(answer_text.split()) > answers_max_len:
                    continue

            if self.model_type =="t5":
                source_text = f"{text[:answer_start]} <hl> {answer_text} </hl> {text[answer_end:]}"
                source_text = f"{t5_prefix} {source_text} </s>"
            elif self.model_type == "prophetnet":
                source_text = f"{answer_text} {self.tokenizer.sep_token} {text}"
            else:
                raise Exception(f"model_type {self.model_type} is unrecognized")



            inputs.append({"answer": answer_text,
                           "verification_type":answer.get("verification_type"),
                           "source_text": source_text,
                           "answer_start":start_in_explanation,
                           "answer_end":end_in_explanation,
                           "answer_id" : answer.get("answer_id"),
                           "explanation":explanation,
                           "score":answer.get("score"),
                           "sentence_id":explanation_id})

        return inputs

class SummarizationPipeline(DLPipeline):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)


    def __call__(self, context=None, sentences=None, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=3,
                 early_stopping=True, **kwargs):

        if context is None:
            if sentences is None:
                raise Exception("context or sentences should be provided")
            else:
                context = " ".join(sentences)

        inputs = self._prepare_inputs_for_summarization(context)
        inputs = self._tokenize(inputs, padding=True)

        outs = self._model_predict(input_ids=inputs['input_ids'].to(self.device),
                                   attention_mask=inputs['attention_mask'].to(self.device),
                                   max_length=max_length,
                                   min_length=min_length,
                                   length_penalty=length_penalty,
                                   num_beams=num_beams,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   early_stopping=early_stopping,
                                   **kwargs)
        result = []
        for out in outs:
            prediction = self.tokenizer.decode(out, skip_special_tokens=True)
            result.append(self._format_uppercase(prediction))
        return result

    def _prepare_inputs_for_summarization(self, context):
        """
        Preparing inputs for
        """
        if isinstance(context, Iterable) and not isinstance(context, str):
            inputs = []
            for text in context:
                inputs.extend(self._prepare_inputs_for_summarization(text))
            return inputs
        else:
            if self.model_type == "t5":
                source_text = "summarize: {}".format(context)
                source_text = source_text + " </s>"
            elif self.model_type == "pegasus":
                source_text = context
            else:
                raise NotImplementedError("Model type {} is not available for summarization".format(self.model_type))
            return [source_text]

    def _format_uppercase(self, text):
        """
        Format a text with uppercases at the start of sentences
        """
        sentences = split_text_in_sentences(text)
        formated_text = ""
        for sentence in sentences:
            formated_text += sentence[0].upper() + sentence[1:] + " "
        formated_text = formated_text.strip()
        if not formated_text.endswith('.'):
            formated_text += '.'

        return formated_text

class DistractorGenerationPipeline(DLPipeline):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)


    def __call__(self, context=None, sentences=None, questions=[], **kwargs):
        if sentences is None and context is None:
            raise Exception("Context or sentences should be provided")
        if sentences is None:
            sentences = split_text_in_sentences(context)
        if context is None:
            context = " ".join(sentences)


        inputs = self._prepare_inputs(context, questions)
        inputs = self._tokenize(inputs, padding=True)


        if len(inputs) == 0:
            distractors = []
        else:
            if self.sagemaker_endpoint is not None:
                outs = self.model_predict_sagemaker(inputs, params={})
                distractors = [out["generated_text"] for out in outs]

            else:
                outs = self._model_predict(input_ids=inputs['input_ids'].to(self.device),
                                           attention_mask=inputs['attention_mask'].to(self.device),
                                           num_beams=4,
                                           use_cache=True)
                distractors = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return distractors

    def _prepare_inputs(self, context, questions, t5_prefix="generate distractors"):
        inputs = []
        for question in questions:
            for answer in question.get("answers", []):
                input_text = f"{t5_prefix} : {context} <sep> {question['text']} <sep> {answer['text']} </s>"
                inputs.append(input_text)

        return inputs