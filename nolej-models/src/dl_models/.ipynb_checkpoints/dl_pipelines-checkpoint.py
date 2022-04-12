import itertools
import logging

from collections.abc import Iterable
from ..utils import split_text_in_sentences
from ..utils.tools import get_list_dimension, find_answers_in_sentences

import torch
from transformers import(PreTrainedModel, PreTrainedTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES  = ["T5ForConditionalGeneration", "MT5ForConditionalGeneration", "PegasusForConditionalGeneration"]

class DLPipeline():
    def __init__(self, model=None, tokenizer=None, use_cuda: bool=True, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        self.model.to(self.device)
        model_class = self.model.__class__.__name__
        if model_class in ["T5ForConditionalGeneration", "MT5ForConditionalGeneration"]:
            self.model_type = "t5"
        elif model_class in ["PegasusForConditionalGeneration"]:
            self.model_type = "pegasus"
        elif model_class in ["MiniLMModel"]:
            self.model_type = "minilm"
        else:
            raise NotImplementedError("Model type '{}' is not available".format(model_class))

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

class AnswerSelectionPipeline(DLPipeline):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def __call__(self, context=None, sentences=None, max_length=32, mask=None, **kwargs):
        if sentences is None:
            if context is None:
                raise Exception("Context or sentences should be provided")
            sentences = split_text_in_sentences(context)

        inputs, analyzed_sentence_ids = self._prepare_inputs(sentences, mask=mask)
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

    def __call__(self, context=None, sentences=None, answers=None, predict_max_length=32, **kwargs):
        if sentences is None:
            if context is None:
                raise Exception("Context or sentences should be provided")
            sentences = split_text_in_sentences(context)

        if answers is None:
            if self.as_pipeline is None:
                raise NotImplementedError("You have to provide answers if no as_model has been set")
            else:
                logger.info("Generating answers")
                answers = self.as_pipeline(context=context, sentences=sentences)

        answers_dim = get_list_dimension(answers)

        if answers_dim>2:
            raise Exception("Incorrect format for answers")
        if answers_dim<2: # We need to locate answers in sentences
            answers = find_answers_in_sentences(answers, sentences)

        if len(list(itertools.chain(*answers))) == 0:
            return []

        examples = self._prepare_inputs(sentences, answers, **kwargs)

        inputs = [example['source_text'] for example in examples]

        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self._model_predict(input_ids=inputs['input_ids'].to(self.device),
                                   attention_mask=inputs['attention_mask'].to(self.device),
                                   num_beams=4,
                                   max_length=predict_max_length,
                                   use_cache=True)

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        output = [{'answer': item['answer'],
                   "verification_type":item["verification_type"],
                   'answer_start': item['answer_start'],
                   'answer_end': item['answer_end'],
                   'question': question,
                   'explanation': item['sentence'],
                   'sentence_id':item['sentence_id'],
                   "type":"open"} for item, question in zip(examples, questions)]
        return output

    def _prepare_inputs(self, sentences, answers, answers_max_len=5, t5_prefix="generate question :"):
        assert len(sentences)==len(answers), "len of sentences and len of answers should be the same, but are {} and {}".format(len(sentences), len(answers))
        inputs = []

        for i, sentence_answers in enumerate(answers):
            if len(sentence_answers) == 0: continue
            for answer in sentence_answers:

                if isinstance(answer, str):
                    answer = {"text":answer}

                answer_text = answer['text']
                sentence = sentences[i]
                sentences_copy = sentences[:]

                answer_text = answer_text.strip()

                if answers_max_len:
                    if len(answer_text.split()) > answers_max_len:
                        continue
                try:
                    ans_start_idx = sentence.lower().index(answer_text.lower())
                except Exception as e :
                    logger.warning("Could not find answer '{}' in sentence '{}', skipping question generation for this answer".format(answer_text, sentence))
                    continue
                ans_end_idx = ans_start_idx + len(answer_text)
                sentences_copy[i] = f"{sentence[:ans_start_idx]} <hl> {answer_text} <hl> {sentence[ans_end_idx: ]}"

                source_text = " ".join(sentences_copy)
                source_text = f"{t5_prefix} {source_text} </s>"

                inputs.append({"answer": answer_text,
                               "verification_type":answer.get("verification_type"),
                               "source_text": source_text,
                               "answer_start":ans_start_idx,
                               "answer_end":ans_end_idx,
                               "sentence":sentence,
                               "sentence_id":i})

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
