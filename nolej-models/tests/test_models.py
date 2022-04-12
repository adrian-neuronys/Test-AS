#!/usr/bin/env python

"""Tests for `uquiz_model_v2` package."""

import unittest

from src.models import Summarizer, QuestionGenerator, AnswerSelector, Linker, Highlighter, BaseModel, DLModel
from .assets import EN_LONG_TEXT
import logging
logger = logging.Logger(__name__)

class TestModelInstanciations(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_instanciate_base(self):
        model =  BaseModel()
        self.assertIsInstance(model, BaseModel)
        self.assertRaises(NotImplementedError, model.predict())

    def test_instanciate_summarizer(self):
        model = Summarizer()
        self.assertIsInstance(model, Summarizer)
        self.assertIsInstance(model, DLModel)
        self.assertIsInstance(model, BaseModel)

    def test_instanciate_summarizer_s3(self):
        model = Summarizer(model_path="s3://neuronys-datascience/models/tests/sumup")
        self.assertIsInstance(model, Summarizer)
        self.assertIsInstance(model, DLModel)
        self.assertIsInstance(model, BaseModel)


    def test_instanciate_qg(self):
        model = QuestionGenerator()
        self.assertIsInstance(model, QuestionGenerator)
        self.assertIsInstance(model, DLModel)
        self.assertIsInstance(model, BaseModel)

    def test_instanciate_as(self):
        model = AnswerSelector()
        self.assertIsInstance(model, AnswerSelector)
        self.assertIsInstance(model, DLModel)
        self.assertIsInstance(model, BaseModel)

    def test_instanciate_link(self):
        model = Linker()
        self.assertIsInstance(model, Linker)
        self.assertIsInstance(model, BaseModel)

    def test_instanciate_hl(self):
        model = Highlighter()
        self.assertIsInstance(model, Highlighter)
        self.assertIsInstance(model, BaseModel)

class TestSummarizationModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Summarizer()

    def test_prediction(self):
        self.model(EN_LONG_TEXT)

class TestASModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = AnswerSelector()

class TestQGModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = QuestionGenerator()

class TestLinkerModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Linker()

class TestHighlighterModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Highlighter()

if __name__ == "__main__":
    unittest.main()