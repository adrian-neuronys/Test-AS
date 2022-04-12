from .models import QuestionGenerator, SpacyExtractor, QuestionAnswerer, SyntaxEvaluator, Linker
from .utils import  group_sentences, get_grade_score, get_explanation_from_position, get_substext_starts
from multiprocess.pool import ThreadPool

import logging, sys
from time import time


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CODE_VERSION  = "0.0.1"

class SpotLinkProcessor():
    def __init__(self):
        self.CACHED_MODELS = {}
        self.link_model = Linker()
        self.spacy_model = SpacyExtractor()

    def handle_message(self, body):
        sentences = body.get("sentences", [])
        document_id = body.get("document_id")

        logger.info("{} - Extracting links for {} sentence(s)".format(document_id, len(sentences)))
        start = time()
        link_result = self.link_model(sentences=sentences, cached_models=self.CACHED_MODELS)
        end = time()

        logger.info(link_result)
        logger.info('{} - SpotLink duration : {}'.format(document_id, end - start))

        content = {"spotlink":link_result,
                   "sentences":sentences,
                   "weights": body.get("weights"),
                   "document_id": body.get("document_id"),
                   "language": body.get("language"),
                   "paragraphs":body.get("paragraphs"),
                   "sections":body.get("sections")}

        logger.info('{} - Result : {}'.format(document_id, content))

        return content

class QGProcessor():
    def __init__(self, model_path=None, sagemaker_endpoint=None):
        self.qg_model = QuestionGenerator(model_path = model_path, sagemaker_endpoint=sagemaker_endpoint)
        self.CACHED_MODELS = {}
        #self.qa_model = QuestionAnswerer()
        #self.se_model = SyntaxEvaluator()

    def handle_message(self, body):
        sentences = body.get("sentences", [])
        paragraphs = body.get('paragraphs', [0] * len(sentences))
        sections = body.get('sections', [0] * len(sentences))
        answers = body.get("answers",[])

        answers_per_sentences = [[] for _ in sentences]

        for i, answer in enumerate(answers):
            if "sentence_id" in answer:
                answers_per_sentences[answer["sentence_id"]].append(answer)
            else:

                if "start" in answer:
                    starts = [answer["start"]]
                else:
                    starts = get_substext_starts(answer["text"], " ".join(sentences))
                    if len(starts)==0:
                        logger.warning(f"Answer {answer['text']} was not found in the text")
                        starts = []
                for start in starts:
                    _, sentence_id = get_explanation_from_position(start=start,
                                                  end=start+len(answer["text"]),
                                                  sentences=sentences)
                    answer["start"] = start
                    answers_per_sentences[sentence_id].append(answer.copy())
        groups = []
        if len(sentences) > 0:
            texts, groups = group_sentences(sentences, paragraphs, sections)

        #By grouping, we have to reajust the answer positions...
        position_in_full_text = 0
        for group in groups:
            sentences_group = [sentences[i] for i in group]
            answers_group = sum([answers_per_sentences[i] for i in group], [])
            for answer in answers_group:
                if "start" in answer:
                    answer["start"] -= position_in_full_text
                if "end" in answer:
                    answer["end"] -= position_in_full_text
            position_in_full_text += sum([len(s) + 1 for s in sentences_group])

        # Now generating questions per group
        logger.info(f" Generating question for {len(groups)} text(s)")
        start = time()

        data = []
        for group in groups:
            sentences_group = [sentences[i] for i in group]
            answers_group = sum([answers_per_sentences[i] for i in group], [])
            if len(answers_group)>0:
                data.append({"sentences":sentences_group,"answers":answers_group})
            else:
                data.append({"sentences":[],"answers":[]})

        # N_THREADS = 4
        # pool = ThreadPool(processes=N_THREADS)
        # result = pool.map(self.predict_model, data, chunksize=1)
        result = [self.predict_model(item) for item in data]

        result = self._format_result(result,
                                     groups=groups,
                                     remove_question_containing_answer=True)

        #logger.debug(result)

        end = time()

        logger.info(f'Question generation duration : {end-start}')

        #5 QUESTION FILTERING
        #flashcards = self.filter_flashcards(flashcards)

        content = {"qg": result}
        logger.info(f'Result : {content}')
        return content

    def predict_model(self, body):
        if len(body["sentences"]) > 0 and len(body["answers"]) > 0:
            return self.qg_model(sentences=body["sentences"], answers=body["answers"])
        else:
            return []

    def _format_result(self, results, groups, remove_question_containing_answer=True):
        """
        Format qg result into flashcards, with explanations, sections, and paragraph IDs.
        """
        formatted_result = []
        for i, result in enumerate(results):
            for flashcard in result:
                flashcard["sentence_id"] = groups[i][flashcard["sentence_id"]]
                if remove_question_containing_answer:
                    if flashcard['answer'].lower() in flashcard['question'].lower():
                        continue
                flashcard["verification_type"] = "semantic"
                formatted_result.append(flashcard)
        return formatted_result

    def filter_flashcards(self, flashcards, syntax_threshold=0.75, qa_check=True):
        if len(flashcards) == 0:
            return []
        logger.info("Checking {} flashcards".format(len(flashcards)))
        to_pop = set({})
        if qa_check:
            if self.qa_model is None:
                self.qa_model = QuestionAnswerer()
            questions = [item['question'] for item in flashcards]
            answers = [item['answer'] for item in flashcards]
            explanations = [item['explanation'] for item in flashcards]
            for i, question in enumerate(questions):
                gen_answer = self.qa_model(question=question, context=explanations[i])['answer']

                if (gen_answer.strip().lower() not in answers[i].strip().lower()) and (answers[i].strip().lower() not in gen_answer.strip().lower()):
                    try:
                        grade_score = get_grade_score(gen_answer, answers[i])
                        if grade_score<0.75:
                            logger.info("Removing question '{}' because expected answer is '{}' and QA answered '{}', which gave a grade_score of {}".format(question,
                                                                                                             answers[i],
                                                                                                             gen_answer,
                                                                                                             grade_score))
                            to_pop.add(i)
                    except Exception as e:
                        logger.warning("Something went wrong with grade call, keeping question but it should be checked")
                        logger.exception(e)


        if syntax_threshold:
            if self.se_model is None:
                self.se_model = SyntaxEvaluator()
            questions = [item['question'] for item in flashcards]
            scores = self.se_model(questions)
            for i, score in enumerate(scores):
                if score < syntax_threshold:
                    logger.info(
                        "Removing question '{}' because syntax evaluation score was {}".format(questions[i], score))
                    to_pop.add(i)
        return [item for i, item in enumerate(flashcards) if i not in to_pop]