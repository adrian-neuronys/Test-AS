from src.models import QuestionGenerator, AnswerSelector, Summarizer, Highlighter, Linker
import argparse
import time

text = 'Euronews space correspondent and European Space Agency (ESA) astronaut Luca Parmitano is documenting his experiences during his six-month stint onboard the International Space Station (ISS). In this episode, Luca is joined in the airlock by NASA astronaut Andrew Morgan. This week the team have carried out two spacewalks, the first in a series they will perform in order to repair the Alpha Magnetic Spectrometer, a particle physics experiment module which collects and analyses cosmic ray events. Morgan said:I’ve actually installed a cell biology incubator system. That took several days and a lot of deliberate time – to install the controller for that incubator, and put the actual incubator into the rack itself. As for our commander Luca, he has been installing a water recovery system. He explains:On the space station, we already recover 95 per cent of the water onboard. We recycle our urine, we recycle the humidity onboard the space station. The Japanese now have an experiment... that plans to do the same thing. It plans to recycle more water, to increase the capabilities, and probably this is something that will improve the design in the future, where we need something smaller in order to travel back and forth from further places. In collaboration with the European Space Agency.'


def predict(task="ask", model_path=None, onnx=False):
    task_dict = {"qg": QuestionGenerator,
                 "as":AnswerSelector,
                 "sum":Summarizer,
                 "hl":Highlighter,
                 "link":Linker}

    assert  task in task_dict, "Task {} is not in {}".format(task, list(task_dict.keys()))

    print("Loading model for task {}".format(task))
    start_time = time.time()
    model = task_dict[task](pretrained_model_name_or_path=model_path, onnx=onnx)
    print("Model loaded in {} seconds".format(time.time()-start_time))

    print("Running predictions")
    start_time = time.time()
    result = model(text)
    print("Predictions run in {} seconds".format(time.time()-start_time))

    return result

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qg", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--onnx", action="store_true", default=False)

    args = parser.parse_args()
    
    print(predict(task=args.task, model_path=args.model_path, onnx=args.onnx))