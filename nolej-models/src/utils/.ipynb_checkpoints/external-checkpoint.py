import requests
import logging

logger = logging.getLogger(__name__)

def get_grade_score(textA, textB, lang="en"):
    URL = "https://grade.dev.nolej.app/grade" # TODO adapt env

    data = {"lang":lang,
        "expected":textA,
        "given":textB}
    response = requests.post(URL, json=data)
    if response.status_code==200:
        try:
            result = response.json()['global_results'][0]['partial_results'][0]['score']
        except Exception as e:
            logger.warning("Result keys are : ".format(response.json().keys()))
            raise e
    else:
        raise Exception("Response status code was {}".format(response.status_code))
    return result