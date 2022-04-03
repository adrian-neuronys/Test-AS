import spacy_streamlit
import spacy
import streamlit as st
import allennlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pysbd
import re
import sys

@st.cache(allow_output_mutation=True)
def load_model(model):
    return spacy.load(model)

@st.cache(allow_output_mutation=True)
def load_allennlp(model):
    return Predictor.from_path(model)


st.title("test AS")
default_text = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\"."

Roles = ["ARG0","ARG1","ARG2","ARG3","ARGM-ADV","ARGM-CAU","ARGM-CND","ARGM-DIR","ARGM-DIS","ARGM-EXT","ARGM-INS","ARGM-LOC","ARGM-MNR","ARGM-NEG","ARGM-PRED","ARGM-PRP","ARGM-TMP"]


texte = st.text_input("veuillez rentrer un texte à analyser",default_text)

choix = st.selectbox("veuillez choisir une méthode de générations de reponses", ("allennlp","Groupes Nominaux"), index=1)

#segmentation en phrases
seg = pysbd.Segmenter(language="en", clean=True)
clean_sentences = seg.segment(texte)
MIN_NC = 1
ANSWERS = []

nlp = load_model('en_core_web_trf')

def troncation(elt):

    liste = elt.split()
    liste = liste[1:]
    phrase = " ".join(liste)
    
    doc = nlp(phrase)
    l = len(doc)
    if (l>MIN_NC):
        if (doc[0].pos_ != "DET") and (doc[0].pos_ != "PRON") and (doc[0].pos_ != "CCONJ") and (doc[0].pos_ != "PUNCT") and (doc[0].pos_ != "INTJ"):
            return phrase
        else:
            phrase = " ".join(liste)
            
            return troncation(phrase)

        
def Get_role(texte,arg):
    ANSWERS = []
    for i in range(len(clean_sentences)):
        res = predictor.predict(sentence=clean_sentences[i])
        
        for j in range(len(res["verbs"])):
            liste = re.split(r"[[[\]]", res["verbs"][j]["description"])
            for elt in liste:
                if elt.startswith(arg):
                    resu = troncation(elt)
                    if resu != None:
                        st.write(troncation(elt))
        

if choix == "allennlp":
    predictor = load_allennlp("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    st.write("Veuillez entrer le role semantique que vous souhaitez obtenir")
    arg = st.selectbox("veuillez choisir un argument semantique", (Roles), index=1)
    
    Get_role(texte,arg)
    
if choix == "Groupes Nominaux":
    doc = nlp(texte)
    candidats = []
    def troncation(chunk):
        l = len(chunk.text.split())
        if (l>MIN_NC):
            if (doc[chunk.start].pos_ != "DET") and (doc[chunk.start].pos_ != "PRON") and (doc[chunk.start].pos_ != "CCONJ") and (doc[chunk.start].pos_ != "PUNCT") and (doc[chunk.start].pos_ != "INTJ"):
                candidats.append(chunk)
            else:
                troncation(chunk[1:])
    
    for chunk in doc.noun_chunks:
        troncation(chunk)
    st.write(candidats)

PATH_TO_NOLEJ_MODELS = '/home/rim/nolej-models' 
sys.path.append(PATH_TO_NOLEJ_MODELS)
from src.models import QuestionGenerator

model = QuestionGenerator(sagemaker_endpoint="inference-en-qg") 
model(texte, answers=ANSWERS)
