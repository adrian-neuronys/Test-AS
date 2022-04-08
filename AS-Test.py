import spacy_streamlit
import spacy
import streamlit as st
import allennlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pysbd
import re
import sys
import pandas as pd
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
from summarizer.sbert import SBertSummarizer
from sentence_transformers import SentenceTransformer, util
print("lol")
print(sys.path)
PATH_TO_NOLEJ_MODELS = '/app/test-as/nolej_models'  
sys.path.append(PATH_TO_NOLEJ_MODELS)
print("lol")
print(sys.path)
from src.models import QuestionGenerator

@st.cache(allow_output_mutation=True)
def load_model(model):
    return spacy.load(model)

@st.cache(allow_output_mutation=True)
def load_allennlp(model):
    return Predictor.from_path(model)

@st.cache(allow_output_mutation=True)
def load_KeyBert(model):
    return SBertSummarizer(model)

@st.cache(allow_output_mutation=True)
def load_SentenceTransformer(model):
    return SentenceTransformer(model)


st.title("test AS")
default_text = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\"."

Roles = ["ARG0","ARG1","ARG2","ARG3","ARGM-ADV","ARGM-CAU","ARGM-CND","ARGM-DIR","ARGM-DIS","ARGM-EXT","ARGM-INS","ARGM-LOC","ARGM-MNR","ARGM-NEG","ARGM-PRED","ARGM-PRP","ARGM-TMP"]


texte = st.text_area("veuillez rentrer un texte à analyser",default_text)

choix = st.selectbox("veuillez choisir une méthode de générations de reponses", ("allennlp","Groupes Nominaux","Keybert1","Keybert2","Keybert3"), index=1)

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
    answers = []
    for i in range(len(clean_sentences)):
        res = predictor.predict(sentence=clean_sentences[i])
        
        for j in range(len(res["verbs"])):
            liste = re.split(r"[[[\]]", res["verbs"][j]["description"])
            for elt in liste:
                if elt.startswith(arg):
                    resu = troncation(elt)
                    if resu != None:
                        answers.append(troncation(elt))
    return answers
        

if choix == "allennlp":
    predictor = load_allennlp("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    st.write("Veuillez entrer le role semantique que vous souhaitez obtenir")
    arg = st.selectbox("veuillez choisir un argument semantique", (Roles), index=1)
    
    Answers = Get_role(texte,arg,)
    
    for elt in Answers:
        st.write(elt)

    ANSWERS = []
    for elt in Answers:
        ANSWERS.append({"text":elt})

    st.caption("Les resultats de Nolej :")
    model = QuestionGenerator(sagemaker_endpoint="inference-en-qg") 
    st.write(model(texte, answers=ANSWERS))

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
    for elt in candidats:
        st.write(elt)
        
    ANSWERSS = []
    for elt in candidats:
        ANSWERSS.append({"text":str(elt)})
    
    """
    st.caption("Les resultats de Nolej :")
    model = QuestionGenerator(sagemaker_endpoint="inference-en-qg") 
    st.write(model(texte, answers=ANSWERSS))
    """


if choix == "Spacy":
    
    vectorizer = KeyphraseCountVectorizer()
    model = load_KeyBert('all-MiniLM-L6-v2')
    nlp = load_model('en_core_web_trf')
    summary = model(texte)
    doc = nlp(summary)
    st.write(summary)
    
    data = []
    colonnes = ["index","token","pos","tag"]
    for token in doc:
        if token.tag_[0] == 'J' or token.tag_[0] == 'N':
            data.append([token.i,token.text,token.pos_,token.tag_])
    df = pd.DataFrame(data, columns = colonnes)
    st.table(df)
    
if choix == "Keybert2":
    
    vectorizer = KeyphraseCountVectorizer()
    model = load_KeyBert('all-MiniLM-L6-v2')
    nlp = load_model('en_core_web_trf')
    summary = model(texte)
    
    sentence_model = load_SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sentence_model)
    data2 = kw_model.extract_keywords(docs=summary, vectorizer=vectorizer, stop_words='english', use_mmr=True, diversity=0.7)
    
    st.caption("kw_model.extract_keywords(docs=summary, vectorizer=vectorizer, stop_words='english', use_mmr=True, diversity=0.7")
    colonne = ["terme","rate"]
    df2 = pd.DataFrame(data2, columns = colonne)
    st.table(df2)
    
if choix == "Keybert3":
    vectorizer = KeyphraseCountVectorizer()
    model = load_KeyBert('all-MiniLM-L6-v2')
    nlp = load_model('en_core_web_trf')
    summary = model(texte)
    doc = nlp(summary)
    sentence_model = load_SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sentence_model)
    
    candidates = []
    chunks = list(doc.noun_chunks)

    for chunk in chunks:
        # check nounchunk length
        l = len(chunk.text.split())
        if (l > 1):
            # filter some non interesting nounchunk with part of speech parsing
            if (doc[chunk.start].pos_ != "DET") and (doc[chunk.start].pos_ != "PRON") and (doc[chunk.start].pos_ != "CCONJ") and (doc[chunk.start].pos_ != "PUNCT") and (doc[chunk.start].pos_ != "INTJ"):
                st.write("lol")
                # avoid duplicates
                if chunk.text not in candidates:
                    # filter some non interesting nounchunk with constituency parsing
                    if ("JJ" not in doc[chunk.start].tag_) and ("RB" not in doc[chunk.start].tag_):
                        candidates.append(chunk.text)
            else:
                # remove first word of the nounchunk
                first, _, rest = chunk.text.partition(" ")
                if (doc[chunk.start+1].pos_ != "DET") and (doc[chunk.start+1].pos_ != "PRON") and (doc[chunk.start+1].pos_ != "CCONJ") and (doc[chunk.start+1].pos_ != "PUNCT") and (doc[chunk.start+1].pos_ != "INTJ"):
                    # avoid duplicates
                    if rest not in candidates:
                        # filter some non interesting nounchunk with constituency parsing
                        if ("JJ" not in doc[chunk.start+1].tag_) and ("RB" not in doc[chunk.start+1].tag_):
                            candidates.append(rest)
    st.caption("candidates : on enleve les groupes nominaux qui commencent par un adjectif")
    
    st.caption("kw_model.extract_keywords(docs=summary, stop_words='english', top_n=20, candidates=candidates")
    
    rates = kw_model.extract_keywords(docs=summary, stop_words='english', top_n=20, candidates=candidates)
    st.table(rates)
    
