import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher


spacy.prefer_gpu()

def print_doc_analysis(doc):
    for token in doc:
        print ("Index: {} |  is_alpha {} | is_punct {} | like_num {} | is_title {} | POS {} | Text: {}".format(
            token.i, token.is_alpha, token.is_punct, token.like_num, token.is_title, token.pos_, token.text))

def print_doc_syn_dep(doc):
    for token in doc:
        print("{} {} {} {}".format(token.text, token.pos_, token.dep_, token.head.text))

def print_doc_named_entities(doc):
    for ent in doc.ents:
        print(ent.text, ent.label_)

def print_matcher_results(doc, matches):
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        print ("{} {} : {}".format(start, end, matched_span.text))
        
