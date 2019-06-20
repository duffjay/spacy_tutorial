import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from less01 import print_doc_analysis

spacy.prefer_gpu()
nlp = English()
nlp = spacy.load("en_core_web_sm")

def print_entities(doc):
    for ent in doc.ents:
        print('{} {}'.format(ent.text, ent.label_))

coffee_hash = nlp.vocab.strings['I love coffee']
# this won't work - because it has noever gone through nlp()
# coffee_string = nlp.vocab.strings[coffee_hash]
# print (coffee_hash, cofffee_string)

# once it goes through nlp() - it's in the data structure
doc = nlp("I love coffee")
print ('hash value:', nlp.vocab.strings['coffee'])
print ('string value:', nlp.vocab.strings[3197928453018144401])

# lexeme
lexeme = nlp.vocab['coffee']
print (lexeme.text)
print (lexeme.orth)
print (lexeme.is_alpha)

# strings to hashes
print ('-- cat --')
doc = nlp("I have a cat")

cat_hash = nlp.vocab.strings['cat']
print ('cat hash:', cat_hash)
print ('cat text:', nlp.vocab.strings[cat_hash])

print ('-- PERSON --')
doc = nlp("David Bowie is a PERSON")

person_hash = nlp.vocab.strings['PERSON']
print ('person hash:', person_hash)
print ('person text:', nlp.vocab.strings[person_hash])

# Doc object
words = ['Hello', 'world', '!']
spaces = [True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print (doc.text)

# Span Object
print ('--- Doc & Span ---')
words = ['spaCy', 'is', 'cool', '!']
spaces = [True, True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print (doc.text)

# 2.6  Doc & Span from Scratch
print ('-- 2.6 Doc and Span from Scratch --')

words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
print (doc.text)

span = Span(doc, 2,4, 'PERSON')
print(span.text, span.label_)

# add span to the doc entities
doc.ents = [span]
print_entities(doc)

# 2.7
# did we set something = None?


doc = nlp("Berlin is a nice city")
print_doc_analysis(doc)

# Get all tokens and part-of-speech tags
print ('-- 2.7 --')

for token in doc:
        print (token.text, token.pos_)
        # check if: proper noun
        if token.pos_ == 'PROPN':
                # is the next token a verb?
                if doc[token.i + 1].pos_ == 'VERB':
                        span = Span(doc, token.i, token.i+2) # + 2 because it is index exclusive1
                        print ('proper noun -> verb:', span.text)


# use large model
nlp = spacy.load("en_core_web_lg")

doc = nlp("Two bananas in pajamas")

bananas_vector = doc[1].vector
print (bananas_vector)