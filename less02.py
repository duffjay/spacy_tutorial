import json

import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

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
print ('-- 2.8 Word vectors & Semantic Similiarities - large model --')
nlp = spacy.load("en_core_web_lg")

# compare 2 doc
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print("Fast Food (doc) Similiarity:", doc1.similarity(doc2))

# compare 2 tokens
doc = nlp("I like chicken thighs and legs")
token1 = doc[3]
token2 = doc[5]
print ("Pizza / Pasta (token) Similarity:", token1.similarity(token2))

# compare doc w/ token
doc = nlp("I like pizza")
token = nlp("pasta")[0]
print ("doc vs token:", doc.similarity(token))

# compare span w/ doc
span = nlp("I like burgers and fries")[2:5]
doc = nlp("McDonalds sells burgers")
print ("doc vs span:", span.similarity(doc))

# --- word vectors ---
nlp = spacy.load('en_core_web_md')

doc = nlp("I have a banana")

# bananas_vector = doc[3].vector
# print ("Banana Vector:", len(bananas_vector))
# print (bananas_vector)

# 2.9 Practice
doc = nlp("Two bananas in pyjamas")

# bananas_vector = doc[1].vector
# print ("Banana Vector:", bananas_vector)

# part 1
doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")
print("Doc Sim:", doc1.similarity(doc2))

# part 2
doc = nlp("magazines and books")
token1, token2 = doc[0], doc[2]
print ("2.9 part 2 - Token Similarity:", token1.similarity(token2))

# part 3
doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")
span1 = doc[3:5]
span2 = doc[12:15]
print ("span1", span1.text)
print ("span2", span2.text)
print ('Similarity:', span1.similarity(span2))

print ('---- 11 Combining Models & Rules ----')

def print_match(matches):
        for match_id, start, end in matches:
                span = doc[start:end]
                print ('matched span:', span.text)
                print ('Root token:', span.root.text)           # category of phrase
                print ('Root Head token:', span.root.head.text) # parent that governs phrase
                print ('Previous token:', doc[start-1].text, doc[start-1].pos_)


matcher = Matcher(nlp.vocab)   # initialize w/ shared vocabulary
pattern = [{'LEMMA': 'love', 'POS': 'VERB'}, {'LOWER': 'cats'}]
matcher.add('LOVE_CATS', None, pattern)

pattern = [{'TEXT': 'very', 'OP': '+'}, {'TEXT': 'happy'}]

# Calling matcher on doc returns list of (match_id, start, end) tuples
doc = nlp("I love cats and I'm very very happy")
matches = matcher(doc)
print_match(matches)

doc = nlp("I have a Golden Retriever")
print (doc.text)
matcher = Matcher(nlp.vocab)
matcher.add('DOG', None, [{'LOWER': 'golden'}, {'LOWER': 'retriever'}])

print_match(matcher(doc))

print ('---- Phrase Matcher ----')
matcher = PhraseMatcher(nlp.vocab)
pattern = nlp("Golden Retriever")
matcher.add('DOG', None, pattern)
doc = nlp("I have a Golden Retriever")

print (doc.text)
print_match(matcher(doc))

print ('--- 13 - Debugging Patterns ---')

doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)

# Create the match patterns
pattern1 = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
pattern2 = [{"LOWER": "ad"}, {"TEXT":'-'}, {'LOWER': 'free'}, {"POS": "NOUN"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", None, pattern1)
matcher.add("PATTERN2", None, pattern2)

print (doc.text)
print ('--- doc analysis ---')
print_doc_analysis(doc)
print ('--- matches ---')
# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Print pattern string name and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)


# 14 Efficient Phrase Matching
with open("exercises/countries.json") as f:
        COUNTRIES = json.loads(f.read())
