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


nlp = English()
doc = nlp("This is a sentence.")
print (doc.text)

# 3 Documents, spans, tokens
doc = nlp("I like three (3) tree kangaroos and narwales.")

first_token = doc[0]
print ("first token:", first_token)



#4 Lexical Attributes
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

print_doc_analysis(doc)

for token in doc:
    # check like number
    if token.like_num:
        # the the next token in the doc
        next_token = doc[token.i + 1]
        # check if next is %
        if next_token.text == '%':
            print ("Percentage:", token.text)

# 5 Statistical Models
# $ python -m spacy download en_core_web_sm

# syntactical dependencies
nlp = spacy.load("en_core_web_sm")
doc = nlp("She ate the hot delicious pepperoni pizza ravenously.")
print_doc_syn_dep(doc)

# named entities
print ("--- named entities ---")
# doesn't pick up:  from Facebook
# doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion from Great Hill Partners")

# this one misses iPhone X
doc = nlp("New iPhone X release date leaked as Apple reveals pre-orders by mistake")
print_doc_named_entities(doc)
print(doc[1:3])


# 10 matcher

# initialize vocabulary
matcher = None
pattern = None
matcher = Matcher(nlp.vocab)

# pattern
pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
matcher.add('IPHONE_PATTERN', None, pattern)

# similar to a regex
# call the matcher
matches = matcher(doc)
print ("Matcher output: {} {}".format(type(matches), matches))
print_matcher_results(doc, matcher(doc))

# another example with more complex pattern
pattern = [
    {'IS_DIGIT': True},
    {'LOWER': 'fifa'},
    {'LOWER': 'world'},
    {'LOWER': 'cup'},
    {'IS_PUNCT': True}
]
doc = nlp("Jay Duff loves 2018 FIFA World Cup: France won!")
# new matcher
matcher = Matcher(nlp.vocab)
matcher.add('FIFA_PATTERN', None, pattern)
print_matcher_results(doc, matcher(doc))

# other matcher using lemmatizer
matcher = None
pattern = None
print ("--- 10.c ---")
# - good example of AND, OR conditions
pattern = [
        {'LEMMA': 'love', 'POS': 'VERB'},
        {'POS' : 'NOUN'}
]
# doc = nlp("I love you.")  # no match, YOU not a noun
# doc = nlp("I loved dogs but now I love cats more.")
# doc = nlp("Jay is in love.   I first loved Betty but now I love June.")
doc = nlp("Jay is in love.   I first loved hamburgers but now I love chicken sandwiches.")
# new matcher
matcher = Matcher(nlp.vocab)
matcher.add('LOVE_PATTERN', None, pattern)
print_doc_syn_dep(doc)
print_matcher_results(doc, matcher(doc))

# 10 - part 1
print ('--- 10 - part 1 ---')
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)
# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [
        {"TEXT": 'iOS'}, 
        {"IS_DIGIT": True}
        ]

# Add the pattern to the matcher and apply the matcher to the doc
matcher = Matcher(nlp.vocab)
matcher.add("IOS_VERSION_PATTERN", None, pattern)
print_doc_syn_dep(doc)
print_matcher_results(doc, matcher(doc))

# 10 - part 2
print ('--- 10 - part 2 ---')
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)
# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [
        {"LEMMA": 'download'}, 
        {"POS": 'PROPN'}
        ]

# Add the pattern to the matcher and apply the matcher to the doc
matcher = Matcher(nlp.vocab)
matcher.add("DOWNLOAD_PATTERN", None, pattern)
print_doc_syn_dep(doc)
print_matcher_results(doc, matcher(doc))

# 10 - part 3
print ('--- 10 - part 3 ---')
doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write a pattern for adjective plus one or two nouns
# OP = ?  make this an optional condition
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]
# Add the pattern to the matcher and apply the matcher to the doc
matcher = Matcher(nlp.vocab)
matcher.add("DOWNLOAD_PATTERN", None, pattern)
print_doc_syn_dep(doc)
print_matcher_results(doc, matcher(doc))
