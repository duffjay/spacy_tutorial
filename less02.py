import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher


spacy.prefer_gpu()
nlp = English()

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