import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

text = """Natural Language Processing is a field of Artificial Intelligence.
It deals with understanding human language and extracting meaningful information."""

# Sentence Tokenization
sentences = sent_tokenize(text)
print("\nSentence Tokenization:", sentences)

# Word Tokenization
words = word_tokenize(text)
print("\nWord Tokenization:", words)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
print("\nStopword Removal:", filtered_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatized Words:", lemmatized)

# POS Tagging
tags = pos_tag(filtered_words)
print("\nPOS Tags:", tags)
