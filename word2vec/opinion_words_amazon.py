# Get overview of negative words usage in Amazon Reviews
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

import subprocess
import gzip
import gensim
from gensim import corpora
from gensim import models
from gensim.corpora import Dictionary
import os.path
from amazon_reviews_reader import AmazonReviewsReader


data_directory = "../../../DATA/EML4U/negative-words/"
file_neg = data_directory + "negative-words.txt"
file_pos = data_directory + "positive-words.txt"
file_amazonreviews = "../../../DATA/EML4U/amazon-reviews/" + "movies.txt.gz"

# Adds words to list
def read_file(file):
    words = []
    with open(file, 'r', encoding="iso-8859-1") as f:
        for line in f:
            if line and line.strip() and not line.startswith(';'):
                words.append(line.strip())
    return words;


# Token/bow iterator
class WordsReader:
    
    def __init__(self, file, max=-1, mode="tokens", dictionary=""):
        self.file = file
        self.max = max
        self.mode = mode
        self.dictionary = dictionary
        
    def __iter__(self):
        i = 0
        with open(self.file, 'r', encoding="iso-8859-1") as f:
            for line in f:
                if line and line.strip() and not line.startswith(';'):

                    i += 1
                    if(self.max != -1 and i > self.max):
                        break
                    
                    tokens = gensim.utils.tokenize(line.strip(), lower=True)
                    if(self.mode == "tokens"):
                        yield tokens
                    
                    elif(self.mode == "bow"):
                        yield self.dictionary.doc2bow(tokens)

                    else:
                        raise ValueError("Unknown mode")


# Amazon iterator
class AmazonReader:
    
    def __init__(self, amazon_reviews_file, dictionary, max=-1, amazon_reader_mode="tokens"):
        self.dictionary = dictionary
        self.amazon_reader = AmazonReviewsReader(amazon_reviews_file, amazon_reader_mode, max_docs=max)
        
    def __iter__(self):
        for tokens in self.amazon_reader:
            yield self.dictionary.doc2bow(tokens)


# Prints first elements of iterator
def print_iter(it, max):
    i = 0
    while i < max:
        try:
            element = next(it)
            if(isinstance(element, list)):
                print(str(i), "", element)
            else:
                print(str(i), "", list(element))
            i += 1
        except StopIteration:
            break


# Read negative words to list
if(False):
    negative_words = read_file(file_neg)
    print(len(negative_words))

# Create dictionary of negative words
tokens_neg = WordsReader(file_neg, 100)
dictionary_neg = corpora.Dictionary(tokens_neg)

# Print dictionary information
print(dictionary_neg.num_docs, "processed documents")
print(len(dictionary_neg), "words")
print(dictionary_neg.num_pos, "corpus positions / processed words")
if(False):
    print(dictionary_neg)
    print(dictionary_neg.token2id)

# Get overview of negative words usage in Amazon Reviews
# Build corpus (using dictionary)
# Bag of words, words as integer
# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples
bow_neg = AmazonReader(file_amazonreviews, dictionary_neg, 1000)
print_iter(iter(bow_neg), 1000)
