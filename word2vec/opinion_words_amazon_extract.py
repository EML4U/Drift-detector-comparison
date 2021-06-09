# Extract overview of negative words usage in Amazon Reviews
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

import yaml
import os.path
import gensim
from gensim import corpora
from amazon_reviews_reader import AmazonReviewsReader
import pickle


config              = yaml.safe_load(open("../config.yaml", 'r'))
negative_words_file = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words.txt")
amazon_gz_file      = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "movies.txt.gz")
pickle_score_one    = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-one.pickle")
pickle_score_five   = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-five.pickle")


# Deprecated: Adds words to list
def read_file(file):
    words = []
    with open(file, 'r', encoding="iso-8859-1") as f:
        for line in f:
            if line and line.strip() and not line.startswith(';'):
                words.append(line.strip())
    return words;


# Token/bow iterator for opinion word files
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
                    
                    if(True):
                        tokens = [line.strip().lower()]
                    else:
                        # Splits words like 'anti-social'
                        tokens = gensim.utils.tokenize(line.strip(), lower=True)
                        
                    if(self.mode == "tokens"):
                        yield tokens
                    
                    elif(self.mode == "bow"):
                        yield self.dictionary.doc2bow(tokens)

                    else:
                        raise ValueError("Unknown mode")


# Amazon iterator
class AmazonReader:
    
    def __init__(self, dictionary, amazon_reviews_file, amazon_reader_mode="tokens", max_docs=-1,
                 min_year=-1, max_year=-1, min_score=-1, max_score=-1):
        self.dictionary = dictionary
        self.amazon_reader = AmazonReviewsReader(amazon_reviews_file, amazon_reader_mode, max_docs=max_docs,
                                                 min_year=min_year, max_year=max_year, min_score=min_score, max_score=max_score)
        
    def __iter__(self):
        for tokens in self.amazon_reader:
            yield self.dictionary.doc2bow(tokens)


# Prints first elements of iterator
def print_iter(it, max_element_to_print):
    i = 0
    while i < max_element_to_print:
        try:
            element = next(it)
            if(isinstance(element, list)):
                print(str(i), "", element)
            else:
                print(str(i), "", list(element))
            i += 1
        except StopIteration:
            break


# Deprecated: Read 4783 negative words to list
if(False):
    negative_words = read_file(negative_words_file)
    print(len(negative_words))
    exit()


# Create dictionary of negative words
max_negative_words  = -1  # -1 for all
tokens_neg = WordsReader(negative_words_file, max_negative_words)
dictionary_neg = corpora.Dictionary(tokens_neg)

# Print dictionary information
print(dictionary_neg.num_docs, "processed documents")
print(len(dictionary_neg), "words")
print(dictionary_neg.num_pos, "corpus positions / processed words")
if(False):
    print(dictionary_neg)
    print(dictionary_neg.token2id)

# Create iterators to filter and read Amazon dataset
# Filter negative words using dictionary
# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples
max_docs            = -1  # -1 for all ; e.g. 10 to test
max_year            = -1  # -1 for all ; 1999 is small corpus

if(False):
    # Create small test dataset
    pickle_score_one      = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-one.pickle")
    pickle_score_five     = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-five.pickle")
    max_docs              = 100
    max_year              = 1999

score_one  = AmazonReader(dictionary_neg, amazon_gz_file, max_score=1, max_docs=max_docs, max_year=max_year)
score_five = AmazonReader(dictionary_neg, amazon_gz_file, min_score=5, max_docs=max_docs, max_year=max_year)

# Iterate and print
if(False):
    print_iter(iter(score_one), max_element_to_print=max_docs)
if(False):
    print_iter(iter(score_five), max_element_to_print=max_docs)

# Iterate to count
if(False):
    print(sum(1 for _ in iter(score_one)))
if(False):
    print(sum(1 for _ in iter(score_five)))

# E.g. the word 'bad' is in 1-star and 5-star docs
# print(dictionary_neg[245])

# Iterate through results and create directory[neg_word_id] = number_of_occurences_in_doc
# Sort by keys and create dictionary [neg_word] = number_of_occurences_in_doc
counts = dict()
for doc in iter(score_one):
    for tup in iter(doc):
        counts[tup[0]] = counts.get(tup[0], 0) + 1
negative_score_one = dict()
for k in sorted(counts.keys()):
    negative_score_one[dictionary_neg[k]] = counts[k]

counts = dict()
for doc in iter(score_five):
    for tup in iter(doc):
        counts[tup[0]] = counts.get(tup[0], 0) + 1
negative_score_five = dict()
for k in sorted(counts.keys()):
    negative_score_five[dictionary_neg[k]] = counts[k]
    
# Save
if(True):
    with open(pickle_score_one, 'wb') as handle:
        pickle.dump(negative_score_one, handle)
    with open(pickle_score_five, 'wb') as handle:
        pickle.dump(negative_score_five, handle)