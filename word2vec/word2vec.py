# Examples for:
# - Memory-friendly iterating through a corpus file
# - Tokenizing documents (get single words as lists)
# - Converting documents to bag-of-words
# Uses:
# - https://snap.stanford.edu/data/web-Movies.html
# - https://radimrehurek.com/gensim/


import subprocess
import gzip
import gensim
from gensim import corpora
from gensim import models
from gensim.corpora import Dictionary
import os.path


# Data storage configuration
# Directory containing movies.txt.gz
data_directory = "/home/adi/DICE/Data/EML4U/amazon-reviews/"
dataset_url = "https://snap.stanford.edu/data/movies.txt.gz"


# Progessing configuration
amazon_reviews_key = "review/text"
#amazon_reviews_key = "review/summary"
max_docs = 10
max_print = 3
write_dict_cache = False
print_texts = True


# Download file if not available
#
# https://snap.stanford.edu/data/web-Movies.html
# 3321791660 bytes / 3 GB
#
# https://www.gnu.org/software/wget/manual/wget.html#Download-Options
# -c  --continue
# -nv --no-verbose
# -P  --directory-prefix=prefix
subprocess.run(["wget", "-c", "-nv", "-P", data_directory, dataset_url])


# Stream corpus (memory efficient)
# See: https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-one-document-at-a-time
class AmazonCorpusTokens:
    
    def __init__(self, max = -1):
        self.max = max
        
    def __iter__(self):
        i = 0
        with gzip.open(data_directory + 'movies.txt.gz', 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                if amazon_reviews_key in line_spilt[0]:
                    i += 1
                    if(self.max != -1 and i > self.max):
                        break
                    if(print_texts):
                        print(str(i), "", line_spilt[1])
                    yield gensim.utils.tokenize(line_spilt[1], lower=True)


# Stream corpus as bag of words
class AmazonCorpusBow:
    
    def __init__(self, max = -1):
        self.max = max
        
    def __iter__(self):
        i = 0
        with gzip.open(data_directory + 'movies.txt.gz', 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                if amazon_reviews_key in line_spilt[0]:
                    i += 1
                    if(self.max != -1 and i >= self.max):
                        break
                    yield dictionary.doc2bow(line_spilt[1].lower().split())


# Print iterator elements
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


# Build dictionary
# API: https://radimrehurek.com/gensim/corpora/dictionary.html
dictionary_file = data_directory + "movies-dictionary-" + str(max_docs) + ".dict"
dictionary_text_file = data_directory + "movies-dictionary-" + str(max_docs) + ".txt"
if(os.path.isfile(dictionary_file)):
    dictionary = Dictionary.load(dictionary_file)
else:
    amazon_corpus_tokens = AmazonCorpusTokens(max_docs)
    dictionary = corpora.Dictionary(amazon_corpus_tokens)
    if(write_dict_cache):
        dictionary.save(dictionary_file)
        dictionary.save_as_text(dictionary_text_file)
    print_iter(iter(amazon_corpus_tokens), max_print)

print("dictionary.num_docs, number of processed documents", dictionary.num_docs)
print("dictionary len, number of words", len(dictionary))
print("dictionary.num_pos, Total number of corpus positions / processed words", dictionary.num_pos)
if(False):
    print("dictionary", dictionary)
    print("dictionary.token2id", dictionary.token2id)


# Build corpus (using dictionary)
# Bag of words, words as integer
# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples
amazon_corpus_bow = AmazonCorpusBow(max_docs)
print_iter(iter(amazon_corpus_bow), max_print)


# Further works:
# - Use fixed, low dimensions
#   e.g. https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# - Also check stopwords and <br />
# - Include "review/summary", as it contains useful positive/negative adjectives
# - Check https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92