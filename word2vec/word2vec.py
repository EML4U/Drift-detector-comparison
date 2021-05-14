import subprocess
import gzip
import gensim
from gensim import corpora
from gensim import models


# Configuration
# Directory containing movies.txt.gz
data_directory = "/home/adi/DICE/Data/EML4U/amazon-reviews/"
dataset_url = "https://snap.stanford.edu/data/movies.txt.gz"


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


# TODO remove <br /> and stopwords
class AmazonCorpusTokens:
    
    def __init__(self, max = -1):
        self.max = max
        
    def __iter__(self):
        i = 0
        with gzip.open(data_directory + 'movies.txt.gz', 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                identifier = line_spilt[0]
                if 'review/text' in identifier:
                    i += 1
                    if(self.max != -1 and i >= self.max):
                        break
                    yield gensim.utils.tokenize(line_spilt[1], lower=True)


class AmazonCorpusBow:
    
    def __init__(self, max = -1):
        self.max = max
        
    def __iter__(self):
        i = 0
        with gzip.open(data_directory + 'movies.txt.gz', 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                identifier = line_spilt[0]
                if 'review/text' in identifier:
                    i += 1
                    if(self.max != -1 and i >= self.max):
                        break
                    yield dictionary.doc2bow(line_spilt[1].lower().split())


max_docs = 8

# Build dictionary
amazon_corpus_tokens = AmazonCorpusTokens(max_docs)
dictionary = corpora.Dictionary(amazon_corpus_tokens)
print("dictionary", len(dictionary))

# Bag of words, words as integer
amazon_corpus_bow = AmazonCorpusBow(max_docs)

# Dev: Build model with reduced number of dimensions
model = models.RpModel(amazon_corpus_bow, num_topics=50)
print("model", model)


# Check https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
    