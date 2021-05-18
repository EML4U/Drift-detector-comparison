# Paragraph Vector (Doc2Vec)
#
# Le and Mikolov: Distributed Representations of Sentences and Documents
# https://cs.stanford.edu/~quocle/paragraph_vector.pdf
#
# Installation note:
# Make sure you have a C compiler before installing Gensim, to use the optimized doc2vec routines.
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
#
# Uses:
# - https://snap.stanford.edu/data/web-Movies.html
# - https://radimrehurek.com/gensim/
# - https://radimrehurek.com/gensim/models/doc2vec.html


import subprocess
import gzip
import os.path
from datetime import datetime
import time
import gensim


# Data storage configuration
# Directory containing movies.txt.gz
data_directory = "../../../DATA/EML4U/amazon-reviews/"
model_file = data_directory + "amazonreviews.model"
dataset_url = "https://snap.stanford.edu/data/movies.txt.gz"


# Progessing configuration
max_year    = 2000   # Max year for training
max_docs    = -1     # -1 to process all (for development)
print_texts = False  # Prints iterated texts (for development)

doc2vec_vector_size = 50  # Dimensionality of the feature vectors
doc2vec_min_count   = 2   # Ignores all words with total frequency lower than this
doc2vec_epochs      = 40  # Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec
doc2vec_seed        = -1  # -1, or int for reproducible results (dev)


# Do not overwrite
if os.path.isfile(model_file):
    print("Model file already exists, exiting.", model_file)
    exit()


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
class Corpus:
    
    def __init__(self, max_docs=-1, max_year=2000, tokens_only=False):
        self.max_docs = max_docs
        self.max_year = max_year
        self.tokens_only = tokens_only

    def __iter__(self):
        i = 0
        with gzip.open(data_directory + 'movies.txt.gz', 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                if "review/time" in line_spilt[0]:
                    if(datetime.fromtimestamp(int(line_spilt[1])).year >= self.max_year):
                        self.continue_loop = True
                    else:
                        self.continue_loop = False
                if "review/summary" in line_spilt[0]:
                    text = line_spilt[1]
                if "review/text" in line_spilt[0]:
                    if(self.continue_loop):
                        continue
                    
                    text += line_spilt[1]
                    
                    i += 1
                    if(self.max_docs != -1 and i > self.max_docs):
                        break
                    
                    if(print_texts):
                        print(str(i), "", text)
                        
                    tokens = gensim.utils.simple_preprocess(text)
                    text = ""
                    
                    if self.tokens_only:
                        yield tokens
                    else:
                        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


time_begin = time.time()
print(datetime.fromtimestamp(time_begin))
corpus = Corpus(max_docs, max_year)

# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
if(doc2vec_seed == -1):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_vector_size, min_count=doc2vec_min_count, epochs=doc2vec_epochs, dm=1)
else:
    # Note: Even setting PYTHONHASHSEED to 0 did not result in reproducible values
    print("PYTHONHASHSEED (should be 0)", os.environ.get("PYTHONHASHSEED"))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_vector_size, min_count=doc2vec_min_count, epochs=doc2vec_epochs, dm=1, seed=doc2vec_seed, workers=1)

# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.build_vocab
print("Building vocabulary")
model.build_vocab(corpus)

# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.train
print("Training model")
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

# Save model
model.save(model_file)
print("Saved model file", model_file)

# Print info
print(model)
time_end = time.time()
print("Runtime: %s seconds" % (time_end-time_begin))

if(max_docs != -1):
    print(max_docs, doc2vec_epochs, (time_end-time_begin)/max_docs, (time_end-time_begin)/max_docs*7911684/60/60)
# docs  it  secs/doc  hours/corpus
# 1000  10  0.0032     7.2236
# 1000  40  0.0121    26.7820
# 10000 10  0.0031     6.9914
# 10000 40  0.0123    27.1997