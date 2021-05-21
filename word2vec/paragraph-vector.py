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
from amazon_reviews_reader import AmazonReviewsReader


# Data storage configuration
# Directory containing movies.txt.gz
dataset_url        = "https://snap.stanford.edu/data/movies.txt.gz"
data_directory     = "../../../DATA/EML4U/amazon-reviews/"
file_amazonreviews = data_directory + "movies.txt.gz"
model_file         = data_directory + "amazonreviews.model"


# Progessing configuration
max_year    = 2000   # Max year for training
max_docs    = -1     # -1 to process all (for development)
print_texts = False  # Prints iterated texts (for development)

doc2vec_vector_size = 50  # Dimensionality of the feature vectors
doc2vec_min_count   = 2   # Ignores all words with total frequency lower than this
doc2vec_epochs      = 10  # Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec
doc2vec_dm          = 1   # Training algorithm, distributed memory (PV-DM) or distributed bag of words (PV-DBOW)
doc2vec_seed        = -1  # -1, or int for reproducible results (under development)


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


time_begin = time.time()
print(datetime.fromtimestamp(time_begin))
corpus = AmazonReviewsReader(file_amazonreviews, "tagdoc", max_docs=max_docs, max_year=max_year)

# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
if(doc2vec_seed == -1):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_vector_size, min_count=doc2vec_min_count, epochs=doc2vec_epochs, dm=doc2vec_dm)
else:
    # https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ Q11
    print("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_vector_size, min_count=doc2vec_min_count, epochs=doc2vec_epochs, dm=doc2vec_dm, seed=doc2vec_seed, workers=1)

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
runtime = time_end - time_begin;
print("Runtime: %s seconds" % (runtime))

if(max_docs != -1):
    print(max_docs, doc2vec_epochs, runtime/max_docs, runtime/max_docs*7911684/60/60, runtime/max_docs*418065/60/60)
# docs  it  secs/doc  hours/corpus  hours/max2000
# 1000  10  0.0078    17.1936       0.9085