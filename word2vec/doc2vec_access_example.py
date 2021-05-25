# Examples

from amazon_reviews_reader import AmazonReviewsReader
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import os.path
import yaml
import gensim


config            = yaml.safe_load(open("../config.yaml", 'r'))
amazon_gz_file    = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "movies.txt.gz")
gensim_model_file = os.path.join(config["GENSIM_MODEL_DIRECTORY"], "amazonreviews_c.model")

max_docs = 2
year = 2002
score = 1
mode = "text"
mode = "tokens"
mode = "tagdoc"
mode = "fields"


# AmazonReviewsReader example
if(False):
    amazon_reviews_reader = AmazonReviewsReader(amazon_gz_file, mode, max_docs=max_docs, min_year=year, max_year=year, min_score=score, max_score=score)
    for item in amazon_reviews_reader:
        print(item)
        print("---")
        
        
# Model example
if(True):
    model = Doc2Vec.load(gensim_model_file)
    amazon_reviews_reader = AmazonReviewsReader(amazon_gz_file, "tokens", max_docs=max_docs, min_year=year, max_year=year, min_score=score, max_score=score)
    tokens = next(iter(amazon_reviews_reader))
    print(model.infer_vector(tokens))


# Variance example
# amazonreviews_c.model: variance 0.12380553807645124, 98 docs
if(False):
    amazon_reader = AmazonReviewsReader(amazon_gz_file, "tokens", max_docs=-1)
    vecsum = 0
    veccount = 0
    i = 0
    for tokens in amazon_reader:
        i += 1
        if(i % 80000 != 0):
            continue
        vectors = []
        vectors.append(model.infer_vector(tokens))
        vectors.append(model.infer_vector(tokens))
        vecsum += np.var(vectors)
        veccount += 1
    print(vecsum/veccount)
    print(veccount)