# Example on how to use the model
# https://hobbitdata.informatik.uni-leipzig.de/EML4U/2021-05-17-Amazon-Doc2Vec/
#
# Please compute vectors only once, cache and reuse them,
# as models may not generate reproducible results.

import gensim
from amazon_reviews_reader import AmazonReviewsReader
from gensim.models.doc2vec import Doc2Vec
import numpy as np

data_directory = "../../../DATA/EML4U/amazon-reviews/"
model_file     = data_directory + "amazonreviews_c.model"
data_file      = data_directory + "movies.txt.gz"

# Load model
model = Doc2Vec.load(model_file)

# Load tokens
if(False):
    year=2002
    score=1
    amazon_reader = AmazonReviewsReader(data_file, "tokens", max_docs=1, min_year=year, max_year=year, min_score=score, max_score=score)
    tokens = next(iter(amazon_reader))

    vectors = []
    vectors.append(model.infer_vector(tokens))
    vectors.append(model.infer_vector(tokens))

    print(vectors)
    print(np.var(vectors))

    # highest variance found so far: 0.17 (2002, score 1)

# Variance
amazon_reader = AmazonReviewsReader(data_file, "tokens", max_docs=-1)
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
#0.12380553807645124
#98