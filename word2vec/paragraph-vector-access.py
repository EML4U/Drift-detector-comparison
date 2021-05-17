import gensim
from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load("../../../DATA/EML4U/amazon-reviews/amazonreviews_test.model")

doc = "great blue product";

tokens = gensim.utils.simple_preprocess(doc)
vector = model.infer_vector(tokens)

tokens2 = gensim.utils.simple_preprocess(doc)
vector2 = model.infer_vector(tokens)

print(tokens)
print(tokens2)
print(vector)
print(vector2)

# Subsequent calls to this function may infer different representations for the same document.
# For a more stable representation, increase the number of steps to assert a stricket convergence.
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector