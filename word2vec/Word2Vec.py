# Minimal example:
#  word2vec = Word2Vec("amazonreviews.model")
#  word2vec.prepare()
#  print(word2vec.embed(["Hello world", "Hey world"]))

from gensim.models.doc2vec import Doc2Vec
import gensim
import numpy as np

class Word2Vec():

    def __init__(self, model_path):
        self.model_path = model_path

    def prepare(self):
        self.model = Doc2Vec.load(self.model_path)

    def embed(self, text_list):
        embeddings = []
        for text in text_list:
            emb = self.model.infer_vector(gensim.utils.simple_preprocess(text))
            embeddings.append(emb)
        return np.stack(embeddings)
