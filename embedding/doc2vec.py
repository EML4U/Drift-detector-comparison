from .embedder import Embedder
import os
import numpy as np
import gensim.models as gensim_models


class Doc2Vec(Embedder):
    def __init__(self, model_path='./data/enwiki_dbow/doc2vec.bin', start_alpha=0.01, infer_epoch=1000):
        self.doc2vec = None
        # inference hyper-parameters
        self.start_alpha = start_alpha
        self.infer_epoch = infer_epoch
        super().__init__(model_path=model_path)

    def prepare(self, **kwargs):
        path = kwargs.pop('model_path')
        if not path or not os.path.exists(path):
            print("cannot load doc2vec model, path does not exist: " + path)
            print("will now acquire doc2vec model...")
            d2v_link = "https://ibm.ent.box.com/s/3f160t4xpuya9an935k84ig465gvymm2"
            os.system('wget ' + d2v_link + ' -p ./data/')
            os.system('tar -xf ./data/enwiki_dbow.tgz -C ./data/')
            path = './data/enwiki_dbow/doc2vec.bin'
        self.doc2vec = gensim_models.Doc2Vec.load(path)

    def embed(self, text_list):
        embeddings = []
        for text in text_list:
            words = text.lower().split()
            emb = (self.doc2vec.infer_vector(words, alpha=self.start_alpha, steps=self.infer_epoch))
            embeddings.append(emb)
        return np.stack(embeddings)

