from .embedder import Embedder
import numpy as np

from sentence_transformers import SentenceTransformer
import torch
import math


class BertUKPLab(Embedder):

    def __init__(self, module_url=None, batch_size=32):
        self.embedder = None
        super().__init__(module_url=module_url, batch_size=batch_size)

    def prepare(self, **kwargs):
        module_url = kwargs['module_url'] or 'bert-base-nli-mean-tokens'

        print("loading model")
        if torch.cuda.is_available():
            self.embedder = SentenceTransformer(module_url, device='cuda')
            print('using bert with cuda')
        else:
            self.embedder = SentenceTransformer(module_url)


    def embed(self, text_list):
        if len(text_list) < self.batch_size:
            return np.stack(self.embedder.encode(text_list))

        emb_list = []
        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        print('Splitting into', num_steps, 'batches...')
        for i in range(num_steps):
            #print(i)
            ul = min((i + 1) * self.batch_size, len(text_list))
            subset = text_list[i * self.batch_size:ul]
            if self.verbose:
                if i%100 == 0:
                    print("at step", i, "of", num_steps)
            try:
                me = self.embedder.encode(subset)
            except:
                print("cannot encode batch nr. "+str(i))
                print("try to encode single texts")
                for text in subset:
                    try:
                        self.embedder.encode([text])
                    except:
                        print("failed to encode: ")
                        print(text)
            emb_list.append(me)
        embeddings = np.vstack(emb_list)
        return embeddings

