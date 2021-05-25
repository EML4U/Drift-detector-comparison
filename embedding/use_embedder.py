from .embedder import Embedder
import numpy as np

import math
import tensorflow_hub as hub


class USEEmbedder(Embedder):

    def __init__(self, module_url=None, batch_size=32):
        self.embedder = None
        super().__init__(module_url=module_url, batch_size=batch_size)

    def prepare(self, **kwargs):
        module_url = kwargs['module_url'] or "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        print("load model: " + module_url)
        self.embedder = hub.load(module_url)


    def embed(self, text_list):
        if len(text_list) < self.batch_size:
            return self.embedder(text_list).numpy()

        emb_list = []
        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        print('Splitting into', num_steps, 'batches...')
        for i in range(num_steps):
            ul = min((i + 1) * self.batch_size, len(text_list))
            subset = text_list[i * self.batch_size:ul]
            me = self.embedder(subset)
            emb_list.append(me)
            if self.verbose:
                if i%100 == 0:
                    print("at step", i, "of", num_steps)
        embeddings = np.vstack(emb_list)
        return embeddings

