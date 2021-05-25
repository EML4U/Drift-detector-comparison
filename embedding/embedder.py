

class Embedder:

    def __init__(self, *args, **kwargs):
        self.batch_size = 32 if 'batch_size' not in kwargs else kwargs.pop('batch_size')
        self.verbose = False if 'verbose' not in kwargs else kwargs.pop('verbose')
        self.prepare(**kwargs)


    def prepare(self, **kwargs):
        pass


    def embed(self, text_list):
        pass


    def save(self, path):
        pass


    def load(self, path):
        pass


    def predict(self, text_list):
        pass


    def retrain_one_epoch(self, text_list, labels):
        pass

