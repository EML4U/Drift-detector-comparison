from fcit import fcit
from .DriftDetector import DriftDetector
import numpy as np
import sklearn

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class CosineSimilarityDriftDetector(DriftDetector):
    def __init__(self,
                 ):
        super().__init__(classifier=None)
        self.boundary = 0.95
        self.original_mean = None
        
    def fit(self, data, targets) -> DriftDetector:
        raise Exception('do not need targets')
    
    def fit(self, data) -> DriftDetector:
        self.original_mean = normalized(sum(data))
        return self
    
    def predict(self, data) -> bool:
        prob = self.predict_proba(data)
        print('probability', prob)
        return prob < self.boundary
    
    def predict_proba(self, data) -> float:
        new_normalized = [normalized(x) for x in data]
        new_mean = normalized(sum(new_normalized))
        cosine = sklearn.metrics.pairwise.cosine_similarity(self.original_mean, new_mean)
        return cosine[0][0]