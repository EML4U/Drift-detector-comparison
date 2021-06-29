from .DriftDetector import DriftDetector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import scipy.stats

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class AdvancedCosineSimilarityDriftDetector(DriftDetector):
    
    def __init__(self,):
        super().__init__(classifier=None)
        self.original_mean = None
        self.norm = scipy.stats.norm()
        self.cos_mean = None
        self.cos_stdev = None
        
        
    def fit(self, data, targets=None) -> DriftDetector:
        new_normalized = np.array([normalized(x) for x in data])
        self.original_mean = normalized(sum(new_normalized))
        
        means = []
        kf = KFold(n_splits=40)
        for train_index, test_index in kf.split(new_normalized):
            train_ = normalized(sum(new_normalized[train_index]))
            test_ = normalized(sum(new_normalized[test_index]))
            means.append(cosine_similarity(train_, test_)[0][0])
        self.cos_mean=np.mean(means)
        self.cos_stdev=np.std(means)
        return self
    
    def predict(self, data) -> bool:
        prob = self.predict_proba(data)
        print('probability', prob)
        return prob < 0.05
    
    def predict_proba(self, data) -> float:
        new_normalized = [normalized(x) for x in data]
        new_mean = normalized(sum(new_normalized))
        cosine = cosine_similarity(self.original_mean, new_mean)[0][0]    
        y = (cosine - self.cos_mean) / self.cos_stdev
        return self.norm.pdf(y) / self.cos_stdev