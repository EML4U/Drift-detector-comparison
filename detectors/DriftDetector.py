from sklearn.base import BaseEstimator

class DriftDetector(BaseEstimator):

    def __init__(self,
                 classifier):
        self.classifier = classifier
        
    def fit(self, data, targets):
        return self
    
    def fit(self, data):
        return self
    
    def predict(self, data) -> bool:
        pass
    
    def predict_proba(self, data) -> float:
        pass
    