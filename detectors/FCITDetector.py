from fcit import fcit
from .DriftDetector import DriftDetector
import numpy as np

class FCITDriftDetector(DriftDetector):
    
    def __init__(self,
                 boundary = 0.5,
                 num_perm = 8
                 ):
        super().__init__(classifier=None)
        self.boundary = boundary
        self.original_data = None
        self.num_perm = num_perm
        
    def fit(self, data, targets=None) -> DriftDetector:
        self.original_data = np.array(data)
        return self
    
    def predict(self, data) -> bool:
        prob = self.predict_proba(data)
        print('probability:', prob)
        return prob < self.boundary
    
    def predict_proba(self, data) -> float:
        m = min(len(self.original_data), len(data))
        p_value = fcit.test(self.original_data[:m],
                            np.array(data[:m]),
                            num_perm=self.num_perm)
        return p_value
    
    
    
    
    
    