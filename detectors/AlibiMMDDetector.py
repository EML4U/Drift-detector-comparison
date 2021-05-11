from .DriftDetector import DriftDetector
from alibi_detect.cd import MMDDrift
import numpy as np

# The Maximum Mean Discrepancy (MMD) detector uses kernel-based method for multivariate 2 sample testing.
# https://docs.seldon.io/projects/alibi-detect/en/stable/methods/mmddrift.html
# https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
class AlibiMMDDetector(DriftDetector):
    
    # p_val: p-value used for significance of the permutation test.
    # backend: tensorflow or pytorch
    def __init__(self,
                 p_val = 0.05,
                 backend = 'tensorflow'):
        self.p_val = p_val
        self.backend = backend
        super().__init__(classifier=None)
        
    def fit(self, data, targets) -> DriftDetector:
        return self.fit(data)
    
    def fit(self, data) -> DriftDetector:
        self.cd = MMDDrift(data, p_val=self.p_val, backend=self.backend)
        return self
    
    def predict(self, data) -> bool:
        return self.cd.predict(data, return_p_val=False, return_distance=False).get('data').get('is_drift')
    
    def predict_proba(self, data) -> float:
        return np.mean(self.cd.predict(data, return_p_val=True, return_distance=False).get('data').get('p_val'))
