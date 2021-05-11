from .DriftDetector import DriftDetector
from alibi_detect.cd import KSDrift
import numpy as np

# Detector applies feature-wise two-sample Kolmogorov-Smirnov (K-S) tests
# https://docs.seldon.io/projects/alibi-detect/en/stable/methods/ksdrift.html
# https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
class AlibiKSDetector(DriftDetector):
    
    # p_val: p-value used for significance of the K-S test. 
    #  If the FDR correction method is used, this corresponds to the acceptable q-value.
    # correction: Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
    def __init__(self,
                 p_val = 0.05,
                 correction = 'bonferroni'):
        self.p_val = p_val
        self.correction = correction
        super().__init__(classifier=None)
        
    def fit(self, data, targets) -> DriftDetector:
        return self.fit(data)
    
    def fit(self, data) -> DriftDetector:
        self.cd = KSDrift(data, p_val=self.p_val, correction=self.correction)
        return self
    
    def predict(self, data) -> bool:
        return self.cd.predict(data, drift_type='batch', return_p_val=False, return_distance=False).get('data').get('is_drift')
    
    def predict_proba(self, data) -> float:
        return np.mean(self.cd.predict(data, drift_type='batch', return_p_val=True, return_distance=False).get('data').get('p_val'))
