from .DriftDetector import DriftDetector
from alibi_detect.cd import ChiSquareDrift
import numpy as np

# Detector applies feature-wise Chi-Squared tests.
# https://docs.seldon.io/projects/alibi-detect/en/stable/methods/chisquaredrift.html
# https://en.wikipedia.org/wiki/Chi-squared_test
class AlibiChiSquaredDetector(DriftDetector):
    
    # p-value used for significance of the Chi-Squared test.
    # correction Correction type for multivariate data. Either ‘bonferroni’ or ‘fdr’ (False Discovery Rate).
    def __init__(self,
                 p_val = 0.05,
                 correction = "bonferroni"):
        self.p_val = p_val
        self.correction = correction
        super().__init__(classifier=None)
        
    def fit(self, data, targets) -> DriftDetector:
        return self.fit(data)
    
    def fit(self, data) -> DriftDetector:
        self.cd = ChiSquareDrift(data, p_val=self.p_val, correction=self.correction)
        return self
    
    def predict(self, data) -> bool:
        return self.cd.predict(data, drift_type='batch', return_p_val=False, return_distance=False).get('data').get('is_drift')
    
    def predict_proba(self, data) -> float:
        return np.mean(self.cd.predict(data, drift_type='batch', return_p_val=True, return_distance=False).get('data').get('p_val'))

