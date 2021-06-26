from .DriftDetector import DriftDetector
from alibi_detect.cd import ChiSquareDrift
import numpy as np

# Detector applies feature-wise Chi-Squared tests.
# https://docs.seldon.io/projects/alibi-detect/en/stable/methods/chisquaredrift.html
# https://en.wikipedia.org/wiki/Chi-squared_test
class AlibiChiSquaredDetector(DriftDetector):
    
    # p-value    used for significance of the Chi-Squared test.
    # correction Correction type for multivariate data. Either ‘bonferroni’ or ‘fdr’ (False Discovery Rate).
    # multiplier e.g. 10, converts float values to create categories. Dev version, did not produce better results.
    def __init__(self,
                 p_val = 0.05,
                 correction = "bonferroni",
                 multiplier = 1):
        self.p_val = p_val
        self.correction = correction
        self.multiplier = multiplier
        super().__init__(classifier=None)
    
    # Dev version of preprocessing conversion, intended to generate categorical features
    def create_categories(self, data):
        if(self.multiplier != 1):
            return np.round(np.multiply(data, self.multiplier))#.astype(int).astype(str)
        else:
            return data
        
    def fit(self, data, targets=None) -> DriftDetector:
        self.cd = ChiSquareDrift(self.create_categories(data), p_val=self.p_val, correction=self.correction)
        return self
    
    def predict(self, data) -> bool:
        return self.cd.predict(self.create_categories(data), drift_type='batch', return_p_val=False, return_distance=False).get('data').get('is_drift')
    
    def predict_proba(self, data) -> float:
        return np.mean(self.cd.predict(self.create_categories(data), drift_type='batch', return_p_val=True, return_distance=False).get('data').get('p_val'))

