from DAWIDD.kernel_two_sample_test import kernel_two_sample_test
from .DriftDetector import DriftDetector
import numpy as np

# Requires kernel_two_sample_test.py
#
# Repository has no setup file
# https://github.com/FabianHinder/DAWIDD
#
# Just info: Origin is https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py
#
# Manual install:
# mkdir DAWIDD ; cd DAWIDD
# wget https://raw.githubusercontent.com/FabianHinder/DAWIDD/master/kernel_two_sample_test.py

class KernelTwoSampleDriftDetector(DriftDetector):
    
    def __init__(self,
                 boundary = 0.05,
                 iterations = 500,
                 ):
        super().__init__(classifier=None)
        self.boundary = boundary
        self.iterations = iterations
        
    def fit(self, data, targets=None) -> DriftDetector:
        self.original_data = np.array(data)
        return self
    
    def predict(self, data) -> bool:
        p_value = self.predict_proba(data)        
        print('p_value:', p_value)
        return p_value < self.boundary
    
    def predict_proba(self, data) -> float:
        p_value = kernel_two_sample_test(self.original_data,
                                         np.array(data),
                                         iterations=self.iterations)[2]
        return p_value
