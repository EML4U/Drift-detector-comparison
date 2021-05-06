from fcit import fcit
from .DriftDetector import DriftDetector
import numpy as np

class FCITDriftDetector(DriftDetector):
    def __init__(self,
                 original_data,
                 ):
        super().__init__(original_data=np.array(original_data), 
                         original_labels=None,
                         classifier=None)
        self.boundary = 0.5
        
    def detect_drift(self, new_data) -> bool:
        m = min(len(self.original_data), len(new_data))
        p_value = fcit.test(self.original_data[:m], np.array(new_data[:m]))
        print('p_value:', p_value)
        return p_value < self.boundary