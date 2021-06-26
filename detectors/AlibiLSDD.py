from .DriftDetector import DriftDetector
from alibi_detect.cd import LSDDDrift
import numpy as np

# Reqires alibi-detect > 0.6.1
# print(alibi_detect.__version__) # 0.6.1
# pip install alibi-detect==0.7.0
# pip3 uninstall tensorflow
# pip3 install -U tensorflow==2.4.0

# Fix for errors in 0.7.0 using 2021-06-08 version
# mv ~/.local/lib/python3.8/site-packages/alibi_detect/cd/pytorch/lsdd.py ~/.local/lib/python3.8/site-packages/alibi_detect/cd/pytorch/lsdd.py_BACKUP
# wget -P ~/.local/lib/python3.8/site-packages/alibi_detect/cd/pytorch/ https://raw.githubusercontent.com/SeldonIO/alibi-detect/a2fc6125eaf8d15068a3069e64d8fbc9cd1b41cb/alibi_detect/cd/pytorch/lsdd.py

# Detector using Least-Squares Density Difference
# https://docs.seldon.io/projects/alibi-detect/en/stable/methods/lsdddrift.html
class AlibiLSDDDetector(DriftDetector):
    
    # backend: tensorflow or pytorch
    def __init__(self,
                 backend='pytorch'):
        self.backend = backend
        super().__init__(classifier=None)
        
    def fit(self, data, targets=None) -> DriftDetector:
        self.cd = LSDDDrift(data, backend=self.backend)
        return self
    
    def predict(self, data) -> bool:
        return self.cd.predict(data).get('data').get('is_drift')
    
    def predict_proba(self, data) -> float:
        return np.mean(self.cd.predict(data).get('data').get('p_val'))
