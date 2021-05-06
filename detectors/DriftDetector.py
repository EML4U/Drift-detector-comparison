class DriftDetector():

    def __init__(self,
                 original_data,
                 original_labels, 
                 classifier):
        self.original_data = original_data
        self.original_labels = original_labels
        self.classifier = classifier
    
    def detect_drift(self, new_data) -> bool:
        raise Exception('Not implemented!')