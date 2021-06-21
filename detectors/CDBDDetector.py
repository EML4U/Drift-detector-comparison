# from .DriftDetector import DriftDetector
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import numpy as np
from detectors.unc_detector.uncertaintyM import uncertainty_ent
from detectors.unc_detector.a_RF import get_prob_matrix
import matplotlib.pyplot as plt


from math import log2
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

class CDBD(BaseEstimator):
    
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, data, targets):
        return self.fit(data)

    def get_distribution(self, data, bins):
        data_digit = np.digitize(data, bins)
        c = np.bincount(data_digit, minlength=len(bins))
        return c / c.sum()
    
    def fit(self, data, batch_size=0, n_batch=10):
        if batch_size==0:
            self.batch_size = int(len(data)/ n_batch + 1)
        else:
            pass

        
        data = np.array(data)
        # data_score = self.classifier.predict_proba(data) # calculate the output prob dist for all test data
        # get the indicator scores by computing total uncertainty
        porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 1) # 1 is laplace
        data_score, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_ent(porb_matrix)

        # get the bins
        # n, bins, patches = plt.hist(data_score, bins=9) # equal distance bins
        # self.bins = bins[1:-1]

        n, bins, patches = plt.hist(data_score, bins=np.logspace(data_score.min(), data_score.max(), 9, base=2)) # log bins
        self.bins = bins
        plt.savefig("unc_dist_test_data.png") # plot to see distribution of the total uncertainty(used as indicator score)
        plt.close()

        # calculating prob distribution for the reference batch
        ref_batch = data_score[0:self.batch_size]
        self.ref_dist = self.get_distribution(ref_batch, self.bins)
        
        # old before get_distribution function
        # ref_digit = np.digitize(ref_batch, bins[1:-1])
        # c = np.bincount(ref_digit, minlength=9)
        # self.ref_dist = c / c.sum()


        # calculate the threshold
        kl_list = []
        for i in range(1,n_batch):
            batch_score = data_score[i*self.batch_size:(i+1)*self.batch_size] # take a batch
            batch_dist = self.get_distribution(batch_score, self.bins)
            kl = rel_entr(batch_dist, self.ref_dist).sum() # calculate KL divergence. sum over values for each class
            kl_list.append(kl)
        kl_array = np.array(kl_list)
        kl_array = kl_array[kl_array < 1E308] # removing inf values of KL
        self.threshold = kl_array.mean() + kl_array.std()

        return self

    def predict(self, data) -> bool:
        kl = self.predict_proba(data)
        if kl > self.threshold:
            return True
        else:
            return False
    
    def predict_proba(self, data) -> float:
        data = np.array(data)
        porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 1) # 1 is laplace
        data_score, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_ent(porb_matrix)

        # [Method 1] KL divergence for the entire test data
        data_dist = self.get_distribution(data_score, self.bins)
        kl_all = rel_entr(data_dist, self.ref_dist).sum()

        # [Method 2] averaged KL for test data seperated into batches that are the same size as the ref_batch
        n_batch = int(data.shape[0] / self.batch_size)
        kl_list = []
        for i in range(n_batch):
            batch_score = data_score[i*self.batch_size:(i+1)*self.batch_size]
            batch_dist = self.get_distribution(batch_score, self.bins)
            kl = rel_entr(batch_dist, self.ref_dist).sum() # sum over values for each class
            kl_list.append(kl)
        kl_array = np.array(kl_list)
        kl_array = kl_array[kl_array < 1E308]
        kl_avg = kl_array.mean()

        return kl_avg
