# from .DriftDetector import DriftDetector
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
from scipy.special import kl_div
import numpy as np
from detectors.unc_detector.uncertaintyM import uncertainty_ent
from detectors.unc_detector.a_RF import get_prob_matrix
import matplotlib.pyplot as plt

from math import log2
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def conficance_score_svm(model, data):
    y = model.decision_function(data)
    w_norm = np.linalg.norm(model.coef_)
    dist = y / w_norm
    dist = dist.mean(axis=1) # average over distances from all the classes [not sure]
    return dist

class CDBD(BaseEstimator):
    
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, data, targets):
        return self.fit(data)

    def get_distribution(self, data, laplace=1):
        data_digit = np.digitize(data, self.bins)
        c = np.bincount(data_digit, minlength=len(self.bins)+1) # +1 is to fix the problem with not counting last bin for sum batches
        c = c + laplace
        return c / c.sum()
    
    def divergence_to_pvalue(self, divergence):
        larger_kl = np.where( self.n_batch_kl < divergence)
        p_value = len(larger_kl[0]) / len(self.n_batch_kl) 
        return p_value
    
    def fit(self, data, batch_size=0, n_batch=50):
        if batch_size==0:
            self.batch_size = int(len(data)/ n_batch + 1)
        else:
            pass

        
        data = np.array(data)
        # data_score = self.classifier.predict_proba(data) # calculate the output prob dist for all test data
        # get the indicator scores by computing total uncertainty

        # data score for Forest
        # porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 1) # 1 is laplace
        # data_score, _, _ = uncertainty_ent(porb_matrix)

        # data score for svm
        data_score = conficance_score_svm(self.classifier, data)

        # get the bins
        n, bins, patches = plt.hist(data_score, bins=9) # equal distance bins
        self.bins = bins[1:-1]
        
        # nn = self.get_distribution(data_score) # just for test

        # n, bins, patches = plt.hist(data_score, bins=np.logspace(data_score.min(), data_score.max(), 9, base=2)) # log bins
        # self.bins = bins

        plt.savefig("unc_dist_test_data.png") # plot to see distribution of the total uncertainty(used as indicator score)
        plt.close()
        # print(n)
        # print(nn)
        # exit()

        # calculating prob distribution for the reference batch
        ref_batch = data_score[0:self.batch_size]
        self.ref_dist = self.get_distribution(ref_batch)
        
        # old before get_distribution function
        # ref_digit = np.digitize(ref_batch, bins[1:-1])
        # c = np.bincount(ref_digit, minlength=9)
        # self.ref_dist = c / c.sum()


        # calculate the threshold
        kl_list = []
        for i in range(1,n_batch):
            if (i)*self.batch_size > len(data_score):
                break
            batch_score = data_score[i*self.batch_size:(i+1)*self.batch_size] # take a batch
            # print("batch_score ", batch_score.shape)
            batch_dist = self.get_distribution(batch_score)
            # kl = rel_entr(self.ref_dist, batch_dist).sum() # calculate KL divergence. sum over values for each class
            kl = kl_div(self.ref_dist, batch_dist).sum()
            kl_list.append(kl)
        kl_array = np.array(kl_list)
        # kl_array = kl_array[kl_array < 1E308] # removing inf values of KL
        self.n_batch_kl = kl_array
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

        # score for forest
        # porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 1) # 1 is laplace
        # data_score, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_ent(porb_matrix)

        # data score for svm
        data_score = conficance_score_svm(self.classifier, data)

        # [Method 1] KL divergence for the entire test data
        data_dist = self.get_distribution(data_score)
        # kl_all = rel_entr(self.ref_dist, data_dist).sum()
        kl_all = kl_div(self.ref_dist, data_dist).sum()

        # # [Method 2] averaged KL for test data seperated into batches that are the same size as the ref_batch
        # n_batch = int(data.shape[0] / self.batch_size)
        # kl_list = []
        # for i in range(n_batch):
        #     batch_score = data_score[i*self.batch_size:(i+1)*self.batch_size]
        #     batch_dist = self.get_distribution(batch_score)
        #     kl = rel_entr(self.ref_dist, batch_dist).sum() # sum over values for each class
        #     kl_list.append(kl)
        # kl_array = np.array(kl_list)
        # kl_array = kl_array[kl_array < 1E308]
        # kl_avg = kl_array.mean()

        p_value = self.divergence_to_pvalue(kl_all)
        return p_value
