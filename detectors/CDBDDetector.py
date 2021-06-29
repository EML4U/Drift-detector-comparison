# from .DriftDetector import DriftDetector
from .DriftDetector import DriftDetector
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
from scipy.special import kl_div
import numpy as np
from detectors.unc_detector.uncertaintyM import uncertainty_ent
from detectors.unc_detector.a_RF import get_prob_matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
from scipy.stats import t

def conficance_score_svm(model, data):
    y = model.decision_function(data)
    w_norm = np.linalg.norm(model.coef_)
    dist = y / w_norm
    try:
        dist = dist.mean(axis=1) # Max of distances from all the classes [not sure]
    except:
        pass
    return dist

class CDBDDetector(DriftDetector):
    
    def __init__(self, model_type="forest"):
        super().__init__(classifier=None)

        self.model_type = model_type
        if model_type == "forest":
            self.classifier = RandomForestClassifier(bootstrap=True,
                criterion='entropy',
                max_depth=10,
                n_estimators=30,
                random_state=42,
                verbose=0,
                warm_start=False)
        else:
            self.classifier = SVC(kernel='linear', random_state=42) # SVM model



    def get_distribution(self, data, laplace=1, log=False):
        data_digit = np.digitize(data, self.bins)
        c = np.bincount(data_digit, minlength=len(self.bins)+1) # +1 is to fix the problem with not counting last bin for sum batches
        if log:
            print("------------------------------------get_distribution log")
            print("c\n", c)
        c = c + laplace
        if log:
            print("c lapcace\n", c)
            print("dist\n", c / c.sum(), " dist sum ", (c / c.sum()).sum())
            print()
        return c / c.sum()
    
    def divergence_to_pvalue(self, divergence, log=False):
        # # emperical p-value
        # larger_kl = np.where( self.n_batch_kl < divergence)
        # p_value = len(larger_kl[0]) / len(self.n_batch_kl)

        # define the normal dist with mean ans sd from kl_array
        p_value = norm(loc=self.n_batch_kl_mean,scale=self.n_batch_kl_std).sf(abs(divergence))

        # t_value = (divergence - self.n_batch_kl_mean) / (self.n_batch_kl_std / math.sqrt(len(self.n_batch_kl)-1))
        # p_value = t.sf(t_value)


        if log:
            print("------------------------------------divergence_to_pvalue log")
            print("divergence\n", divergence)
            print("kl batch array\n", self.n_batch_kl)
            # print(f"len larger {len(larger_kl[0])} len kl batch {len(self.n_batch_kl)}")
            print("p_value ", p_value) 
        return p_value

    def fit(self, data, targets, n_batch=20, train_split=0.5, batch_size=0):
        # train a model
        data = np.array(data)
        targets = targets.astype('int')
        x_train, data, y_train, y_test = train_test_split(data, targets, test_size=1-train_split, shuffle=True)#, shuffle=False)
        self.classifier.fit(x_train, y_train)

        if batch_size==0:
            self.batch_size = int(len(data)/ (n_batch + 1))
        else:
            pass
                
        # data_score = self.classifier.predict_proba(data) # calculate the output prob dist for all test data
        # get the indicator scores by computing total uncertainty
        
        if self.model_type == "forest":
            # data score for Forest
            porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 0) # 1 is laplace
            data_score, _, _ = uncertainty_ent(porb_matrix)
        else:
            # data score for svm
            data_score = conficance_score_svm(self.classifier, data)

        # get the bins
        n, bins, patches = plt.hist(data_score, bins=9) # equal distance bins
        self.bins = bins[1:-1]
        
        # nn = self.get_distribution(data_score) # just for test

        # n, bins, patches = plt.hist(data_score, bins=np.logspace(data_score.min(), data_score.max(), 9, base=2)) # log bins
        # self.bins = bins

        # plt.savefig("unc_dist_test_data.png") # plot to see distribution of the total uncertainty(used as indicator score)
        plt.close()
        # print(n)
        # print(nn)
        # exit()

        # calculating prob distribution for the reference batch
        ref_batch = data_score[0:self.batch_size]
        # print("reference batch ", ref_batch.shape)
        self.ref_dist = self.get_distribution(ref_batch, log=False)

        # calculate the threshold
        kl_list = []
        for i in range(1,n_batch+1):
            # if (i+1)*self.batch_size > len(data_score):
            #     break
            batch_score = data_score[i*self.batch_size:(i+1)*self.batch_size] # take a batch
            # print("batch_score ", batch_score.shape)
            batch_dist = self.get_distribution(batch_score, log=False)
            kl = rel_entr(self.ref_dist, batch_dist).sum() # calculate KL divergence. sum over values for each class
            # kl = kl_div(self.ref_dist, batch_dist).sum()
            # kl = kl_div(batch_dist, self.ref_dist).sum()
            kl_list.append(kl)
        kl_array = np.array(kl_list)
        # kl_array = kl_array[kl_array < 1E308] # removing inf values of KL
        self.n_batch_kl = kl_array
        self.n_batch_kl_mean = kl_array.mean()
        self.n_batch_kl_std  = kl_array.std(ddof=1)
        self.threshold = kl_array.mean() + kl_array.std()
        return self

    # def fit(self, data):
    #     pass

    def predict(self, data) -> bool:
        kl = self.predict_proba(data)
        if kl > self.threshold:
            return True
        else:
            return False
    
    def predict_proba(self, data) -> float:
        data = np.array(data)
        
        if self.model_type == "forest":
            # data score for Forest
            porb_matrix = get_prob_matrix(self.classifier, data, self.classifier.n_estimators, 0) # 1 is laplace
            data_score, _, _ = uncertainty_ent(porb_matrix)
        else:
            # data score for svm
            data_score = conficance_score_svm(self.classifier, data)

        data_dist = self.get_distribution(data_score, log=False)
        kl_all = rel_entr(self.ref_dist, data_dist).sum()
        p_value = self.divergence_to_pvalue(kl_all, log=False)

        return p_value
