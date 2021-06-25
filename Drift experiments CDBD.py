import numpy as np
import pickle
import matplotlib.pyplot as plt
from detectors.CDBDDetector import CDBDDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

file = '../data/amazon_drift_bow_50.pickle'
# file = '../data/amazon_drift_bow_768.pickle'

# load the data
with open(file, 'rb') as handle:
    gradual_dict = pickle.load(handle)

# get model training data
# features = gradual_dict['train'][0]
# targets = np.array(gradual_dict['train'][1])[:,1] # take the labels from dictionary, convert to np.array and slice to only get the scores
# targets = targets.astype('int')

# get detector training data
features_d = gradual_dict['orig'][0]
targets_d = np.array(gradual_dict['orig'][1])[:,1] # take the labels from dictionary, convert to np.array and slice to only get the scores
targets_d = targets_d.astype('int')


# split `to train and test. Here, test will be used to train the CDBD drift detector
# x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, shuffle=False)


# # train a model
# model = None
# model = SVC(kernel='linear', random_state=42) # SVM model
# model.fit(x_train, y_train)

# Create detector
detector = CDBDDetector()
detector.fit(features_d, targets_d, n_batch=50)

runs_p_s = []
for i in range(10):
    p_s = []
    for percentage in gradual_dict['drifted'][0]:
        # index = np.random.choice(percentage.shape[0], detector.batch_size, replace=False) 
        # sample_percentage = percentage[index]
        sample_percentage = percentage[i*detector.batch_size:(i+1)*detector.batch_size]
        p_value = detector.predict_proba(sample_percentage)
        p_s.append(p_value)
        # print(p_value)
    runs_p_s.append(p_s)
    print(f"run {i} done")

p_s = np.array(runs_p_s)
p_s = p_s.mean(axis=0)

# plot results
plt.plot(p_s)
plt.savefig("figures/drift_detectors/uncertainty/amz50_final50.png")