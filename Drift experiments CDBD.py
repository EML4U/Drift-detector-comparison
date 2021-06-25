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
features = gradual_dict['train'][0]
targets = np.array(gradual_dict['train'][1])[:,1] # take the labels from dictionary, convert to np.array and slice to only get the scores
targets = targets.astype('int')

# get detector training data
features_d = gradual_dict['orig'][0]
# targets_d = np.array(gradual_dict['orig'][1])[:,1] # take the labels from dictionary, convert to np.array and slice to only get the scores
# targets_d = targets_d.astype('int')


# split `to train and test. Here, test will be used to train the CDBD drift detector
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, shuffle=False)
# j = 1
# while len(np.unique(y_train)) < len(np.unique(y_test)):
#     print("y_train does not have all the classes, spliting again")
#     x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, random_state=42*i+j)
#     j = j+1


# train a model
model = None
model = SVC(kernel='linear', random_state=42) # SVM model
model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)
# print("acc of the model on test data ", acc)

# Create detector
detector = CDBDDetector(model)
detector.fit(features_d, n_batch=20)

# dfirt detection on the original train data
# train_p_value = detector.predict_proba(x_train)
# test_p_value = detector.predict_proba(x_test)
# print("dfirt detection train ", train_p_value)
# print("dfirt detection test ", test_p_value)
# print("--------------")
# exit()

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
plt.savefig("figures/drift_detectors/uncertainty/amz768_l1_50b_bin9_run10_train_min.png")


# target_percentages = [0, 0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]

# plt.plot(p_s)
# plt.xticks(ticks=np.arange(len(target_percentages)), labels=target_percentages, rotation=60)
# plt.savefig('figures/drift_detectors/uncertainty/amz50_l1_20b_bin9_run10_samesampleR.png')

# plt.plot([1-x for x in p_s])
# plt.yscale('log')
# plt.xticks(ticks=np.arange(len(target_percentages)), labels=target_percentages, rotation=60)
# plt.savefig('figures/drift_detectors/uncertainty/amz50_l1_20b_bin9_run10_samesample-logR.png')
