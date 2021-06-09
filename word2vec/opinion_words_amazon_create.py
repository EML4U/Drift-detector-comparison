# Create overview of negative words usage in Amazon Reviews

import yaml
import os.path
import pickle
    
config            = yaml.safe_load(open("../config.yaml", 'r'))
pickle_score_one  = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-one.pickle")
pickle_score_five = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-five.pickle")

if(True):
    # Use small test dataset
    pickle_score_one      = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-one.pickle")
    pickle_score_five     = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-five.pickle")

with open(pickle_score_one, 'rb') as handle:
    negative_score_one = pickle.load(handle)
with open(pickle_score_five, 'rb') as handle:
    negative_score_five = pickle.load(handle)

print(negative_score_one)
print(negative_score_five)