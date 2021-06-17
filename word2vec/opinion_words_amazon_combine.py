# Create overview of negative words usage in Amazon Reviews

import yaml
import os.path
import pickle
import csv

    
config            = yaml.safe_load(open("../config.yaml", 'r'))
pickle_score_one  = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-one.pickle")
pickle_score_five = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words-score-five.pickle")
pickle_results = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words.pickle")
csv_results    = os.path.join(config["OPINION_WORDS_DIRECTORY"], "negative-words.csv")

if(False):
    # Use small test dataset
    pickle_score_one  = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-one.pickle")
    pickle_score_five = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words-score-five.pickle")
    pickle_results    = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words.pickle")
    csv_results       = os.path.join(config["OPINION_WORDS_DIRECTORY"], "small-negative-words.csv")


# Read
with open(pickle_score_one, 'rb') as handle:
    negative_score_one = pickle.load(handle)
with open(pickle_score_five, 'rb') as handle:
    negative_score_five = pickle.load(handle)

# Get all negative words / keys
set_1 = set(negative_score_one.keys())
set_2 = set(negative_score_five.keys())
list_2_items_not_in_list_1 = list(set_2 - set_1)
all_keys = list(negative_score_one.keys()) + list_2_items_not_in_list_1
if(False):
    print(len(negative_score_one.keys()))
    print(len(negative_score_five.keys()))
    print(len(list_2_items_not_in_list_1))
    print(len(all_keys), len(negative_score_one.keys())+len(list_2_items_not_in_list_1))

# Combine values into tuples
data = []
for key in all_keys:
    score_one_count  = negative_score_one.get(key, 0)
    score_five_count = negative_score_five.get(key, 0)
    data.append((key, score_one_count, score_five_count))

# Sort by result / difference
data = sorted(data, key=lambda tup: tup[1], reverse=True)

# Save
with open(pickle_results, 'wb') as handle:
    pickle.dump(data, handle)
with open(csv_results ,'w') as out:
    csv_out=csv.writer(out)
    for row in data:
        csv_out.writerow(row)

# Print
if(False):
    print(data)
