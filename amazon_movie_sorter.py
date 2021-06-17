# Reads movies.txt.gz
# Sorts data by date
# Stores as pickle
# - list of keys: helpfulness, score, time, number
# - list of texts (concatenated fields summary and text)

import os
import pickle
from datetime import datetime
import yaml
from word2vec.amazon_reviews_reader import AmazonReviewsReader
import time

config          = yaml.safe_load(open("config.yaml", 'r'))
amazon_gz_file  = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "movies.txt.gz")
amazon_raw_file = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "amazon_raw.pickle")
max_docs        = -1  # -1 for all

text_list = []
key_list = []
ident = []

timeBegin = time.time()
print("Begin", time.asctime())
print("Reading", amazon_gz_file)

for item in AmazonReviewsReader(amazon_gz_file, "fields", max_docs=max_docs):
    ident.append(item['helpfulness'])
    ident.append(int(float(item['score'])))
    ident.append(datetime.fromtimestamp(int(item['time'])))
    ident.append(item['number'])
    key_list.append(ident)
    ident = []
    text_list.append((item['summary'] + " " + item['text']).replace('<br />', ' '))

text_list, key_list = (list(t) for t in zip(*sorted(zip(text_list, key_list), key=lambda x: x[1][-2])))

print("Writing", amazon_raw_file)
with open(amazon_raw_file, 'wb') as handle:
    pickle.dump((text_list, key_list), handle)

print("Runtime (secs)", time.time()-timeBegin)