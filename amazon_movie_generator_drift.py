import os
import sys
from embedding import BertHuggingface
import pickle
from datetime import datetime
import time
import random

## First: Download the dataset
#os.system('mkdir -p data/movies/embeddings')
#os.system('wget -c https://snap.stanford.edu/data/movies.txt.gz')
#os.system('gzip -d movies.txt.gz')
#os.system('mv movies.txt data/movies/movies.txt')



## Second: embed everything, save in chunks so the memory doesn't explode

bert = BertHuggingface(8, batch_size=8)

with open('data/movies/embeddings/amazon_raw.pickle', 'rb') as handle:
    texts, keys = pickle.load(handle)
for i in range(len(keys)):
    keys[i][1] -= 1   # fix class names from 1..5 to 0..4 for easier 1-hot encoding
    
    
new = 16*[[]]
for year in range(len(new)):
    classes = 5*[[]]
    data = [x for x in list(zip(texts, keys)) if keys[0][-1] + timedelta(days=365*year) < x[1][-1] < keys[0][-1] + timedelta(days=365*(year+1))] # gather amazon reviews of the third year only
    
    for point in data:
        classes[point[1][1]].append(point)
    for i in range(len(classes)):
        random.shuffle(classes[i])
    
    for i in range(len(classes)):
        new[year] = classes[i][:1000]
        
#data = [list(t) for t in zip(*data)]
    
# add words
positive_words = []
with open('data/sentiment_words/positive-words.txt', 'r', errors='ignore') as f:
    for line in f:
        if line and line.strip() and not line.startswith(';'):
            positive_words.append(line.strip())
            
negative_words = []
with open('data/sentiment_words/negative-words.txt', 'r', errors='ignore') as f:
    for line in f:
        if line and line.strip() and not line.startswith(';'):
            negative_words.append(line.strip())
            
            
            
embs = [[],[]]
for year in range(len(new)):
    t, k = [list(t) for t in zip(*new[year])]
    embs[0].extend(bert.embed([x + ' ' + random.choice(negative_words) for x in t]))
    embs[1].extend(k)

    
with open('data/movies/embeddings/amazon_drift_ordered_by_time.pickle', 'wb') as handle:
    pickle.dump(embs, handle)