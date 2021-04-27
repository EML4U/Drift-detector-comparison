import os
import sys
from embedding import BertHuggingface
import pickle
from datetime import datetime, timedelta
import time
import random
random.seed(42)

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
        
embs = [[],[]]
for year in range(len(new)):
    t, k = [list(t) for t in zip(*new[year])]
    embs[0].extend(bert.embed(t))
    embs[1].extend(k)

    
with open('data/movies/embeddings/amazon_small.pickle', 'wb') as handle:
    pickle.dump(embs, handle)
