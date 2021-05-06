import os
import sys
from embedding import BertHuggingface
import pickle
from datetime import datetime, timedelta
import time
import random
random.seed(42)
import re

# target percentages for 
target_percentages = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
num_samples = 500

# variables used later
indices = []

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
    
    
original_data = []
drift_data = []
classes = [[] for x in range(5)]
data = [x for x in list(zip(texts, keys)) if keys[0][-1] + timedelta(days=365*4) < x[1][-1] < keys[0][-1] + timedelta(days=365*(4+1))] # gather amazon reviews of the fourth year only

for point in data:
    classes[point[1][1]].append(point)
for i in range(len(classes)):
    random.shuffle(classes[i])

for i in range(len(classes)):
    original_data.extend(classes[i][:num_samples])
    drift_data.extend(classes[i][num_samples:2*num_samples])
        

# add words            
negative_words = []
with open('data/sentiment_words/negative-words.txt', 'r', errors='ignore') as f:
    for line in f:
        if line and line.strip() and not line.startswith(';'):
            negative_words.append(line.strip())
            
    
def inject(texts, percent):
    """ texts will be injected so that they then are at a injection rate of #percent
    """
    num = int(percent*len(texts)) - len(indices)
    for _ in range(num):
        # find a text which hasnt been injected yet
        injection_index = random.choice([x for x in range(len(texts)) if x not in indices])
        indices.append(injection_index)
        # actual injection process
        spaces = [m.start() for m in re.finditer(' ', texts[injection_index])]
        if spaces:
            injection_point = random.choice(spaces)
        else:
            injection_point = len(texts[injection_index])
        t = texts[i]
        t = t[:injection_point] + ' ' + random.choice(negative_words) + t[injection_point:]
        texts[i] = t
    return texts
                     
drifted_embeddings = []  
drifted_texts, drifted_keys = [list(t) for t in zip(*drift_data)]
drifted_embeddings.append(bert.embed(drifted_texts))

for num, perc in enumerate(target_percentages):
    drifted_texts = inject(drifted_texts, perc)
    drifted_embeddings.append(bert.embed(drifted_texts))
    
original_texts, original_keys = [list(t) for t in zip(*original_data)]
original_embs = bert.embed(original_texts)

dump_data = {'orig': (original_embs, original_keys), 'drifted': (drifted_embeddings, drifted_keys)}
    
with open('data/movies/embeddings/amazon_small_gradual_drift.pickle', 'wb') as handle:
    pickle.dump(dump_data, handle)