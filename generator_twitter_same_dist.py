import os
import sys
import pickle
from datetime import datetime, timedelta
import time
import random
random.seed(42)
import re
import yaml



modes = ['bert_768', 'bow_50', 'bow_768']
if len(sys.argv) < 2 or sys.argv[1] not in modes:
    print('Need mode {mode} as parameter!'.format(mode=modes))
    exit(1)

mode = sys.argv[1]

num_samples = 500
num_permutations = 20


# embeddings_file    is to be generated here
start_file_b = 'data/twitter/biden_{}_embeddings.pickle'.format(mode)
start_file_t = 'data/twitter/trump_{}_embeddings.pickle'.format(mode)
embeddings_file   = 'data/twitter/twitter_{mode}_same_dist.pickle'.format(mode=mode)


with open(start_file_b, 'rb') as handle:
    biden = pickle.load(handle)
with open(start_file_t, 'rb') as handle:
    trump = pickle.load(handle)
    
# sanity check: dont create to many permutations if there are not enough datapoints
classes = [biden, trump]
mini = min([len(x) for x in classes])
if mini < num_permutations * num_samples:
    print('WARNING: too few samples for {n} permutations, doing '.format(num_permutations), end='')
    num_permutations = int(mini/num_samples)
    print(' {} permutations instead!'.format(num_permutations))
    

for i in range(len(classes)):
    random.shuffle(classes[i])

data = []
for perm in range(num_permutations):
    entry = []
    for i in range(len(classes)):
        entry.extend(classes[i][:num_samples])
    data.append(entry)
    
    
embeddings = []
e_keys = []
for d in data:
    emb_texts, emb_keys = [list(t) for t in zip(*d)]
    embeddings.append(embed(emb_texts))
    e_keys.append(emb_keys)


dump_data = {'data': (embeddings, e_keys)}
    
with open(embeddings_file, 'wb') as handle:
    pickle.dump(dump_data, handle)