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

start_file = 'data/movies/embeddings/amazon_{mode}_same_dist.pickle'.format(mode=mode)
embeddings_file   = 'data/movies/embeddings/amazon_{mode}_different_classes.pickle'.format(mode=mode)

with open(start_file, 'rb') as handle:
    permutations_embs, permutation_keys = pickle.load(handle)['data']

    
embs = []
keys = []

for e in permutations_embs:
    embs.extend(e)
    
for k in permutation_keys:
    keys.extend(k)
    

both = zip(embs, keys)

classes = [[] for i in range(5)]

for point in both:
    classes[point[1][1]].append(point)

classes_unzipped  = []
for class_ in classes:
    classes_unzipped.append([list(t) for t in zip(*class_)])
    


with open(embeddings_file, 'wb') as handle:
    pickle.dump(classes_unzipped, handle)
    