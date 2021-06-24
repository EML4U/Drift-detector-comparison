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


start_file_b = 'data/twitter/biden_{}_embeddings.pickle'.format(mode)
start_file_t = 'data/twitter/trump_{}_embeddings.pickle'.format(mode)
embeddings_file   = 'data/twitter/twitter_{mode}_diff_dist.pickle'.format(mode=mode)


with open(start_file_b, 'rb') as handle:
    biden = pickle.load(handle)
with open(start_file_t, 'rb') as handle:
    trump = pickle.load(handle)

    

d100 = min([x[1] for x in biden]) + timedelta(hours=100)
d124 = min([x[1] for x in biden]) + timedelta(hours=124)
d192 = min([x[1] for x in biden]) + timedelta(hours=192)
d480 = min([x[1] for x in biden]) + timedelta(hours=480)
    
b_100 = [b for b in biden if b[1] > d100 and b[1] < d100 + timedelta(hours=24)]
b_124 = [b for b in biden if b[1] > d124 and b[1] < d124 + timedelta(hours=24)]
b_192 = [b for b in biden if b[1] > d192 and b[1] < d192 + timedelta(hours=24)]
b_480 = [b for b in biden if b[1] > d480 and b[1] < d480 + timedelta(hours=24)]
    
t_100 = [t for t in trump if t[1] > d100 and t[1] < d100 + timedelta(hours=24)]
t_124 = [t for t in trump if t[1] > d124 and t[1] < d124 + timedelta(hours=24)]
t_192 = [t for t in trump if t[1] > d192 and t[1] < d192 + timedelta(hours=24)]
t_480 = [t for t in trump if t[1] > d480 and t[1] < d480 + timedelta(hours=24)]
    
    
b_100 = b_100[:min(len(b_100),len(t_100))]
t_100 = t_100[:min(len(b_100),len(t_100))]

b_124 = b_124[:min(len(b_124),len(t_124))]
t_124 = t_124[:min(len(b_124),len(t_124))]

b_192 = b_192[:min(len(b_192),len(t_192))]
t_192 = t_192[:min(len(b_192),len(t_192))]

b_480 = b_480[:min(len(b_480),len(t_480))]
t_480 = t_480[:min(len(b_480),len(t_480))]
    
data = {}

chunk_size= 500

b_100_chunks = [[list(x) for x in zip(*b_100[i:i + chunk_size])][0] for i in range(0, len(b_100), chunk_size)][:-1]
t_100_chunks = [[list(x) for x in zip(*t_100[i:i + chunk_size])][0] for i in range(0, len(t_100), chunk_size)][:-1]

data['100'] = (b_100_chunks, t_100_chunks)

b_124_chunks = [[list(x) for x in zip(*b_124[i:i + chunk_size])][0] for i in range(0, len(b_124), chunk_size)][:-1]
t_124_chunks = [[list(x) for x in zip(*t_124[i:i + chunk_size])][0] for i in range(0, len(t_124), chunk_size)][:-1]

data['124'] = (b_124_chunks, t_124_chunks)

b_192_chunks = [[list(x) for x in zip(*b_192[i:i + chunk_size])][0] for i in range(0, len(b_192), chunk_size)][:-1]
t_192_chunks = [[list(x) for x in zip(*t_192[i:i + chunk_size])][0] for i in range(0, len(t_192), chunk_size)][:-1]

data['192'] = (b_192_chunks, t_192_chunks)

b_480_chunks = [[list(x) for x in zip(*b_480[i:i + chunk_size])][0] for i in range(0, len(b_480), chunk_size)][:-1]
t_480_chunks = [[list(x) for x in zip(*t_480[i:i + chunk_size])][0] for i in range(0, len(t_480), chunk_size)][:-1]

data['480'] = (b_480_chunks, t_480_chunks)


with open(embeddings_file, 'wb') as handle:
    pickle.dump(data, handle)
    
    
exit()
