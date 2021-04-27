import os
import sys
from embedding import BertHuggingface
import pickle
from datetime import datetime
import time

## First: Download the dataset
os.system('mkdir -p data/movies/embeddings')
os.system('wget -c https://snap.stanford.edu/data/movies.txt.gz')
os.system('gzip -d movies.txt.gz')
os.system('mv movies.txt data/movies/movies.txt')



## Second: embed everything, save in chunks so the memory doesn't explode

bert = BertHuggingface(8, batch_size=8)

counter = 0
batch_size = 100000
text_list = []
key_list = []
ident = []

with open('data/movies/movies.txt', 'r', errors='ignore') as f:
    for line in f:
        identifier = line.spl it(':')[0]
        if 'review/helpfulness' in identifier:
            helpfulness = line.split(':')[1].strip()
            ident.append(helpfulness)
        elif 'review/score' in identifier:
            score = line.split(':')[1].strip()
            ident.append(int(float(score)))
        elif 'review/time' in identifier:
            time_i = line.split(':')[1].strip()
            ident.append(datetime.fromtimestamp(int(time_i)))
            key_list.append(ident)
            ident = []
        elif 'review/text' in identifier:
            text = line.split(':')[1].strip().replace('<br />', ' ')
            text_list.append(text)
            
        if len(text_list) >= batch_size:
            # embedding
            embs = bert.embed(text_list)
            # save embedded data
            with open('data/movies/embeddings/{}.pickle'.format(counter), 'wb') as handle:
                pickle.dump((embs, key_list), handle)
            counter += 1
            # resetting
            text_list = []
            key_list = []
            print('At step', counter, 'of', 7_911_684/batch_size)
    embs = bert.embed(text_list)
    # save embedded data
    with open('data/movies/embeddings/{}.pickle'.format(counter), 'wb') as handle:
        pickle.dump((embs, key_list), handle)
    counter += 1
    # resetten
    text_list = []
    key_list = []
    print('At step', counter, 'of', 7_911_684/batch_size)

# Third: reload all the files into memory (should work for machines with more than 32Gb RAM, confirmed to work on 64Gb)
# and sort the reviews by time. Necessary because they are sorted by product code by default.
os.system('rm data/movies/movies.txt') # movies.txt no longer needed

embs = []
key = []
for counter in range(80):
    with open('data/movies/embeddings/{}.pickle'.format(counter), 'rb') as handle:
        data, keys = pickle.load(handle)
    embs.extend(data)
    key.extend(keys)

# in-place sorting magic
embs, key = (list(t) for t in zip(*sorted(zip(embs, key), key=lambda x: x[1][-1])))

with open('data/movies/embeddings/amazon_ordered_by_time{}.pickle'.format(''), 'wb') as handle:
    pickle.dump((embs, key), handle)
    
for i in range(counter + 1): # part files no longer needed
    os.system('data/movies/embeddings/{}.pickle'.format(i))
