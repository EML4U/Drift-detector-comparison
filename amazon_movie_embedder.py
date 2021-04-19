import os
import sys
from embedding import BertHuggingface
import pickle
from datetime import datetime
import time

bert = BertHuggingface(8, batch_size=8)

counter = 0
batch_size = 100000
text_list = []
key_list = []
ident = []

with open('data/movies/movies.txt', 'r', errors='ignore') as f:
    for line in f:
        identifier = line.spl it(':')[0]
        if 'review/helpfulness:' in identifier:
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
