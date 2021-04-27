import os
import sys
import pickle
from datetime import datetime
import time


text_list = []
key_list = []
ident = []

with open('data/movies/movies.txt', 'r', errors='ignore') as f:
    for line in f:
        identifier = line.split(':')[0]
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


text_list, key_list = (list(t) for t in zip(*sorted(zip(text_list, key_list), key=lambda x: x[1][-1])))

with open('data/movies/embeddings/amazon_raw.pickle', 'wb') as handle:
    pickle.dump((text_list, key_list), handle)