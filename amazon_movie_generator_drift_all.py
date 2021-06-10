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
target_percentages = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]
num_samples = 500

# variables used later
indices = []
more_than_one = 0

mode = "bert-768"

## First: Download the dataset
#os.system('mkdir -p data/movies/embeddings')
#os.system('wget -c https://snap.stanford.edu/data/movies.txt.gz')
#os.system('gzip -d movies.txt.gz')
#os.system('mv movies.txt data/movies/movies.txt')



## Second: embed everything, save in chunks so the memory doesn't explode

# Configure model by mode
if(mode == "bert-768"):
    bert = BertHuggingface(8, batch_size=8)
    embed = bert.embed
elif(mode == "bow-50"):
    print("bow")
    # TODO Required: funtion(text-list) -> list of embeddings
    # https://github.com/UBI-AGML-NLP/Embeddings/blob/main/embedding/doc2vec.py 
    # list to docs -> foreach
    #bow = Doc2Vec.load(gensim_model_file)
    #tokens = gensim.utils.simple_preprocess(___TEXT___)
    #next(iter(amazon_reviews_reader))
    #embed = bow.infer_vector(next.iter tokens)
else:
    print("error")
    exit()

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
#negative_words = ['bad', 'horrible']
negative_words = ['waste', 'unwatchable', 'stinks', 'atrocious', 'yawn', 'ugh', 'abomination', 'garbage', 'worst', 'rubbish', 'defective', 'incoherent', 'ripoff', 'unconvincing', 'awful', 'dud', 'wasted', 'abysmal', 'travesty', 'wasting', 'poorly']

#with open('data/sentiment_words/negative-words.txt', 'r', errors='ignore') as f:
#    for line in f:
#        if line and line.strip() and not line.startswith(';'):
#            negative_words.append(line.strip())
            
    
def inject(texts, percent):
    """ texts will be injected so that they then are at a injection rate of #percent
    """
    global indices
    global more_than_one
    num = int(percent*len(texts)) - (more_than_one*len(texts) + len(indices))
    for _ in range(num):
        # find a text which hasnt been injected yet
        not_yet_injected = [x for x in range(len(texts)) if x not in indices]
        if not not_yet_injected:
            indices = []
            more_than_one += 1
            not_yet_injected = [x for x in range(len(texts))]
        injection_index = random.choice(not_yet_injected)
        indices.append(injection_index)
        # actual injection process
        spaces = [m.start() for m in re.finditer(' ', texts[injection_index])]
        if spaces:
            injection_point = random.choice(spaces)
        else:
            injection_point = len(texts[injection_index])
        t = texts[injection_index]
        t = t[:injection_point] + ' ' + random.choice(negative_words) + t[injection_point:]
        texts[injection_index] = t
    return texts
                     
drifted_embeddings = []  
drifted_texts, drifted_keys = [list(t) for t in zip(*drift_data)]
drifted_embeddings.append(embed(drifted_texts))

for num, perc in enumerate(target_percentages):
    drifted_texts = inject(drifted_texts, perc)
    drifted_embeddings.append(embed(drifted_texts))

    
original_texts, original_keys = [list(t) for t in zip(*original_data)]
original_embs = embed(original_texts)

dump_data = {'orig': (original_embs, original_keys), 'drifted': (drifted_embeddings, drifted_keys)}
    
with open('data/movies/embeddings/amazon_bert_small_gradual_drift_100.pickle', 'wb') as handle:
    pickle.dump(dump_data, handle)