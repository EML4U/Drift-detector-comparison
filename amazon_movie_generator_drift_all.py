import os
import sys
import pickle
from datetime import datetime, timedelta
import time
import random
random.seed(42)
import re
import yaml

mode = "bert_768"
#mode = "bow_50"
num_samples = 500

target_percentages = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]
negative_words = ['waste', 'unwatchable', 'stinks', 'atrocious', 'yawn', 'ugh', 'abomination', 'garbage', 'worst', 'rubbish', 'defective', 'incoherent', 'ripoff', 'unconvincing', 'awful', 'dud', 'wasted', 'abysmal', 'travesty', 'wasting', 'poorly']

config            = yaml.safe_load(open("config.yaml", 'r'))
amazon_raw_file   = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "amazon_raw.pickle")
gensim_model_file = os.path.join(config["GENSIM_MODEL_DIRECTORY"], "amazonreviews.model") # amazonreviews_d.model
embeddings_file   = os.path.join(config["EMBEDDINGS_DIRECTORY"], "amazon_drift_" + mode + ".pickle")


## Embed everything, save in chunks so the memory doesn't explode

# Configure model to use by mode
if(mode == "bert_768"):
    from embedding import BertHuggingface
    bert = BertHuggingface(8, batch_size=8)
    embed = bert.embed
elif(mode == "bow_50"):
    from word2vec.Word2Vec import Word2Vec
    word2vec = Word2Vec(gensim_model_file)
    word2vec.prepare()
    embed = word2vec.embed
else:
    raise ValueError("Unknown mode " + mode)

with open(amazon_raw_file, 'rb') as handle:
    texts, keys = pickle.load(handle)
for i in range(len(keys)):
    keys[i][1] -= 1   # fix class names from 1..5 to 0..4 for easier 1-hot encoding

# gather amazon reviews of the fourth year only
classes = [[] for x in range(5)]
data = [x for x in list(zip(texts, keys)) if keys[0][-2] + timedelta(days=365*4) < x[1][-2] < keys[0][-2] + timedelta(days=365*(4+1))]
for point in data:
    classes[point[1][1]].append(point)
for i in range(len(classes)):
    random.shuffle(classes[i])

original_data = []
drift_data = []
for i in range(len(classes)):
    original_data.extend(classes[i][:num_samples])
    drift_data.extend(classes[i][num_samples:2*num_samples])

indices = []
more_than_one = 0
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
    
with open(embeddings_file, 'wb') as handle:
    pickle.dump(dump_data, handle)
