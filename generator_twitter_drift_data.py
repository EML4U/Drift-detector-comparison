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
num_samples = 10000  # Will be taken from each class, e.g. 10,000*2=20,000 samples overall

if len(sys.argv) < 2 or sys.argv[1] not in modes:
    print('Need mode {mode} as parameter!'.format(mode=modes))
    exit(1)

mode = sys.argv[1]

# amazon_raw_file    generated by amazon_movie_sorter.py
# gensim_model_files available at https://hobbitdata.informatik.uni-leipzig.de/EML4U/2021-05-17-Amazon-Doc2Vec/
#                    generated by word2vec/doc2vec.py
# embeddings_file    is to be generated here
# sample_file        also generated here, ensures same data for each model
twitter_raw_file   = 'data/twitter/election_dataset_raw.pickle'
#gensim_model_50_file  = 'data/twitter/amazonreviews_d.model' TODO
#gensim_model_768_file = 'data/twitter/amazonreviews_e.model' TODO
embeddings_file       = 'data/twitter/twitter_drift_{}.pickle'.format(mode)

#target_percentages = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]
target_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # From every 40th word to every second word

negative_words = ['waste', 'unwatchable', 'stinks']
if(True): # ratio at least 2.0
    negative_words.extend(['atrocious', 'yawn', 'ugh', 'abomination', 'stupidest', 'garbage', 'laughably', 'worst', 'rubbish', 'defective', 'incoherent', 'ripoff', 'unconvincing', 'awful', 'dud', 'wasted', 'abysmal', 'travesty', 'wasting'])
if(False): # ratio at least 1.6
    negative_words.extend(['poorly', 'disgrace', 'lame', 'insulting', 'lousy', 'disapointment', 'pointless', 'horrible', 'insult', 'laughable', 'terrible', 'moronic'])
if(False): # ratio at least 1.5
    negative_words.extend(['sucks', 'idiotic', 'boycott', 'froze', 'nauseating', 'worthless', 'shoddy'])

if os.path.isfile(embeddings_file):  # Do not overwrite
    print("Embeddings file already exists, exiting.", embeddings_file)
    exit()

time_begin = time.time()
print(datetime.fromtimestamp(time_begin))

# Configure model to use by mode
print("mode", mode)
print("amazon_raw_file", amazon_raw_file)
print("embeddings_file", embeddings_file)
print("sample_file", sample_file)
if(mode == "bert_768"):
    from embedding import BertHuggingface
    bert = BertHuggingface(8, model_name='bert-base-multilingual-cased', batch_size=8)
    embed = bert.embed
elif(mode == "bow_50"):
    print("gensim_model_50_file", gensim_model_50_file)
    from word2vec.Word2Vec import Word2Vec
    word2vec = Word2Vec(gensim_model_50_file)
    word2vec.prepare()
    embed = word2vec.embed
elif(mode == "bow_768"):
    print("gensim_model_768_file", gensim_model_768_file)
    from word2vec.Word2Vec import Word2Vec
    word2vec = Word2Vec(gensim_model_768_file)
    word2vec.prepare()
    embed = word2vec.embed
elif(mode == "dummy"):
    print("Skipping model")
    def justpass(x): 
        pass
    embed = justpass
else:
    raise ValueError("Unknown mode " + mode)
    
    ###########################################################################
    
with open(twitter_raw_file, 'rb') as handle:
    twitter = pickle.load(handle)
    
    
biden, trump = twitter['biden'], twitter['trump']

biden = [x for x in biden if 'trump' not in x[1].lower()]
trump = [x for x in trump if 'biden' not in x[1].lower()]



    ###########################################################################



# Load data
print("Loading data")
if os.path.isfile(sample_file):
    with open(sample_file, 'rb') as handle:
        original_data = pickle.load(handle)
        drift_data = pickle.load(handle)
        train_data = pickle.load(handle)
    print("Loaded", len(drift_data), "|", len(original_data), "|", len(train_data), "samples")
else:
    docs_in_years = {}
    with open(amazon_raw_file, 'rb') as handle:
        texts, keys = pickle.load(handle)
    for i in range(len(keys)):
        keys[i][1] -= 1   # fix class names from 1..5 to 0..4 for easier 1-hot encoding
        docs_in_years[keys[i][-2].year] = docs_in_years.get(keys[i][-2].year , 0) + 1
    print("Docs in year overview:", docs_in_years)
    print("Example keys", keys[0])

    # Gather amazon reviews of year range only
    # e.g. classes[2] = [..., ("text", ['7/15', 2, datetime.datetime(1997, 12, 19, 1, 0), 39862]), ...]
    classes = [[] for x in range(5)]
    for i in range(len(keys)):
        if(keys[i][-2].year == year):
            classes[keys[i][1]].append((texts[i],keys[i]))

    # Randomize order and get samples
    # Same amount for original data and drift data
    for i in range(len(classes)):
        random.shuffle(classes[i])
    original_data = []
    drift_data = []
    train_data = []
    for i in range(len(classes)):
        original_data.extend(classes[i][:num_samples])
        drift_data.extend(classes[i][num_samples:2*num_samples])
        train_data.extend(classes[i][2*num_samples:3*num_samples])
        
    #sample_data = {'original_data': original_data, 'drift_data': drift_data}
    with open(sample_file, 'wb') as handle:
        pickle.dump(original_data, handle)
        pickle.dump(drift_data, handle)
        pickle.dump(train_data, handle)
    print("Created", len(drift_data), "|", len(original_data), "|", len(train_data), "samples")

# Injection
# 0.5 means insertion of 1 words in 50 percent of texts
# 2.0 means insertion of 2 words in each text
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

if(False):  # Injection test
    texts = ["0 0 0 0","1 1 1 1","2 2 2 2","3 3 3 3","4 4 4 4","5 5 5 5","6 6 6 6","7 7 7 7","8 8 8 8","9 9 9 9"]
    print(inject(texts, 0.5))
    print(inject(texts, 2))
    exit()

# Inject percentages
drifted_embeddings = []  
drifted_texts, drifted_keys = [list(t) for t in zip(*drift_data)]
drifted_embeddings.append(embed(drifted_texts))
for num, perc in enumerate(target_percentages):
    print("Incecting", num, "|", perc)
    drifted_texts = inject(drifted_texts, perc)
    drifted_embeddings.append(embed(drifted_texts))

# Save
original_texts, original_keys = [list(t) for t in zip(*original_data)]
original_embs = embed(original_texts)
train_texts, train_keys = [list(t) for t in zip(*train_data)]
train_embs = embed(train_texts)
dump_data = {'orig': (original_embs, original_keys), 'drifted': (drifted_embeddings, drifted_keys), 'train': (train_embs, train_keys)}
with open(embeddings_file, 'wb') as handle:
    pickle.dump(dump_data, handle)

# Stats
time_end = time.time()
runtime = time_end - time_begin;
print("Runtime: %s minutes" % (runtime/60))
