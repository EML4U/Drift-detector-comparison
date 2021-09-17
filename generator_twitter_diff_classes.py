import os
import sys
import pickle
from datetime import datetime, timedelta
import time
import random
random.seed(42)
import re


modes = ['bert_768', 'bow_50', 'bow_768']
if len(sys.argv) < 2 or sys.argv[1] not in modes:
    print('Need mode {mode} as parameter!'.format(mode=modes))
    exit(1)

mode = sys.argv[1]

embeddings_file_b   = 'data/twitter/biden_{mode}_embeddings.pickle'.format(mode=mode)
embeddings_file_t   = 'data/twitter/trump_{mode}_embeddings.pickle'.format(mode=mode)
gensim_model_50_file  = 'data/twitter/twitter_election_model/twitter_election_50.model'
gensim_model_768_file = 'data/twitter/twitter_election_model/twitter_election_768.model'

if os.path.isfile(embeddings_file_b) and os.path.isfile(embeddings_file_t):  # Do not overwrite
    print("Embeddings file already exists, exiting.", embeddings_file_t)
    exit()

with open('data/twitter/election_dataset_raw.pickle', 'rb') as handle:
    twitter = pickle.load(handle)
    
    
biden, trump = twitter['biden'], twitter['trump']

biden = [x for x in biden if 'trump' not in x[1].lower()]
trump = [x for x in trump if 'biden' not in x[1].lower()]

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

def embed_tweets(data):
    times, tweets = zip(*data)
    embs = embed(tweets)
    z = zip(embs, times)
    return list(z)


embs_biden = embed_tweets(biden)
with open(embeddings_file_b, 'wb') as handle:
    pickle.dump(embs_biden, handle)
    

embs_trump = embed_tweets(trump)
with open(embeddings_file_t, 'wb') as handle:
    pickle.dump(embs_trump, handle)