import yaml
import os.path
import pickle
import gensim
import time

# Config 50
doc2vec_vector_size = [50, 768]  # Dimensionality of the feature vectors
doc2vec_min_count   = 2   # Ignores all words with total frequency lower than this
doc2vec_epochs      = 40  # Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec
doc2vec_dm          = 1   # Training algorithm, distributed memory (PV-DM) or distributed bag of words (PV-DBOW)
num_of_tweets       = -1  # -1 to process all (for development)


# Set data paths
#config            = yaml.safe_load(open("../config.yaml", "r"))
twitter_file      = "../data/twitter/election_dataset_raw.pickle"
gensim_model_file = "../data/twitter/twitter_election_model/twitter_election_{dim}.model"

    
    
    
print("Loading twitter file", twitter_file)
with open(twitter_file, "rb") as handle:
    data = pickle.load(handle)

all_tweets = len(data["biden"])+len(data["trump"])
print("all_tweets", all_tweets)



# Data exploration
if(False):
    print(type(data))
    for key in data:
        print(key)
        print(type(data[key]), len(data[key]))
        print(type(data[key][0]), len(data[key][0]))
        print(data[key][0][0])
        print(data[key][0][1])
        print()
        
        
        
# Doc2Vec method
# twitter_list: list of tuples in form (date, tweet)
# mode:         tagdoc or tokens or text
# max:          maximim tweets to process
def embedd(twitter_list, mode="tagdoc", max=-1):
    results = []
    i = 0
    for tup in twitter_list:
        
        if(max != -1 and i >= max):
            break
        i += 1
        
        if(mode == "text"):
            results.append(tup[1])
            continue

        tokens = gensim.utils.simple_preprocess(tup[1])
        if(mode == "tokens"):
            results.append(tokens)
            continue
        
        if(mode != "tagdoc"):
            raise ValueError("Unknown mode", mode)
            
        results.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))

    return results



print("Creating tagged documents")
mode = "tagdoc"
num = num_of_tweets
if(num != -1):
    num = num/2
tagdocs = embedd(data["biden"], mode, num) + embedd(data["trump"], mode, num)

for vec_size in doc2vec_vector_size:

    print("Building vocabulary")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=doc2vec_min_count, epochs=doc2vec_epochs, dm=doc2vec_dm)
    model.build_vocab(tagdocs)



    print("Training model")
    time_begin = time.time()
    model.train(tagdocs, total_examples=model.corpus_count, epochs=model.epochs)
    time_end = time.time()



    print("Saving model file", gensim_model_file.format(dim=vec_size))
    model.save(gensim_model_file.format(dim=vec_size))



    # Info
    print(model)
    runtime = time_end - time_begin;
    print("Runtime: %s minutes" % (runtime/60))
    print("Gensim version:", gensim.__version__)



    # Estimate runtime
    if(num_of_tweets != -1):
        mean_runtime = runtime / num_of_tweets
        print(num_of_tweets, vec_size, doc2vec_epochs, mean_runtime, all_tweets * mean_runtime / 60)

# tweets  dim  it  sec/doc  estimation (min)
# 5000     50  40  0.0040   80.6454
# 5000    768  40  0.0044   88.7797
