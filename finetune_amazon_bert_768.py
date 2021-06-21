from embedding import BertHuggingface
import pickle
from Classifier import balance


year = 2011

bert = BertHuggingface(5, batch_size=8)
#bert.verbose = True


with open('data/movies/embeddings/amazon_raw.pickle', 'rb') as handle:
    texts, keys = pickle.load(handle)
for i in range(len(keys)):
    keys[i][1] -= 1   # fix class names from 1..5 to 0..4 for easier 1-hot encoding
    
data = [x for x in list(zip(texts, keys)) if x[1][-2].year == year] # gather amazon reviews of the fourth year only

# finetune
texts, keys = [list(t) for t in zip(*data)]
keys = [x[1] for x in keys]

texts, keys = balance(texts, keys)


def one_epoch():
    bert.retrain_one_epoch(texts[1000:], keys[1000:])
    f1 = bert.eval(texts[:1000], keys[:1000])
    print(f1)

for e in range(1, 10):
    print('Results after {epoch} epochs: '.format(epoch=e), end='')
    one_epoch()
    bert.save('movie_{epoch}e'.format(epoch=e))