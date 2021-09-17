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
    bert.save('data/movies/movie_{epoch}e'.format(epoch=e))
    
    
# Results after 1 epochs: (0.6176892647813851, 0.622, 0.6167358103186324, None)
# Results after 2 epochs: (0.6475851563698874, 0.65, 0.6468360798829099, None)
# Results after 3 epochs: (0.6940775111279727, 0.696, 0.6921234953784917, None)
# Results after 4 epochs: (0.7169389074280628, 0.717, 0.7128005341769665, None)
# Results after 5 epochs: (0.7517967654497437, 0.752, 0.7499616280673936, None)
# Results after 6 epochs: (0.7620419789779298, 0.761, 0.7575083466262609, None)
# Results after 7 epochs: (0.7839583454008342, 0.784, 0.7816583503573143, None)
# Results after 8 epochs: (0.7995420201064476, 0.8, 0.798613564692751, None)
# Results after 9 epochs: (0.8073645407216155, 0.807, 0.8049110130226318, None)
