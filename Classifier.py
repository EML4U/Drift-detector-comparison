import os
import sys
import pickle
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
random.seed(42)

def one_hot_enc(y):
    Y = []
    for c in y:
        l = 5*[0]
        l[c] = 1
        Y.append(l)
    return Y

def balance(x, y, shuffle=True):
    amounts = []
    for i in range(5):
        amount = len([x for x in y if x == i])
        amounts.append(amount)
    amount = min(amounts)
    pairs = list(zip(x, y))
    if shuffle:
        random.shuffle(pairs)
    wanted = []
    for i in range(5):
        wanted.extend([x for x in pairs if x[1] == i][:amount])
    random.shuffle(wanted)
    x, y = (list(t) for t in zip(*wanted))
    return x, y

class ScorePredictor():
    def __init__(self):
        inputs = keras.Input(shape=(768,))
        #x = keras.layers.Dropout(0.2)(inputs) # TODO: try out / may be interesting for Hossein
        x = inputs
        outputs = keras.layers.Dense(5, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam()
        loss = keras.losses.CategoricalCrossentropy()
        metric = keras.metrics.CategoricalAccuracy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    def predict(self, inputs):
        inputs = np.array(inputs)
        return np.argmax(self.model.predict(inputs), axis=1)
    
    def evaluate(self, X, Y):
        y = one_hot_enc(Y)
        x, y = np.array(X), np.array(y)
        return self.model.evaluate(x, y)
    
    def train(self, X, Y, patience=20, verbose=True):
        X, Y = balance(X, Y) # fix potential class imbalance
        y = one_hot_enc(Y) # one-hot encode labels
        x, y = np.array(X), np.array(y) # stupid keras rejects lists
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, restore_best_weights=True) # dont train longer then necessary
        self.model.fit(x, y, epochs=1000,
                       verbose=verbose,
                       callbacks=[callback],
                       validation_split=0.2
                      )
    
    def safe(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = keras.models.load_model(path)
            

# if invoked as script, train a classifier
if __name__ == '__main__':
    from datetime import timedelta
    from sklearn.model_selection import train_test_split
    
    print('Loading data...')
    with open('data/movies/embeddings/amazon_ordered_by_time{}.pickle'.format(''), 'rb') as handle:
        embs, key = pickle.load(handle)
        
    print('Restructuring data...')
    third_year = [x for x in list(zip(embs, key)) if key[0][-1] + timedelta(days=365*2) < x[1][-1] < key[0][-1] + timedelta(days=365*3)] # gather amazon reviews of the third year only
    third_year = [list(t) for t in zip(*third_year)]
    third_year[1] = [x[0]-1 for x in third_year[1]] # fix class names from 1..5 to 0..4 for easier 1-hot encoding
    
    X_train, X_test, y_train, y_test = train_test_split(third_year[0], third_year[1], test_size=0.33, random_state=42)
    s = ScorePredictor()
    
    # evaluate before training, just for sanity checks, should be garbage
    first_eval = s.evaluate(X_test, y_test)
    print('Accuracy before training:', first_eval[1])
    
    print('Starting training...')
    s.train(X_train, y_train)
    
    print('Training finished!')
    second_eval = s.evaluate(X_test, y_test)
    print('Accuracy after training:', second_eval[1])
    
    safe_path = 'classifier'
    os.system('mkdir -p ' + safe_path)
    print('Saving classifier as "{}"'.format(safe_path))
    s.safe(safe_path)