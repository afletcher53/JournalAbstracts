# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K  
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
from utils import clean_text, recall_m, precision_m, f1_m


LOG_DIR = f"{int(time.time())}"


# some configuration
MAX_SEQUENCE_LENGTH = 450
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 20
possible_labels = ["canine","feline"]



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../large_files/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM),  encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare text samples and their labels
print('Loading in texts...')


dataset_df = pd.read_csv("../data/data.csv")

dataset_df['Text'] = dataset_df['Text'].apply(clean_text)
dataset_df['Text'].apply(lambda x: len(x.split(' '))).sum()


sentences = dataset_df["Text"].fillna("DUMMY_VALUE").values
get_dummies = pd.get_dummies(dataset_df["species"])
targets = get_dummies[possible_labels].values


# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)


# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)


X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state = 42)


# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


print('Building model...')


def build_model():    
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    #x = LSTM(16, return_sequences=True)(x)
    x = Bidirectional(LSTM(15, return_sequences=True))(x)
    x= Dense(112,name='FC1')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(len(possible_labels), activation="sigmoid")(x)
    model = Model(input_, output)
   
    model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.032),
    metrics=['acc',f1_m,precision_m, recall_m])
    

    return model

model = build_model()


r = model.fit(
  X_train,
  y_train,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)


results = model.evaluate(X_test, y_test, batch_size=128)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='accuracy')
plt.plot(r.history['val_acc'], label='val_accuracy')
plt.legend()
plt.show()

p = model.predict(data)
aucs = []

for j in range(2):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

model_json = model.to_json()

with open("../saved_models/SimpleRNN.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("../saved_models/SimpleRNN.h5")
print("Saved model to disk")

