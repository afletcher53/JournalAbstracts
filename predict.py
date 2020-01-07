
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential, model_from_json
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


possible_labels = ["canine","feline"]
EMBEDDING_DIM = 50
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
with open(os.path.join('large_files/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM),  encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

data = {'Text' : ['toxoplasma FIV leukocell renal diesease']}
dataset_df = pd.DataFrame(data) 
dataset_df['Text'] = dataset_df['Text'].apply(clean_text)
dataset_df['Text'].apply(lambda x: len(x.split(' '))).sum()

sentences = dataset_df["Text"].fillna("DUMMY_VALUE").values

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



json_file = open('saved_models/SimpleRNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("saved_models/SimpleRNN.h5")
print("Loaded model from disk")

print(loaded_model.predict(data))

pred_name = possible_labels[np.argmax(loaded_model.predict(data))]
print(pred_name)