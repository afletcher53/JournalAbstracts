#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:40:53 2020

@author: aaron
"""


#!/usr/bin/env python

import pickle
from functools import partial

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#########################################

def USELayer(embed,x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

def make_model(embed, n_categories, latent_dim=16, embedding_dim=512):
    UniversalEmbedding = partial(USELayer,embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, output_shape=(embedding_dim, ))(text_in)

    x = layers.Dense(latent_dim, activation='relu')(x)

    x_out = layers.Dense(n_categories, activation='softmax')(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="AbstractClassifier")

#########################################

if __name__ == '__main__':

    LATENT_DIM = 16
    TEST_SIZE = 0.3
    N_EPOCHS = 20
    BATCH_SIZE = 128

    f_name = "../../data/species_pubmed.pkl"
    f = open(f_name, 'rb')
    df = pickle.load(f)

    categories = list(set(df['category_txt'].values))
    n_categories = len(categories)
    print("There are {} known categories: {}".format(n_categories, categories))

    X_txt = df['abstract'].values

    y = np.array(df['category_id'].values)
    y = to_categorical(y)

    X_txt_train, X_txt_test, y_train, y_test = train_test_split(X_txt, y)

    print("Training set has {} samples".format(X_txt_train.shape[0]))
    print("Testing set has {} samples".format(X_txt_test.shape[0]))

    # initialize USE embedder
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)

    model = make_model(embed, n_categories=n_categories, latent_dim=LATENT_DIM)
    #model = make_model_quantum(embed, n_categories=n_categories)

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss='categorical_crossentropy',
        optmizer=optimizer,
        metrics=['acc'],
    )

    print("Training...")

    callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.005)

    r = model.fit(
        X_txt_train, y_train,
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[],
    )
    print("Done training")

    print("Testing...")
    test_score = model.evaluate(X_txt_test, y_test, verbose=2)
    

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


model.save_weights("../../saved_models/USE.h5")
print("Saved model to disk")