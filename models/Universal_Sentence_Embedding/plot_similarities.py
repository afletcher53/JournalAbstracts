#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:37:59 2020

@author: aaron
"""


#!/usr/bin/env python

import pickle

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

#########################################

def calc_similarities(X, Y):
    sims = cosine_similarity(X,Y)
    sims = sims.flatten()
    sims = sims[sims<1.]

    #sims = np.arccos(sims)/np.pi # radians
    #sims = 1. - sims

    return sims 

#########################################

if __name__ == '__main__':
    SUBSPACE_DIM = .20

    f_name = "../../data/species_pubmed.pkl"
    f = open(f_name, 'rb')
    df = pickle.load(f)

    print(df.sample(n=10))

    print("Total number of examples: {}".format(df.shape[0]))

    cat_txt = df['category_txt'].drop_duplicates().values
    print("Known categories:", cat_txt)

    cat_ids = df['category_id'].drop_duplicates().values
    n_categories = len(cat_ids)
    print("Number of categories: {}".format(n_categories))

    pca = PCA(n_components=SUBSPACE_DIM)
    X = np.array([emb for emb in df['embedding'].values])
    pca.fit(X)
    X = pca.transform(X)
    pca_dim = X.shape[1]

    X = [ list(X[i]) for i in range(X.shape[0])]
    df['pca'] = X
    print(df.head())

    fig, axes = plt.subplots(n_categories, n_categories, sharex=False, sharey=False, figsize=(12,10))

    for i in range(n_categories):
        for j in range(n_categories):
            if i > j:
                axes[i, j].axis('off')

    all_similarities = []
    y_max = 0.30
    for id1 in range(n_categories):
        X = df.loc[df.category_id==id1]
        X_512 = np.array([ x for x in X['embedding'].values ])
        X_pca = np.array([ x for x in X['pca'].values ])

        for id2 in range(id1, n_categories):

            Y = df.loc[df.category_id==id2]
            Y_512 = np.array([ y for y in Y['embedding'].values ])
            Y_pca = np.array([ y for y in Y['pca'].values ])

            sims_512 = calc_similarities(X_512, Y_512)
            sims_pca = calc_similarities(X_pca, Y_pca)
    
            axes[id1, id2].hist(sims_512, 40, histtype='stepfilled', weights=np.ones(len(sims_512)) / len(sims_512), label="512-dim")
            axes[id1, id2].hist(sims_pca, 40, histtype='stepfilled', alpha=0.7, weights=np.ones(len(sims_pca)) / len(sims_pca), color='red', label="PCA ({})".format(SUBSPACE_DIM))

            axes[id1, id2].legend(loc='upper left', prop={'size': 6})

            axes[id1, id2].set_xlim([-1., 1.])
            axes[id1, id2].set_ylim([0., y_max])

            #axes[id1, id2].set_ylabel("Fraction")

            axes[id1, id2].text(0.1, y_max-0.03, cat_txt[id1], fontsize=9)
            if id1 != id2:
                axes[id1, id2].text(0.1, y_max-0.06, cat_txt[id2], fontsize=9)

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace = 0.5)
    plt.suptitle("Universal Sentence Embeddings - Dimensionality Reduction 512 -> {}".format(pca_dim))

    plt.show()
    fig.savefig("../../graphs/species_pubmed.png")