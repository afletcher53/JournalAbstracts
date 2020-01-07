#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:30:22 2020

@author: aaron
"""
import sys
import pandas as pd
from utils import clean_text, convert_category, search, fetch_details
import tensorflow_hub as hub
import numpy as np
import pickle

categories = ['canine', 'feline']
map_categories = { i:c for c,i in enumerate(categories) }

total_results = 4000

print("There are {} known categories: {}".format(len(categories), categories))


df = pd.DataFrame(columns=["abstract", "category_txt", "category_id"])

for category in categories:
    print("Processing category {}...".format(category))
    c_id = map_categories[category]
       
    # issue search with cat
    results = search(convert_category(category), total_results)
    id_list = results['IdList']
    papers = fetch_details(id_list)
    for i, paper in enumerate(papers['PubmedArticle']): 
        if 'Abstract' in paper['MedlineCitation']['Article']:
            abstract = ''.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText']) + paper['MedlineCitation']['Article']['ArticleTitle']
            clean_abstract = clean_text(abstract)
            article = {
                'abstract':clean_abstract,
                'category_txt':category,
                'category_id':c_id,
            }
            df = df.append(article, ignore_index=True)
    
print("Found {} articles".format(df.shape[0]))
print(df.sample(n=10))

   
print("Creating embeddings...")

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
embed = hub.load(elmo)

X_txt = df.abstract.values
X_embed = embed(X_txt)
print(X_embed.shape)
X_embed = [ np.array(emb) for emb in X_embed]
df['embedding'] = X_embed

print(df.sample(n=10))

f_out = open("../../data/species_pubmed.pkl", 'wb')
pickle.dump(df, f_out)
f_out.close()
     