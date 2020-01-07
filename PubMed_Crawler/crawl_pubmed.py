#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:30:22 2020

@author: aaron
"""
import pandas as pd
from utils import clean_text, convert_category, search, fetch_details, supported_species
import argparse
import warnings
import pickle
import tensorflow_hub as hub
import numpy as np


if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='PubmedCrawler for Species Abstracts and Texts')
    parser.add_argument('-s', action='append', dest='species_args', default=[], help='Specify species for crawler. Currently supported species are canine, feline and equine. EXPERIMENTAL - Can supply RAW queries to Enterez / Species also through this option')
    parser.add_argument('-res', action='store', dest='total_results',default=['100000'], help='Total Results Wanted')
    parser.add_argument('-output', action='store', dest='output_file',default=['pubmed_crawler'], help='Name of output file')
    parser.add_argument('-embedUSE', action='store_true', default=False, dest='embed_USE', help='Embed results with Universal Single Encoder')
    parser.add_argument('-embedELMO', action='store_true', default=False, dest='embed_ELMO', help='Embed results with ELMO')
    parser.add_argument('-speciesStopWords', action='store_true', default=True, dest='species_stopwords', help='Remove species as stopwords from pubmed')

    argresults = parser.parse_args()
    
    
    if not argresults.species_args:
        argresults.species_args=['canine', 'feline']
 

    categories = argresults.species_args
    map_categories = { i:c for c,i in enumerate(categories) }
    
    argresults.species_stopwords = False
    
    print("There are {} known categories: {}".format(len(categories), categories))
     
    df = pd.DataFrame(columns=["abstract", "category_txt", "category_id"])
    
    for category in categories:
        if category not in supported_species():
            warnings.warn('The {} species is not supported, cannot exclude cross species contamination'. format(category))
        print("Processing category {}...".format(category))
        c_id = map_categories[category]
        results = search(convert_category(category), argresults.total_results)
        id_list = results['IdList']
        papers = fetch_details(id_list)
        for i, paper in enumerate(papers['PubmedArticle']): 
            if 'Abstract' in paper['MedlineCitation']['Article']:
                abstract = ''.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText']) + ' ' + paper['MedlineCitation']['Article']['ArticleTitle']
                clean_abstract = clean_text(abstract, argresults.species_stopwords)
                article = {
                    'abstract':clean_abstract,
                    'category_txt':category,
                    'category_id':c_id,
                }
                df = df.append(article, ignore_index=True)
        
    print("Found {} articles".format(df.shape[0]))
    
    
    f_out = open(argresults.output_file[0] + ".pkl", 'wb')
    pickle.dump(df, f_out)
    f_out.close()
    
    #check for embeddings option
    
    if(argresults.embed_USE == True):
        
        print("Creating USE embeddings...")

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        embed = hub.load(module_url)
    
        X_txt = df.abstract.values
        X_embed = embed(X_txt)
        print(X_embed.shape)
        X_embed = [ np.array(emb) for emb in X_embed]
        df['embedding'] = X_embed
    
        print(df.sample(n=10))
    
        f_out = open(argresults.output_file[0] + "USE.pkl", 'wb')
        pickle.dump(df, f_out)
        f_out.close()

    if(argresults.embed_ELMO == True):
        
        print("Creating ELMO embeddings...")

        module_url = "https://tfhub.dev/google/elmo/3"
        embed = hub.load(module_url)
    
        X_txt = df.abstract.values
        X_embed = embed(X_txt)
        print(X_embed.shape)
        X_embed = [ np.array(emb) for emb in X_embed]
        df['embedding'] = X_embed
    
        print(df.sample(n=10))
    
        f_out = open(argresults.output_file[0] + "USE.pkl", 'wb')
        pickle.dump(df, f_out)
        f_out.close()


