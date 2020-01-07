# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:08:59 2019

@author: Aaron
"""
import pandas as pd
import glob
from bs4 import BeautifulSoup
import numpy as np
import wordcloud as wordcloud
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections
from collections import Counter

graph_directory="graphs/"
data_directory="data/"
species = ['Canine', 'Feline']
baseDir= "D:/aaron/MachineLearning/data/JournalAbstracts/"
col_list = ['Title', 'Abstract Text', 'Species']
final_col_list=['Text', 'Species']
stop_words = set(stopwords.words("english"))
custom_stopwords = ["canine", "canines", "bitch",  "dog", "felis", "canis", "dogs", "pup",  "puppies", "puppy", "feline",  "felines", "cats",  "cat", "queen", "domestic"]
stop_words = stop_words.union(custom_stopwords)
appended_data = []
#Instantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#the amount of data you want to see in Ngrams
n_print = 20
#array for species count
aniarray = []

def remove_html(text):
    soup = BeautifulSoup(text, features="lxml")
    html_free = soup.get_text()
    return html_free

def remove_punctuation(text):
    no_punct = "".join(c for c in text if c not in string.punctuation)
    return no_punct

def remove_stopwords(text):
    words = ' '.join([word for word in text.split() if word not in stop_words])
    return words

def remove_custom_stopwords(text):
    words = [w for w in text if w not in custom_stopwords]
    return words


for animal in species:

   files = glob.glob(baseDir+animal+'/*.txt')
   df = pd.concat([pd.read_csv(fp, sep='\t',header=(0)) for fp in files], ignore_index=True)
   
   #add in species category
   df['Species'] = animal
   
    #drop other columns except the ones we want
   df = df[col_list]
   

   for col in col_list:
   #drop any rows with empty data
       df[col].replace('', np.nan, inplace=True)
       df.dropna(subset=[col], inplace=True)
       
   #combine the Title and Abstract into one column
   df['Text'] = df['Title'] + ' ' + df['Abstract Text']
   
   # keep the finalcolumns
   df = df[final_col_list]
   
   #strip HTML if there       
   df.Text = df.Text.apply(lambda x: remove_html(x))
   
   #convert to lowercase
   df = df.astype(str).apply(lambda x: x.str.lower())
   
   # remove punctuation
   df.Text = df.Text.apply(lambda x: remove_punctuation(x))
   
   # remove stopwords
   df.Text = df.Text.apply(lambda x: remove_stopwords(x))
   
   # remove numbers
   df.Text= df.Text.str.replace('\d+', '')
   
   #add total counts
   aniarray.append(df.Text.count())
   
   #add to the list of dataframes     
   appended_data.append(df)

result = pd.concat(appended_data)
species_dummies = pd.get_dummies(result.Species)
result = pd.concat([result, species_dummies], axis=1)

#remove the species category as we have ONE it
result.pop('Species')

#save the file
result.to_csv(r'data.csv')

# Just making the plots look better

#def Plot_Total_Records():
#    y = aniarray[0], aniarray[1]
#    x = "Canine", "Feline"
#    plt.style.use('fivethirtyeight')
#    # Pad margins so that markers don't get clipped by the axes
#    plt.margins(0.2)
#    plt.subplots_adjust(bottom=0.15)
#    plt.title(" Total amount of records in data")
#    plt.bar(x,y)
#    plt.savefig(graph_directory+ 'totalrecords.png')
#
#Plot_Total_Records()

#save a version which isn't species agnositic
for animal in species:

   files = glob.glob(baseDir+animal+'/*.txt')
   df = pd.concat([pd.read_csv(fp, sep='\t',header=(0)) for fp in files], ignore_index=True)
   
   #add in species category
   df['Species'] = animal
   
    #drop other columns except the ones we want
   df = df[col_list]
   

   for col in col_list:
   #drop any rows with empty data
       df[col].replace('', np.nan, inplace=True)
       df.dropna(subset=[col], inplace=True)
       
   #combine the Title and Abstract into one column
   df['Text'] = df['Title'] + ' ' + df['Abstract Text']
   
   # keep the finalcolumns
   df = df[final_col_list]
   
   #strip HTML if there       
   df.Text = df.Text.apply(lambda x: remove_html(x))
   
   #convert to lowercase
   df = df.astype(str).apply(lambda x: x.str.lower())
   
   # remove punctuation
   df.Text = df.Text.apply(lambda x: remove_punctuation(x))
   
   # remove stopwords
   df.Text = df.Text.apply(lambda x: remove_stopwords(x))
   
   # remove numbers
   df.Text= df.Text.str.replace('\d+', '')
   
   #add to the list of dataframes     
   appended_data.append(df)

result = pd.concat(appended_data)
species_dummies = pd.get_dummies(result.Species)
result = pd.concat([result, species_dummies], axis=1)

#remove the species category as we have ONE it
result.pop('Species')

#save the file
result.to_csv(r'data_not_agnostic.csv')


import nltk
def Word_Counter(text):
    wordcount = {}
    for word in text.split():
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
    return wordcount

def Word_Frequency_Graph_and_Text(text, catdog):
    word_list = Word_Counter(text) 
    plt.style.use('fivethirtyeight')
    word_counter = collections.Counter(word_list)
    for word, count in word_counter.most_common(n_print):
        print(word, ": ", count) 
    plt.bar(*zip(*word_counter.most_common(20)))
    plt.xticks(rotation=90)
    plt.title(" most common unigram for " + catdog + " data")
    plt.savefig(graph_directory+ catdog+'unigram.png')
    plt.show()


def Plot_Bigram(text, n, catdog):
     
    words = nltk.word_tokenize(text)
    gram = nltk.bigrams(words)
   
    fdist = nltk.FreqDist(gram)
    
    #print the top 20 most frequent bigrams into a bar chart
    chart = fdist.most_common(n)
   
    labels, ys = zip(*chart)
    xs = np.arange(len(labels))     
    plt.style.use('fivethirtyeight')
    plt.bar(xs, ys, align='center')
    plt.xticks(xs, labels, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.ylim(auto=True)
    plt.title(str(n) + " most common bigrams for " + catdog + " data")
    plt.savefig(graph_directory+ catdog+'bigram.png')
    plt.show()
    
    
def Plot_Trigram(text, n, catdog): 
    words = nltk.word_tokenize(text)
    gram = nltk.trigrams(words)
   
    fdist = nltk.FreqDist(gram)
    
    #print the top 20 most frequent bigrams into a bar chart
    chart = fdist.most_common(n)
   
    labels, ys = zip(*chart)
    xs = np.arange(len(labels))     
    plt.style.use('fivethirtyeight')
    plt.bar(xs, ys, align='center')
    plt.xticks(xs, labels, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.ylim(auto=True)
    plt.title(str(n) + " most common trigrams for " + catdog + " data")
    plt.savefig(graph_directory+ catdog+'trigram.png')
    plt.show()
    


def Plot_WordCloud(text, catdog):
    print("\nOK. The wordcloud for \n".format(catdog))
    #Create wordclouds
    wordcloud = WordCloud(background_color="black", width=1600, height=800).generate(text)
     # Display the generated image:
    # the matplotlib way:

    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file(graph_directory+ catdog+".png")
    


## Lets have a look at our species specific data
for animal in species:
    
    catdog = str(animal).lower()
    #find only the ones that match
    results = result.loc[result[catdog] == 1]
    
    #save the amount in the dataset to a variable
    
    text = " ".join(catdog for catdog in results.Text)
    Word_Frequency_Graph_and_Text(text, catdog)
    
    
    Plot_Bigram(text, 20, catdog)
    Plot_Trigram(text, 20, catdog)
    Plot_WordCloud(text, catdog)
    
del text, wordcloud, catdog

    
