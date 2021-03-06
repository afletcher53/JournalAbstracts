import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from Bio import Entrez
import tensorflow as tf
import tensorflow.keras.backend as K  

custom_stops = ["canine", "canines", "bitch",  "dog", "felis", "canis", "dogs", "pup",  "puppies", "puppy", "feline",  "felines", "cats",  "cat", "queen", "domestic"]
stops = set(stopwords.words("english"))

# Removes meaningless symbols, and replace api placeholder symbols
def clean_text(raw):
    
    
    # Remove HTML
    review_text = BeautifulSoup(raw,features="html.parser").get_text()
    review_text = raw
    
    # Remove non-letters
    review_text = re.sub(r'http\S+', '', review_text)
    review_text = re.sub(r'â€¦', '', review_text)
    review_text = re.sub(r'…', '', review_text)
    review_text = re.sub(r'â€™', "'", review_text)
    review_text = re.sub(r'â€˜', "'", review_text)
    review_text = re.sub(r'\$q\$', "'", review_text)
    review_text = re.sub(r'&amp;', "and", review_text)
    review_text = re.sub('[/(){}\[\]\|@,;]', "", review_text)
    review_text = re.sub('[^0-9a-z #+_]', "", review_text)
    review_text = re.sub('[^A-Za-z0-9#@ ]+', "", review_text)
    review_text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", review_text)
    # review_text = re.sub(r'RT ', '', review_text)
    # review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # review_text = re.sub(r' q ', ' ', review_text)

    # Convert to lower case, split into individual words
    words = review_text.lower().split()
    #words = review_text.split()

    #remove_stopwords:
    
    words = [w for w in words if not w in stops]
    words = [w for w in words if not w in custom_stops]


    
    #for w in words:
        #w = wordnet_lemmatizer.lemmatize(w)
        # very slow
        # w = spell(w)

    return( " ".join(words))


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def convert_category(raw):
    
    if raw == "canine":
        return "(canine[MeSH Terms]) NOT (feline[MeSH Terms] OR equine[MeSH Terms] OR bovine[MeSH Terms] OR porcine[MeSH Terms])"
    if raw == "feline":
        return "(feline[MeSH Terms]) NOT (canine[MeSH Terms] OR equine[MeSH Terms] OR bovine[MeSH Terms] OR porcine[MeSH Terms])"
    else:
        return raw



def search(query, total_results):
    Entrez.email = 'afletcher53@gmail.com'
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax=total_results,
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'afletcher53@gmail.com'
    handle = Entrez.efetch(db='pubmed',
                            retmode='xml',
                            id=ids)
    results = Entrez.read(handle)
    return results

