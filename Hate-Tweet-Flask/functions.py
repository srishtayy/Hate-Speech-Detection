
import pickle
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS




def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    for token in word_tokens:    
        if token.lower not in stop_words and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def preprocess_tweet(tweet):
 
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    processed = tokenize(result)
    return processed



def make_prediction(tweet):
    model = pickle.load(open("static/log_best.pickle", "rb"))
    processed = preprocess_tweet(tweet)
    lst = []
    lst.append(processed)
    vec = pickle.load(open("static/vec.pickle", "rb"))
    vectorized = vec.transform([str(item) for item in lst])
    pred = model.predict(vectorized)
    prob = model.predict_proba(vectorized)[:,1]
    mapping = {0: 'Non Hate', 1: 'Hate'}
    prediction = mapping[pred[0]]
    probability = str(prob)[1:-1]
    return tweet, prediction, probability