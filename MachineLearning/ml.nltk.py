import sys
import time

import numpy as numpy
#import matplotlib.pyplot as plt
import scipy as scipy
import pandas as pandas

import string

import nltk

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from collections import Counter

import re
 
def plot_corrs(df_training, featureSize):
    corr = df_training.corr()
    fig, ax = plt.subplots(figsize=(featureSize,featureSize))

    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

def show_metrices(model, predicted_result, actual_result, file):
    file.write("accuracy_score : {0} \n {1} \n".format(model, metrics.accuracy_score(actual_result, predicted_result)))
    file.write("confusion_matrix : {0} \n {1} \n".format(model, metrics.confusion_matrix(actual_result, predicted_result)))
    file.write("classification_report : {0} \n {1} \n".format(model, metrics.classification_report(actual_result, predicted_result)))
    return

def get_cleanedup_data(stopwords, text):
   
    text = get_readable_text(text)

    print("raw : {0}".format(text))
    
    stemmer = nltk.PorterStemmer()
    stemmed_data = [stemmer.stem(w) for w in nltk.word_tokenize(text)]
    print("stemmed_data : {0}".format(' '.join(stemmed_data) ))

    #cleaned_data = [w for w in stemmed_data if w not in stopwords]
    #print("cleaned_data : {0}".format(' '.join(cleaned_data) ))

    return ' '.join(stemmed_data)

def get_readable_text(text):    
    text = text.lower()
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = text.strip()
    return text

def get_low_freq_words(data, threshold):
    words = []
    for text in data:
        text = get_readable_text(text)
        print("raw : {0}".format(text))
        for word in text.split():
            words.append(word)
        
    print("corpus {0}".format(words))

    wrd_counter = Counter(words)
    print(wrd_counter.most_common())
    low_freq_wrds = [item[0] for item in wrd_counter.items() if item[1] < threshold]
    print(low_freq_wrds) 

    return low_freq_wrds

def print_predictions(predictions, realdata, file):
    idx = 0
    file.write(">>>>>> predicted result for cyber_bullying --- start\n")
    for item in predictions:
        file.write("'{0}' : '{1}' \n".format(realdata[idx], bool(item)))
        idx = idx + 1
    
    file.write("<<<<<< predicted result for cyber_bullying --- end \n")

    return

def main():

    stemmer = nltk.PorterStemmer()
    #stopwords = set([stemmer.stem(w.lower()) for w in nltk.corpus.stopwords.words('english')])
    stopwords = set(nltk.corpus.stopwords.words('english'))

    punctuations = set(string.punctuation)
    stopwords.update(punctuations)
    print("stopwords len \n {0}".format(len(stopwords)))
    print("stopwords  \n {0}".format(stopwords))

    must_retain_words = pandas.read_csv("./alert-data/must_retain_words")
    
    must_retain_words = set([stemmer.stem(w.lower()) for w in must_retain_words['words']])
    print(must_retain_words)

    
    #1. read the data using pandas lib
    #df_train_tn = pandas.read_csv("./alert-data/alert-training-dataset-1-raw.csv")
    #df_train_tp = pandas.read_csv("./alert-data/alert-training-dataset-1-raw.csv")
    #print(df_train_tn.shape)
    #print(df_train_tp.shape)
    
    df_training = pandas.read_csv("./alert-data/alert-training-dataset-1-raw.csv")
    print(df_training.shape)

    df_training = df_training.drop_duplicates(keep='first')
    print(df_training.shape)

    print(len(df_training.loc[df_training['cyber_bullying'] == 1]))
    print(len(df_training.loc[df_training['cyber_bullying'] == 0]))
    
    print(df_training.head(df_training.shape[0]))

    df_actual = pandas.read_csv("./alert-data/alert-data-for-prediction.csv")

    low_freq_words = set(get_low_freq_words(df_training["search_text"], 2))    
    print("low_freq_words len \n {0}".format(len(low_freq_words)))

    low_freq_words = set([stemmer.stem(w.lower()) for w in low_freq_words])
    print(low_freq_words)

    print("stopwords len \n {0}".format(len(stopwords)))
    #stopwords = stopwords | low_freq_words
    print("stopwords len \n {0}".format(len(stopwords)))
    
    #stopwords = stopwords - must_retain_words
    print("stopwords len \n {0}".format(len(low_freq_words)))
    print("stopwords \n {0}".format(stopwords))

    features = [get_cleanedup_data(stopwords, str(text)) for text in df_training["search_text"]]

    y = df_training['cyber_bullying'].values
    X = pandas.DataFrame(features, columns=['search_text'])['search_text']
    #X = df_training["search_text"]
    print(X)

    #print("features - \n {0}".format(X))
    #print("labels - \n {0}".format(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42 )
    
    #using MultinomialNB
    pipeline = Pipeline([('bow',CountVectorizer(analyzer='word', stop_words=stopwords)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])

    pipeline.fit(X_train, y_train)

    result = pipeline.predict(X_test)

    f = open("MultinomialNB - {0}.txt".format(time.time()), mode='wt', encoding='utf-8')
    show_metrices("MultinomialNB", result, y_test, f)

    realdata = [get_cleanedup_data(stopwords, str(text)) for text in df_actual["search_text"]]
#    z = pandas.DataFrame(realdata, columns=['search_text'])['search_text']
    predictions = pipeline.predict(realdata)
    print_predictions(predictions, df_actual["search_text"], f)
    f.close()
    
    # using logistic regression
    C_start = 0.1
    C_End = 5.0
    C_inc = 0.1

    C_Values, recall_scores = [], []
     
    best_recall_score = 0

    C_val = C_start
    while(C_val < C_End):
        C_Values.append(C_val)

        pipeline = Pipeline([('bow',CountVectorizer(analyzer='word', stop_words=stopwords)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',LogisticRegression(C=C_val, class_weight="balanced", random_state=42))])

        pipeline.fit(X_train, y_train)
        pred_result = pipeline.predict(X_test)

        rscore = metrics.recall_score(y_test, pred_result)
        recall_scores.append(rscore)

        if(rscore > best_recall_score):
            best_recall_score = rscore
            #print("C_val : {0}".format(C_val))

        C_val = C_val + C_inc
    
    #plt.plot(C_Values, recall_scores, "-")
    #plt.xlabel("C label")
    #plt.ylabel("recall score")
    #plt.show()

    best_C_val = C_Values[recall_scores.index(best_recall_score)]

    pipeline = Pipeline([('bow',CountVectorizer(analyzer='word', stop_words=stopwords)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',LogisticRegression(C=best_C_val, class_weight="balanced", random_state=42))])

    pipeline.fit(X_train, y_train)
    result = pipeline.predict(X_test)

    f = open("LogisticRegression - {0}.txt".format(time.time()), mode='wt', encoding='utf-8')
    show_metrices("LogisticRegression", result, y_test, f)
    
    predictions = pipeline.predict(realdata)
    print_predictions(predictions, df_actual["search_text"], f)
    f.close()

    #using cross validation
    pipeline = Pipeline([('bow',CountVectorizer(analyzer='word', stop_words=stopwords)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',LogisticRegressionCV(n_jobs=1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced"))])

    pipeline.fit(X_train, y_train)
    result = pipeline.predict(X_test)

    f = open("LogisticRegressionCV - {0}.txt".format(time.time()), mode='wt', encoding='utf-8')
    show_metrices("LogisticRegressionCV", result, y_test, f)

    predictions = pipeline.predict(realdata)
    print_predictions(predictions, df_actual["search_text"], f)
    f.close()

    return
     
if __name__ == "__main__":
    main()
