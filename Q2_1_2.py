#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from string import digits 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import nltk
file = open("yelp_labelled.txt","r").read()
x = file.split('\n')
X = []
Y = []
for i in x:
    if i == '':
        continue
    a = i.split('\t')
    X.append(a[0])
    Y.append(a[1])
X= np.array(X)

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stop_words = set(stopwords.words('english'))
for i in range(X.shape[0]):
    X[i] = X[i].lower()
    string = ""
    for j in range(len(X[i])):
        if X[i][j] in punctuations:
            continue
        string += X[i][j]
    X[i] = string
X_data = []
for i in range(X.shape[0]):
    text_token = word_tokenize(X[i])
    text_token_wsw = [word for word in text_token if not word in stopwords.words()]
    X_data.append(text_token_wsw)
x=X_data
y = Y
df = pd.DataFrame({"Text" : x, "Target" : y})
df
X = df['Text']
Y = df['Target']
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
X_train.shape,X_test.shape,y_train.shape,y_test.shape,


