#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from string import digits 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
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
    value = TreebankWordDetokenizer().detokenize(text_token_wsw)
    X_data.append(value)

x=X_data
y = Y
df = pd.DataFrame({"Text" : x, "Target" : y})
print(df)
X = df['Text']
Y = df['Target']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,shuffle=False)
X_train.shape,X_test.shape,y_train.shape,y_test.shape,

unique = set(X_train.str.replace('[^a-zA-Z ]', '').str.split(' ').sum())
voc_train=list(sorted(unique))
uni = set(X_test.str.replace('[^a-zA-Z ]', '').str.split(' ').sum())
voc_test=list(sorted(uni))

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

X_train = X_train.reindex(columns=[*X_train.columns.tolist(), *voc_train], fill_value=0 )
X_test =X_test.reindex(columns=[*X_test.columns.tolist(), *voc_test], fill_value=0 )

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_test.reset_index(inplace = True)

def feature_mat(sample_train,voc):
    for i in range(0,len(sample_train)):
        sent=sample_train['Text'].iloc[i]
        word=nltk.word_tokenize(sent)
        for j in range(0,len(word)):
            if word[j] in voc:
                sample_train.at[i, word[j]] = sample_train.at[i, word[j]] + 1
    return sample_train

X_train = feature_mat(X_train,voc_train)
X_test = feature_mat(X_test,voc_test)

X_train

X_test



