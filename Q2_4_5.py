#!/usr/bin/env python
# coding: utf-8

# In[90]:


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


# #### Training Dataset

# In[91]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[94]:


vect = CountVectorizer().fit(X_train["Text"])
trans = vect.transform(X_train["Text"])
model = MultinomialNB().fit(trans,y_train["Target"])
y_pred_train = model.predict(trans)


# In[96]:


metrics.accuracy_score(y_train, y_pred_train)


# #### Validation dataset

# In[99]:


validation = CountVectorizer().fit(X_test["Text"])
trans_val = validation.transform(X_test["Text"])
model = MultinomialNB().fit(trans_val,y_test["Target"])
y_pred_test = model.predict(trans_val)


# In[100]:


metrics.accuracy_score(y_test, y_pred_test)


# #### Misclasification of Training set

# In[140]:


y_pred_train_np = np.array(y_pred_train)
y_train_np = np.array(y_train)
mis_x_train = y_pred_train_np != (y_train_np.reshape(-1,))
X_train_np = np.array(X_train[mis_x_train])
x=X_train_np.reshape(-1,)
y=y_train_np[mis_x_train].reshape(-1,)
z=y_pred_train_np[mis_x_train].reshape(-1,)
w = mis_x_train[mis_x_train].reshape(-1,)
mis_train_df = pd.DataFrame({"Text":x,"y_train":y,"y_predict":z,"Misclassification":w})
mis_train_df


# #### Misclassification of Vlidation set

# In[142]:


y_pred_test_np = np.array(y_pred_test)
y_test_np = np.array(y_test)
mis_x_test = y_pred_test_np != (y_test_np.reshape(-1,))
X_test_np = np.array(X_test[mis_x_test])
x=X_test_np.reshape(-1,)
y=y_test_np[mis_x_test].reshape(-1,)
z=y_pred_test_np[mis_x_test].reshape(-1,)
w = mis_x_test[mis_x_test].reshape(-1,)
miss_df = pd.DataFrame({"Text":x,"y_test":y,"y_predict":z,"Misclassification":w})
miss_df


# In[ ]:




