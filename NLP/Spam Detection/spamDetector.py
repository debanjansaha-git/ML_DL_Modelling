# -*- coding: utf-8 -*-
"""
Date: Oct 29 16:12:23 2021
@description: This module performs a fake news classification using Naive Bayes algorithm
              The dataset is taken from Kaggle and contains either "ham" or "spam" for each news
@author :     Debanjan Saha
@licence:     MIT License
"""
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

messages = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'message'])
# using Porter Stemmer for stemming words
ps = PorterStemmer()
corpus = []
# Data Cleaning and Pre-Processing
for i in range(len(messages)):
    # remove anything other than alphabets and convert all to lower case
    pre_data = re.sub("[^a-zA-Z]"," ", messages['message'][i])
    pre_data = pre_data.lower()
    pre_data = pre_data.split()
    # stem each words remove stop words and create corpus
    pre_data = [ps.stem(word) for word in pre_data if not word in stopwords.words('english')]
    pre_data = ' '.join(pre_data)
    corpus.append(pre_data)
# use Count Vectorizer
vect = CountVectorizer(max_features=2500)
# transform x and y
x = vect.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['label'], drop_first=True)
# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=365)
# Naive Bayes model
spam_model = MultinomialNB().fit(x_train, y_train)
# predictions
y_pred = spam_model.predict(x_test)
# confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_mat)
# accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Model: {round(acc*100, 4)}%")
# model predicted accuracy of 98.8341%