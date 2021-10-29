# -*- coding: utf-8 -*-
"""
Date: Oct 29 16:12:23 2021
@description: This module performs Sentiment Analysis using Random Forest Classifier algorithm
              The dataset is taken from Kaggle and contains various reactions from 2000-2016
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('data/Data.csv', encoding="ISO-8859-1")

train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']
# select ROI
train_df = train.iloc[:,2:27]
test_df = test.iloc[:,2:27]


def data_cleaner(df):
    # replace all special characters except alphabets
    df.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    # Renaming cols to 1 to 25
    list1= [i for i in range(1, 26)]
    new_Index=[str(i) for i in list1]
    df.columns= new_Index
    # convert all data to lower case
    df = df.astype(str).apply(lambda x: x.str.lower())
    ' '.join(str(x) for x in df.iloc[1, 0:25])
    toList = []
    for row in range(len(df.index)):
        toList.append(' '.join(str(x) for x in df.iloc[row, 0:25]))
    
    return toList

headlines = data_cleaner(train_df)
test_set = data_cleaner(test_df)

vect = CountVectorizer(ngram_range=(2,2))
train_ds = vect.fit_transform(headlines)

# MODELLING
RFClass = RandomForestClassifier(n_estimators=200, criterion='entropy')
RFClass.fit(train_ds,train['Label'])

test_ds = vect.transform(test_set)
predictions = RFClass.predict(test_ds)

# METRICS
matrix=confusion_matrix(test['Label'],predictions)
print("\nConfusion Matrix: \n", matrix)

print("\nScore: ", RFClass.score)

precision=(matrix[0][0]/(matrix[0][0]+matrix[1][0]))*100
print("\n Precision: ", precision)  ## 94.5945945945946

accuracy=(matrix[0][0]+matrix[1][1])/((matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1]))*100
print("\n Accuracy: ", accuracy)    ## 85.71428571428571