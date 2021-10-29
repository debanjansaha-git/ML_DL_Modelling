# -*- coding: utf-8 -*-
"""
Created on Tue October 29 06:58:16 2021
@description: This module summarizes text from from any url
@author: Debanjan
"""
import bs4 as bs
import urllib.request
import re
import nltk
import heapq
import json
from datetime import datetime

# nltk.download('stopwords')
# nltk.download('punkt')

# Parameters
URI = "https://en.wikipedia.org/wiki/The_Avengers_(2012_film)"
max_lines = 5

# Fetch data from URI
data = urllib.request.urlopen(URI).read()
soup = bs.BeautifulSoup(data,'lxml')
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text
    
# Preprocessing the text
text = re.sub(r'\[[0-9]*\]',' ',text)    
text = re.sub(r'\s+',' ',text)    
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)

sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('english')

# Building the Histogram
word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word]=1
        else:
            word2count[word]+=1

# Weighted Histogram
for key in word2count.keys():
    word2count[key]=word2count[key]/max(word2count.values())
  
# Calculate the score
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' '))<30:
                if sentence not in sent2score.keys():
                     sent2score[sentence]=word2count[word]
                else:
                    sent2score[sentence]+=word2count[word]
    
# Top n Sentences
best_sentences = heapq.nlargest(max_lines,sent2score,key=sent2score.get)                    
sentence_dict = {'Line' + str(k+1): v for k, v in enumerate(best_sentences)}

# Save summarization as JSON object in output folder
json_object = json.dumps(sentence_dict, indent=2, separators=(',', ':'))
filename = 'summary-' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'
with open('output/' + filename, 'w') as outfile:
    outfile.write(json_object)
            