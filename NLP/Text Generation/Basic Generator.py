# -*- coding: utf-8 -*-
"""
Created on Tue October 29 05:46:48 2021
@description: This module creates randomly generated text using Markovify
@author: Debanjan
"""

import markovify
import nltk
import re
import json

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = [ "::".join(tag) for tag in nltk.pos_tag(words) ]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

with open('data/data.txt', encoding='utf-8') as f:
    text = f.read()

text_model = markovify.Text(text, state_size=4)

# Export Model
model_json = text_model.to_json()
with open('models/json_model.json', 'w', encoding='utf-8') as f:
    json.dump(model_json, f, ensure_ascii=False, indent=4)

# Read Saved Model
with open('models/json_model.json') as json_file:
    read_model = json.load(json_file)
reconstituted_model = markovify.Text.from_json(read_model)

count=0
# Print five randomly-generated sentences
while count<5:
    new_sentence = reconstituted_model.make_sentence()
    print(new_sentence)
    count += 1