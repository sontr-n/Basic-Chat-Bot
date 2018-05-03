#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pprint
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from pyvi import ViTokenizer
import preprocessing

"""
X is a dimention of messages
Y is a dimention of labels

"""

class Classify(object):

    def __init__(self):
        # self.bow = {}
        self.nb = MultinomialNB()
        #self.nb = BernoulliNB()



    def read_data(self, file):
        # self.bow = json.load(open("tmp/bag_of_words.json"));
        # self.len_vector = len(self.bow)
        f_data = json.load(open(file))

        #Modifing all upper character to lower
        label = np.array([d["category"].lower() for d in f_data])
        ques = [d["ques"].lower() for d in f_data]
        #transforming messages to a vectors
        mess = [preprocessing.transforming_tokens(q) for q in ques]

        mess = np.array(mess)
        return mess, label


    def training(self):
        x, y = self.read_data("data.json")
        self.nb.fit(x, y)




    def preprocessing_input(self, text):
        input = [0]*self.len_vector
        text = ViTokenizer.tokenize(text)
        for key in self.bow.keys():
            if key in text:
                input[self.bow[key]] += 1

        input = np.array([input])
        return input

    def predict(self, text):
        input = self.preprocessing_input(text)
        print(self.nb.predict_proba(input))
        return self.nb.predict(input)[0]
