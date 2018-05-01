import json
import os
import re
from pyvi import ViTokenizer




def transforming_tokens(text):
    with open("stop_words.txt", "r") as f:
        stop_words = f.read()
    f.close()
    stop_words = stop_words.split()
    text = ViTokenizer.tokenize(text.lower()).split()
    for word in stop_words:
        word = word.replace(" ", "_")
        if word in text:
            text.remove(word)
    return text


