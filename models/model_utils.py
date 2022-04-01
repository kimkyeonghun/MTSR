import os
import json
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(model_url)

DATA_PATH = '../data/'

def embed_text(input):
    return tf.expand_dims(model(input), 0)

def load_price():
    pass

def load_text():
    TEXT_PATH = os.path.join(DATA_PATH, 'tweet','preprocessed')
    text_dataset= defaultdict(dict)
    for stock in os.listdir(TEXT_PATH):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)
        max_text_len = 0
        text_dataset[stock] = defaultdict(list)
        for text_file in text_list:
            data = [" ".join(json.loads(line)['text']) for line in open(os.path.join(stock_path, text_file), 'r')]
            embedded_data = model(data)
            text_dataset[stock][text_file].append(embedded_data)
            max_text_len = max(max_text_len,embedded_data.shape[0])
        print(max_text_len)
        #print(text_dataset)


        #text_dataset.append(text_day)            
    #print(np.array(text_dataset).shape)
    return 

def load_graph():
    pass

def load_dataset():
    load_price()
    text_data = load_text()
    load_graph()
    return text_data

print(load_dataset())