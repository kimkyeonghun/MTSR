from collections import defaultdict
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import torch

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(model_url)

DATA_PATH = './data/'

def embed_text(input):
    return tf.expand_dims(model(input), 0)

def load_price(logger):
    PRICE_PATH = os.path.join(DATA_PATH, 'price', 'raw')
    price_dataset = defaultdict(dict)
    logger.info("#Load Price")
    for stock in tqdm(os.listdir(PRICE_PATH),desc="Stock Iteration"):
        if stock=='GMRE.csv':
            logger.info("Price data has more stock than tweet, so exclude GMRE")
            continue
        prices = pd.read_csv(os.path.join(PRICE_PATH, stock))
        temp_prices = copy.deepcopy(prices)
        prices['y'] = -1
        for i in range(1, len(prices)):
            day = prices['Date'].iloc[i]
            prices['High'][i] /= temp_prices['Adj Close'][i-1]
            prices['Low'][i] /= temp_prices['Adj Close'][i-1]
            prices['Adj Close'][i] /= temp_prices['Adj Close'][i-1]
            prices['y'][i] = prices['Adj Close'][i]>=1
            price_dataset[stock][day] = ((prices['Adj Close'][i], prices['High'][i], prices['Low'][i]), prices['y'][i])

    return price_dataset

def load_text(logger):
    TEXT_PATH = os.path.join(DATA_PATH, 'tweet','preprocessed')
    
    logger.info("#Universal Sentence Encoder Start")
    text_dataset= defaultdict(dict)
    for stock in tqdm(os.listdir(TEXT_PATH), desc='Stock Iteration'):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)
        for day in tqdm(text_list, desc="Day Iteration"):
            data = [" ".join(json.loads(line)['text']) for line in open(os.path.join(stock_path, day), 'r')]
            data =list(dict.fromkeys(data))
            embedded_data = model(data)
            text_dataset[stock][day] = embedded_data

    train_range = pd.date_range(start='2014-01-01', end='2015-07-31', freq='1D')
    val_range = pd.date_range(start='2015-08-01', end='2015-09-30', freq='1D')
    test_range = pd.date_range(start='2015-10-01', end='2016-01-01', freq='1D')

    train_dataset = dict()
    val_dataset = dict()
    test_dataset = dict()

    logger.info("#Train Range Start")
    
    for date in train_range:
        for stock in os.listdir(TEXT_PATH):
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                if text_dataset[stock][key].shape[0] <= 5:
                    zero_padding = tf.zeros([5-text_dataset[stock][key].shape[0], 512], dtype=tf.float32)
                    train_dataset[stock][key] = tf.concat([text_dataset[stock][key], zero_padding], 0)
                else:
                    train_dataset[stock][key] = text_dataset[stock][key][:5]
            else: 
                zero_padding = tf.zeros([5, 512], dtype=tf.float32)
                train_dataset[stock][key] = zero_padding

    logger.info("#Validation Range Start")
    for date in val_range:
        for stock in os.listdir(TEXT_PATH):
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                if text_dataset[stock][key].shape[0] <= 5:
                    zero_padding = tf.zeros([5-text_dataset[stock][key].shape[0], 512], dtype=tf.float32)
                    val_dataset[stock][key] = tf.concat([text_dataset[stock][key], zero_padding], 0)
                else:
                    val_dataset[stock][key] = text_dataset[stock][key][:5]
            else: 
                zero_padding = tf.zeros([5, 512], dtype=tf.float32)
                val_dataset[stock][key] = zero_padding

    logger.info("#Test Range Start")
    for date in test_range:
        for stock in os.listdir(TEXT_PATH):
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                if text_dataset[stock][key].shape[0] <= 5:
                    zero_padding = tf.zeros([5-text_dataset[stock][key].shape[0], 512], dtype=tf.float32)
                    test_dataset[stock][key] = tf.concat([text_dataset[stock][key], zero_padding], 0)
                else:
                    test_dataset[stock][key] = text_dataset[stock][key][:5]
            else: 
                zero_padding = tf.zeros([5, 512], dtype=tf.float32)
                test_dataset[stock][key] = zero_padding               
            
    return train_dataset, val_dataset, test_dataset

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def load_graph(market_name, logger):
    GRAPH_PATH = os.path.join(DATA_PATH, 'relation')
    adj = np.load(os.path.join(GRAPH_PATH, market_name+'_graph.npy'))
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def load_dataset(args, logger):
    price_dataset = load_price(logger)
    train_text, val_text, test_text = load_text(logger)
    graph_net = load_graph(args.market_name, logger)
    return train_text, val_text, test_text
