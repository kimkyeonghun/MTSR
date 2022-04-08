from collections import defaultdict, deque
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
#not use tensorflow_text, but need because of USE module
import tensorflow_text
import tensorflow_hub as hub
import torch

from MTSRDataset import MTSRDataset

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(model_url)

train_range = pd.date_range(start='2014-01-01', end='2015-07-31', freq='1D')
val_range = pd.date_range(start='2015-08-01', end='2015-09-30', freq='1D')
test_range = pd.date_range(start='2015-10-01', end='2016-01-01', freq='1D')

DATA_PATH = './data/'
PRICE_PATH = os.path.join(DATA_PATH, 'price', 'raw')
TEXT_PATH = os.path.join(DATA_PATH, 'tweet','preprocessed')
GRAPH_PATH = os.path.join(DATA_PATH, 'relation')


def gen_price_dataset(price_dataset, date_range):
    pdataset = []
    target = []
    for stock in price_dataset.keys():
        stock_day = []
        ydata = []
        time_dur = deque()
        for date in date_range:
            date = date.strftime('%Y-%m-%d')
            if date in price_dataset[stock]:
                price_feature, y = price_dataset[stock][date]
            else:
                price_feature, y = (0,0,0), 0
            if len(time_dur)!=5:
                time_dur.append(price_feature)
                if len(time_dur)==5:
                    stock_day.append(list(time_dur))
                    ydata.append(y)
            else:
                time_dur.popleft()
                time_dur.append(price_feature)
                stock_day.append(list(time_dur))
                ydata.append(y)
        pdataset.append(stock_day)
        target.append(ydata)
    
    #need to shift
    target = np.array(target)
    pdataset = np.array(pdataset)
    return pdataset, target

def gen_text_dataset(text_dataset, date_range):
    dataset = []
    for stock in os.listdir(TEXT_PATH):
        stock_day = []
        time_dur = deque()
        for date in date_range:
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                if text_dataset[stock][key].shape[0] <= 5:
                    zero_padding = tf.zeros([5-text_dataset[stock][key].shape[0], 512], dtype=tf.float32)
                    data = tf.concat([text_dataset[stock][key], zero_padding], 0)
                else:
                    data = text_dataset[stock][key][:5]
            else: 
                zero_padding = tf.zeros([5, 512], dtype=tf.float32)
                data = zero_padding
            if len(time_dur) != 5:
                time_dur.append(data)
                if len(time_dur) ==5:
                    stock_day.append(time_dur)
            else:
                time_dur.popleft()
                time_dur.append(data)
                stock_day.append(time_dur)
        dataset.append(stock_day)

    dataset = np.array(dataset)
    return dataset

def load_price(logger):
    price_dataset = defaultdict(dict)
    logger.info("#Load Price")
    for stock in tqdm(os.listdir(PRICE_PATH),desc="Price Iteration"):
        prices = pd.read_csv(os.path.join(PRICE_PATH, stock))
        stock, _ = stock.split(".")
        if stock=='GMRE':
            logger.info("Price data has more stock than tweet, so exclude GMRE")
            continue
        temp_prices = copy.deepcopy(prices)
        prices['y'] = -1
        for i in range(1, len(prices)):
            day = prices['Date'].iloc[i]
            prices['High'].iloc[i] /= temp_prices['Adj Close'].iloc[i-1]
            prices['Low'].iloc[i] /= temp_prices['Adj Close'].iloc[i-1]
            prices['Adj Close'].iloc[i] /= temp_prices['Adj Close'].iloc[i-1]
            prices['y'].iloc[i] = prices['Adj Close'].iloc[i]>=1
            price_dataset[stock][day] = ((prices['Adj Close'][i], prices['High'][i], prices['Low'][i]), prices['y'][i])

    logger.info("#Price Train Range Start")
    train_dataset, train_target = gen_price_dataset(price_dataset, train_range)
    logger.info("#Price Val Range Start")
    val_dataset, val_target = gen_price_dataset(price_dataset, val_range)
    logger.info("#Price Test Range Start")
    test_dataset, test_target = gen_price_dataset(price_dataset, test_range)  

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)

def load_text(logger):
    logger.info("#Universal Sentence Encoder Start")

    text_dataset= defaultdict(dict)
    for stock in tqdm(os.listdir(TEXT_PATH), desc='Text Iteration'):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)
        for day in tqdm(text_list, desc="Day Iteration"):
            data = [" ".join(json.loads(line)['text']) for line in open(os.path.join(stock_path, day), 'r')]
            data =list(dict.fromkeys(data))
            embedded_data = model(data)
            text_dataset[stock][day] = embedded_data

    logger.info("# Text Train Range Start")
    train_dataset = gen_text_dataset(text_dataset, train_range)
    logger.info("# Text Val Range Start")
    val_dataset = gen_text_dataset(text_dataset, val_range)
    logger.info("# Text Test Range Start")
    test_dataset = gen_text_dataset(text_dataset, test_range)          
            
    return train_dataset, val_dataset, test_dataset

def load_graph(market_name, logger):
    logger.info("#Graph Load from relation data")
    
    def normalize_adj(mx):
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    adj = np.load(os.path.join(GRAPH_PATH, market_name+'_graph.npy'))
    rel_shape = [adj.shape[0], adj.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                    np.sum(adj, axis=2))
    adj = np.where(mask_flags, np.ones(rel_shape)*1e-9, np.zeros(rel_shape))
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def make_dataset(price_dataset, text_dataset):
    price, labels = price_dataset
    assert price.shape[1]==text_dataset.shape[1]
    features = []
    for idx in range(labels.shape[1]-1):
        #(n_stock, n_feature, n_day, price_feature)
        price_feature = price[:,idx, :, :]
        #(n_stock, n_feature, n_day, n_text, text_feature)
        text_feature = text_dataset[:,idx, :, :, :]
        # future prediction
        label = labels[:, idx+1]
        features.append((price_feature, text_feature, label))

    dataset = MTSRDataset(features)
    return dataset


def load_dataset(args, logger):
    train_price, val_price, test_price = load_price(logger)
    train_text, val_text, test_text = load_text(logger)
    adj = load_graph(args.market_name, logger)

    train_dataset = make_dataset(train_price, train_text)
    val_dataset = make_dataset(val_price, val_text)
    test_dataset = make_dataset(test_price, test_text)

    return train_dataset, val_dataset, test_dataset, adj