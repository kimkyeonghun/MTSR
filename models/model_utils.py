from collections import defaultdict, deque
import copy
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from MTSRDataset import MTSRDataset

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
    stocks = []
    target_dates = []
    for stock in price_dataset.keys():
        stock_day = []
        ydata = []
        target_date = []
        time_dur = deque()
        dates  = deque()
        for date in date_range:
            date = date.strftime('%Y-%m-%d')
            if date in price_dataset[stock]:
                price_feature, y = price_dataset[stock][date]
            else:
                price_feature, y = (0,0,0), 0
            if len(time_dur)!=5:
                dates.append(date)
                time_dur.append(price_feature)
                if len(time_dur)==5:
                    stock_day.append(list(time_dur))
                    target_date.append(list(dates))
                    ydata.append(y)
            else:
                time_dur.popleft()
                time_dur.append(price_feature)
                dates.popleft()
                dates.append(date)
                target_date.append(list(dates))
                stock_day.append(list(time_dur))
                ydata.append(y)
        pdataset.append(stock_day)
        target.append(ydata)
        stocks.append(stock)
        target_dates.append(np.array(target_date))
    
    target = np.array(target)
    target_dates = np.array(target_dates)
    pdataset = np.array(pdataset)
    stocks = np.array(stocks)

    return pdataset, target, stocks, target_dates

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
    train_dataset, train_target, stocks, train_dates = gen_price_dataset(price_dataset, train_range)
    logger.info("#Price Val Range Start")
    val_dataset, val_target, _, val_dates = gen_price_dataset(price_dataset, val_range)
    logger.info("#Price Test Range Start")
    test_dataset, test_target, _, test_dates = gen_price_dataset(price_dataset, test_range)  

    return (train_dataset, train_target, stocks, train_dates),\
        (val_dataset, val_target, stocks, val_dates),\
        (test_dataset, test_target, stocks, test_dates)

def load_text(logger):
    logger.info("Load text data from numpy array")
    train_text = np.load("./text_data/text_train.npy")
    val_text = np.load("./text_data/text_val.npy")
    test_text = np.load("./text_data/text_test.npy")

    return train_text, val_text, test_text

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
    # print(adj.shape)
    # print(rel_shape)
    # print(np.sum(adj, axis=2).shape)
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                    np.sum(adj, axis=2))
    adj = np.where(mask_flags, np.ones(rel_shape)*1e-9, np.zeros(rel_shape))
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def make_dataset(price_dataset, text_dataset):
    price, labels, stocks, dates = price_dataset
    assert price.shape[1]==text_dataset.shape[1]
    features = []
    for idx in range(labels.shape[1]-1):
        #(n_stock, n_feature, n_day, price_feature)
        price_feature = price[:,idx, :, :]
        #(n_stock, n_feature, n_day, n_text, text_feature)
        text_feature = text_dataset[:,idx, :, :, :]
        # future prediction
        label = labels[:, idx+1]
        date = dates[:, idx, :]
        features.append(((text_feature, price_feature, label, date), stocks))

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