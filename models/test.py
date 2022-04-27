import os
import argparse

from model import GAT
from model_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs to trian')
parser.add_argument('--train_batch_size', type=int, default=1,)
parser.add_argument('--val_batch_size', type=int, default=1,)
parser.add_argument('--test_batch_size', type=int, default=1,)
parser.add_argument('--n_head', type=int, default=8, help='Number of attention head')
parser.add_argument('--stock_num', type=int, default=87,)
parser.add_argument('--lr', type=float, default=5e-5,)
parser.add_argument('--alpha', type=float, default=0.2,)
parser.add_argument('--dropout', type=float, default=0.38,)
parser.add_argument('--market_name',type=str, default='NASDAQ')

parser.add_argument('--dir', type=str)
parser.add_argument('--model_num', type=str)
args = parser.parse_args()

from collections import defaultdict, deque
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from torch.utils.data import DataLoader, RandomSampler

from MTSRDataset import MTSRDataset, collate

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

def load_price():
    price_dataset = defaultdict(dict)
    for stock in tqdm(os.listdir(PRICE_PATH),desc="Price Iteration"):
        prices = pd.read_csv(os.path.join(PRICE_PATH, stock))
        stock, _ = stock.split(".")
        if stock=='GMRE':
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

    train_dataset, train_target, stocks, train_dates = gen_price_dataset(price_dataset, train_range)
    val_dataset, val_target, _, val_dates = gen_price_dataset(price_dataset, val_range)
    test_dataset, test_target, _, test_dates = gen_price_dataset(price_dataset, test_range)  

    return (train_dataset, train_target, stocks, train_dates),\
        (val_dataset, val_target, stocks, val_dates),\
        (test_dataset, test_target, stocks, test_dates)

def load_text():
    train_text = np.load("./text_data/text_train.npy")
    val_text = np.load("./text_data/text_val.npy")
    test_text = np.load("./text_data/text_test.npy")

    return train_text, val_text, test_text

def load_graph(market_name):
    
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


def load_dataset(args):
    train_price, val_price, test_price = load_price()
    train_text, val_text, test_text = load_text()
    adj = load_graph(args.market_name, )

    train_dataset = make_dataset(train_price, train_text)
    val_dataset = make_dataset(val_price, val_text)
    test_dataset = make_dataset(test_price, test_text)

    return train_dataset, val_dataset, test_dataset, adj

def prepare_for_training():
    model = GAT(n_feature= 64, n_hidden= 64, n_class = 2,
        dropout= args.dropout,
        alpha= args.alpha,
        n_heads= args.n_head,
        stock_num= args.stock_num,
        logger = None)
    model.cuda()

    #optimizer = Adam(model.parameters(), lr = args.lr)

    return model


def main():

    (_, _, test_dataset, adj) = load_dataset(args)

    model = prepare_for_training()

    model_path = os.path.join('./model_save', args.dir, 'model_' + args.model_num + '.pt')
    model.load_state_dict(torch.load(model_path))

    testSampler = RandomSampler(test_dataset)
    testDataLoader = DataLoader(
        test_dataset, sampler = testSampler, batch_size=args.test_batch_size, collate_fn= collate
    )

    model.eval()

    results = dict()
    for _, batch in enumerate(tqdm(testDataLoader, desc="Iteration")):
        test_price, test_text, _, date, stocks = batch
        test_text = test_text.unsqueeze(0)
        test_price = test_price.unsqueeze(0)
        
        price_attentions, intra_day_attentions, inter_day_attentions,\
            stock_attention_score = model.extract_attention_map(test_text, test_price, stocks[0])
        date = tuple(date[0])
        results[date] = defaultdict(dict)
        for stock in stocks[0]:
            results[date][stock]['price'] = price_attentions[stock]
            results[date][stock]['intra'] = intra_day_attentions[stock]
            results[date][stock]['inter'] = inter_day_attentions[stock]
            results[date][stock]['stock'] = stock_attention_score

    import pickle

    with open("results.json",'wb') as fp:
        pickle.dump(results, fp)


if __name__ == "__main__":
    main()
