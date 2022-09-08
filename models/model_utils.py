import os

import numpy as np
import pandas as pd

from data_utils import gen_text, gen_price
from MTSRDataset import MTSRDataset

DATA_PATH = './data/'
PRICE_PATH = os.path.join(DATA_PATH, 'price', 'raw')
TEXT_PATH = os.path.join(DATA_PATH, 'tweet','preprocessed')
GRAPH_PATH = os.path.join(DATA_PATH, 'relation')

def load_price(logger):
    logger.info("Load price data from numpy array")
    train_price = np.load("./price_data/price_train.npy", allow_pickle=True)
    val_price = np.load("./price_data/price_val.npy", allow_pickle=True)
    test_price = np.load("./price_data/price_test.npy", allow_pickle=True)

    return train_price, val_price, test_price

def load_text(logger):
    logger.info("Load text data from numpy array")
    train_text = np.load("./text_data/text_train.npy", allow_pickle=True)
    val_text = np.load("./text_data/text_val.npy", allow_pickle=True)
    test_text = np.load("./text_data/text_test.npy", allow_pickle=True)

    return train_text, val_text, test_text

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
    train_price, val_price, test_price = gen_price(logger)
    print(train_price[0].shape)
    (train_text, train_time), (val_text, val_time), (test_text, val_time) = gen_text(logger)
    print(train_text.shape)
    train_dataset = make_dataset(train_price, train_text)
    val_dataset = make_dataset(val_price, val_text)
    test_dataset = make_dataset(test_price, test_text)

    return train_dataset, val_dataset, test_dataset