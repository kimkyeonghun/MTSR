from collections import defaultdict, deque
import copy
import json
import os
import re
from tqdm import tqdm

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer

train_range = pd.date_range(start='2014-01-01', end='2015-07-31', freq='1D')
val_range = pd.date_range(start='2015-08-01', end='2015-09-30', freq='1D')
test_range = pd.date_range(start='2015-10-01', end='2016-01-01', freq='1D')

DATA_PATH = './data/'
PRICE_PATH = os.path.join(DATA_PATH, 'price', 'raw')
TEXT_PATH = os.path.join(DATA_PATH, 'tweet','raw')
GRAPH_PATH = os.path.join(DATA_PATH, 'relation')

tokenizer = TweetTokenizer(strip_handles=True,
                               reduce_len=True)
stop_words = stopwords.words('english')
bert_tokenzier = BertTokenizer.from_pretrained('bert-base-cased')

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

def gen_price( ):
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

    if not os.path.exists('./price_data'):
        os.mkdir('./price_date')
    
    np.save('./price_date/price_train.npy', train_dataset)
    np.save('./price_date/price_val.npy', val_dataset)
    np.save('./price_date/price_test.npy', test_dataset)

    print("Complete Price data generation")

def remove_emoji(string):
    emoji_pattetn = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    return emoji_pattetn.sub(r'', string)

def tweet_preprocessing(string):
    string = re.sub(r'^RT[\s]+', '', string)
    string = re.sub(r'http\S+', '', string)
    string = re.sub(r'#', '', string)
    string = re.sub(r'\$', '', string)
    string = re.sub(r'\:|\-|\%|\&|\?|\!|\~|\(|\)','',string)
    string = re.sub(r'([A-Za-z])\1{2,}', r'\1', string)
    string = remove_emoji(string)
    string = tokenizer.tokenize((string))
    string = [w for w in string if not w.lower() in stop_words]
    return " ".join(string)

def gen_text_dataset(text_dataset, date_range):
    dataset = []
    for stock in os.listdir(TEXT_PATH):
        stock_day = []
        time_dur = deque()
        for date in date_range:
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                if text_dataset[stock][key].shape[0] <= 10:
                    zero_padding = torch.zeros([1-text_dataset[stock][key].shape[0], 60], dtype=torch.long)
                    data = torch.cat([text_dataset[stock][key], zero_padding], 0)
                else:
                    data = text_dataset[stock][key][:1]
            else: 
                zero_padding = torch.zeros([10, 60], dtype=torch.long)
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

def gen_text():
    text_dataset = defaultdict(dict)
    for stock in tqdm(os.listdir(TEXT_PATH), desc = "Text Iteration"):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)
        for day in tqdm(text_list, desc = "Day Iteration"):
            data = [json.loads(line)['text'] for line in open(os.path.join(stock_path, day), 'r')]
            for d in data:
                string = tweet_preprocessing(d)
                inputs = bert_tokenzier(string, max_length= 60, padding= 'max_length', truncation= True, return_tensors='pt')
                text_dataset[stock][day] = inputs['input_ids']

    print("# Text Train Range Start")
    train_dataset = gen_text_dataset(text_dataset, train_range)
    print("# Text Val Range Start")
    val_dataset = gen_text_dataset(text_dataset, val_range)
    print("# Text Test Range Start")
    test_dataset = gen_text_dataset(text_dataset, test_range)

    if not os.path.exists('./text_data'):
        os.mkdir('./text_data')
    
    np.save('./text_data/text_train.npy', train_dataset)
    np.save('./text_data/text_val.npy', val_dataset)
    np.save('./text_data/text_test.npy', test_dataset)
            
    print("Complete Price data generation")

def gen_data_npy():
    gen_price()
    gen_text()


if __name__ == "__main__":
    gen_data_npy()