from collections import defaultdict, deque
import copy
from dateutil.parser import parse
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

def gen_price(logger=False):
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
        os.mkdir('./price_data')
    
    np.save('./price_data/price_train.npy', train_dataset)
    #np.save('./price_data/price_train.npy', train_dataset)
    np.save('./price_data/price_val.npy', val_dataset)
    np.save('./price_data/price_test.npy', test_dataset)

    print("Complete Price data generation")
    
    return [train_dataset, train_target, stocks, train_dates], [val_dataset, val_target, _, val_dates], [test_dataset, test_target, _, test_dates]

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
    tweet_dataset, time_dataset = [], []
    for stock in os.listdir(TEXT_PATH):
        text_stock_day, time_stock_day = [], []
        time_text_dur, time_dur = deque(), deque()
        for date in date_range:
            key = date.strftime('%Y-%m-%d')
            if key in text_dataset[stock]:
                text_input, time_delta = text_dataset[stock][key]
                if text_input.shape[0] <= 10:
                    zero_padding = torch.zeros([10-text_input.shape[0], 60], dtype=torch.long)
                    one_padding = torch.ones([10-time_delta.shape[0],1], dtype=torch.float32)

                    text_data = torch.cat([text_input, zero_padding], 0)
                    time_data = torch.cat([time_delta, one_padding], 0)
                else:
                    text_data = text_input[:10]
                    time_data = time_delta[:10]
            else: 
                zero_padding = torch.zeros([10, 60], dtype=torch.long)
                one_padding = torch.ones([10,1], dtype=torch.float32)
                text_data = zero_padding
                time_data = one_padding
            #window
            #print(text_data.shape)
            if len(time_dur) != 5:
                time_text_dur.append(text_data)
                time_dur.append(time_data)
                if len(time_dur) ==5:
                    text_stock_day.append(time_text_dur)
                    time_stock_day.append(time_dur)
            else:
                time_dur.popleft()
                time_text_dur.popleft()
                time_text_dur.append(text_data)
                time_dur.append(time_data)
                text_stock_day.append(time_text_dur)
                time_stock_day.append(time_dur)
            #print(len(time_text_dur))
        tweet_dataset.append(text_stock_day)
        time_dataset.append(time_stock_day)

    text_dataset = np.array(tweet_dataset)
    time_dataset = np.array(time_dataset)

    return text_dataset, time_dataset

def gen_text(logger=False):
    text_dataset = defaultdict(dict)
    for stock in tqdm(os.listdir(TEXT_PATH), desc = "Text Iteration"):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)

        for day in tqdm(text_list, desc = "Day Iteration"):
            inputs_list, time_delta = [], []
            data = [json.loads(line) for line in open(os.path.join(stock_path, day), 'r')]
            data = sorted(data, key= lambda x:x['created_at'])
            last_time = 0
            for d in data:
                if parse(d['created_at']).strftime('%Y-%m-%d')==day:
                    string = tweet_preprocessing(d['text'])
                    inputs = bert_tokenzier(string, max_length= 60, padding= 'max_length', truncation= True, return_tensors='pt')
                    now_time = parse(d['created_at'])
                    if last_time:
                        if ((now_time - last_time).seconds/60)==0:
                            continue
                        time_delta.append(1/((now_time - last_time).seconds/60))
                    else:
                        time_delta.append(torch.tensor(1.0))

                    inputs_list.append(inputs['input_ids'])
                    last_time = now_time
            if len(inputs_list):
                inputs = torch.cat(inputs_list)
                time_delta = torch.tensor(time_delta).unsqueeze(-1)
            else:
                inputs = torch.zeros((1,60), dtype=torch.long)
                time_delta = torch.ones((1,1), dtype=torch.float32)
            # if len(time_delta) < 10:
            #     one_padding = torch.ones([10-len(time_delta),1], dtype=torch.float32)
            #     time_delta = torch.cat([time_delta, one_padding], 0)
            text_dataset[stock][day] = (inputs, time_delta)

    print("# Text Train Range Start")
    train_text_dataset, train_time_dataset = gen_text_dataset(text_dataset, train_range)
    print("# Text Val Range Start")
    val_text_dataset, val_time_dataset = gen_text_dataset(text_dataset, val_range)
    print("# Text Test Range Start")
    test_text_dataset, test_time_dataset = gen_text_dataset(text_dataset, test_range)

    if not os.path.exists('./text_data'):
        os.mkdir('./text_data')

    if not os.path.exists('./time_data'):
        os.mkdir('./time_data')
    
    np.save('./text_data/text_train.npy', train_text_dataset)
    np.save('./text_data/text_val.npy', val_text_dataset)
    np.save('./text_data/text_test.npy', test_text_dataset)
            
    np.save('./time_data/time_train.npy', train_time_dataset)
    np.save('./time_data/time_val.npy', val_time_dataset)
    np.save('./time_data/time_test.npy', test_time_dataset)

    print("Complete Text and Time data generation")

    return (train_text_dataset, train_time_dataset),(val_text_dataset, val_time_dataset),(test_text_dataset, test_time_dataset)

def gen_data_npy():
    gen_price()
    gen_text()


if __name__ == "__main__":
    gen_data_npy()