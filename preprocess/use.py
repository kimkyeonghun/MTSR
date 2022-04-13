from collections import deque, defaultdict
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
#not use tensorflow_text, but need because of USE module
import tensorflow_text
import tensorflow_hub as hub

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(model_url)

train_range = pd.date_range(start='2014-01-01', end='2015-07-31', freq='1D')
val_range = pd.date_range(start='2015-08-01', end='2015-09-30', freq='1D')
test_range = pd.date_range(start='2015-10-01', end='2016-01-01', freq='1D')

DATA_PATH = './data/'
TEXT_PATH = os.path.join(DATA_PATH, 'tweet','preprocessed')

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

def make_text_array():
    print("#Universal Sentence Encoder Start")

    text_dataset= defaultdict(dict)
    for stock in tqdm(os.listdir(TEXT_PATH), desc='Text Iteration'):
        stock_path = os.path.join(TEXT_PATH, stock)
        text_list = os.listdir(stock_path)
        for day in tqdm(text_list, desc="Day Iteration"):
            data = [" ".join(json.loads(line)['text']) for line in open(os.path.join(stock_path, day), 'r')]
            data =list(dict.fromkeys(data))
            embedded_data = model(data)
            text_dataset[stock][day] = embedded_data

    print("# Text Train Range Start")
    train_dataset = gen_text_dataset(text_dataset, train_range)
    print("# Text Val Range Start")
    val_dataset = gen_text_dataset(text_dataset, val_range)
    print("# Text Test Range Start")
    test_dataset = gen_text_dataset(text_dataset, test_range)

    np.save("./text_data/text_train.npy", train_dataset)
    np.save("./text_data/text_val.npy", val_dataset)
    np.save("./text_data/text_test.npy", test_dataset)


if __name__ == "__main__":
    make_text_array()
    