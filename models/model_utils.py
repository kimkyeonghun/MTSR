import os
import tensorflow_hub as hub

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'

DATA_PATH = '../data/'
test = ["$", "abb", "-", "7:45", "am", "abb", "ltd", "misses", "by", "$", "0.08", ",", "reports", "revs", "in-line", ";", "confirms", "2011-2", "...", "->", "URL", "stock", "stocks", "stockaction"]
def load_dataset():
    model = hub.load(model_url)
    return model(test)

print(load_dataset())