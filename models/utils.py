import os
import datetime
import logging
import logging.handlers

def get_logger(log_path='./logs'):
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(['[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format])

    i = 0
    today = datetime.datetime.now()
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(os.path.join(log_path, name)):
        i+=1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'

    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.setLevel(logging.INFO)
    logger.info("Writing logs at {}".format(os.path.join(log_path,name)))

    return logger, os.path.join(log_path, name)

    