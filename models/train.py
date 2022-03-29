import os
import logging
import argparse

from model import GAT
import utils
from model_utils import *
from trainer import train

logger, log_dir = utils.get_logger(os.path.join('./logs'))

parser = argparse.ArgumentParser()
parser.add_argument()

args = parser.parse_args()

def prepare_for_training():
    pass


def main():
    logger.info("#Load Dataset")

    (train_dataset, val_dataset, test_dataset) = loda_dataset()

    logger.info("#Prepare for Training")
    model, optimizer, scheduler = prepare_for_training()

    train(args, model, train_dataset, val_dataset, test_dataset, optimizer, scheduler, logger)


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
