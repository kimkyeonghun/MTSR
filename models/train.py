import os
import logging
import argparse

from torch.optim import AdamW

from model import GAT
import utils
from model_utils import *
from trainer import train

logger, log_dir = utils.get_logger(os.path.join('./logs'))

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs to trian')
parser.add_argument('--train_batch_size', type=int, default=8,)
parser.add_argument('--h_head', type=int, default=8, help='Number of attention head')
parser.add_argument('--stock_num', type=int, default=8,)
parser.add_argument('--lr', type=float, default=1e-5,)
parser.add_argument('--alpha', type=float, default=0.2,)
parser.add_argument('--dropout', type=float, default=0.5,)
args = parser.parse_args()

def prepare_for_training():
    model = GAT(args.h_head, args.stock_num)
    model.cuda()

    optimizer = AdamW(model.parameter(), lr = args.lr)

    return model, optimizer


def main():
    logger.info("#Load Dataset")

    (train_dataset, val_dataset, test_dataset) = load_dataset()

    logger.info("#Prepare for Training")
    model, optimizer = prepare_for_training()

    train(args, model, train_dataset, val_dataset, test_dataset, optimizer, logger)


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
