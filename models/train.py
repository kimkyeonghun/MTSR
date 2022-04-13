import os
import logging
import argparse

from torch.optim import Adam

from model import GAT
import utils
from model_utils import *
from trainer import train

logger, log_dir = utils.get_logger(os.path.join('./logs'))

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs to trian')
parser.add_argument('--train_batch_size', type=int, default=1,)
parser.add_argument('--val_batch_size', type=int, default=1,)
parser.add_argument('--test_batch_size', type=int, default=1,)
parser.add_argument('--n_head', type=int, default=8, help='Number of attention head')
parser.add_argument('--stock_num', type=int, default=87,)
parser.add_argument('--lr', type=float, default=1e-5,)
parser.add_argument('--alpha', type=float, default=0.2,)
parser.add_argument('--dropout', type=float, default=0.5,)
parser.add_argument('--market_name',type=str, default='NASDAQ')
args = parser.parse_args()

def prepare_for_training(logger):
    model = GAT(n_feature= 64, n_hidden= 64, n_class = 2,
        dropout= args.dropout,
        alpha= args.alpha,
        n_heads= args.n_head,
        stock_num= args.stock_num,
        logger = logger)
    model.cuda()

    optimizer = Adam(model.parameters(), lr = args.lr)

    return model, optimizer


def main():
    logger.info("#Load Dataset")

    (train_dataset, val_dataset, test_dataset, adj) = load_dataset(args, logger)

    logger.info("#Prepare for Training")
    model, optimizer = prepare_for_training(logger)

    train(args, model, train_dataset, val_dataset, test_dataset, adj, optimizer, logger)


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
