import os
import logging
import argparse

from torch.optim import Adam

from model import FISA
import utils
import model_utils
from trainer import train

logger, log_dir = utils.get_logger(os.path.join('./logs'))

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000,
                    help='Number of epochs to trian')
parser.add_argument('--train_batch_size', type=int, default=1,)
parser.add_argument('--val_batch_size', type=int, default=1,)
parser.add_argument('--test_batch_size', type=int, default=1,)
parser.add_argument('--stock_batch_size', type=int, default=3,)
parser.add_argument('--stock_num', type=int, default=88,)
parser.add_argument('--lr', type=float, default=1e-6,)
parser.add_argument('--dropout', type=float, default=0.5,)
parser.add_argument('--market_name', type=str, default='NASDAQ')

parser.add_argument('--n_day', type=int, default=5)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def prepare_for_training(logger, real_stock_num):

    model = FISA(num_stocks=real_stock_num, drop_prob=args.dropout, logger=logger)
    logger.info(model)
    #model = nn.DataParallel(model,device_ids=[0,1], output_device=1)
    optimizer = Adam(model.parameters(), lr=args.lr)

    return model, optimizer


def main():
    logger.info("#Load Dataset")

    (train_dataset, val_dataset,
     test_dataset), real_stock_num = model_utils.load_dataset(args, logger)

    logger.info("#Prepare for Training")
    model, optimizer = prepare_for_training(logger, real_stock_num)

    train(args, model, train_dataset, val_dataset,
          test_dataset, optimizer, logger)


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
