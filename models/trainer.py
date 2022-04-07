from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler

import utils

def train_epoch(args, model, train_dataset, optimizer):

    trainSampler = RandomSampler(train_dataset)
    trainDataLoader = DataLoader(
        train_dataset, sampler =trainSampler, batch_size=args.train_batch_size
    )
    train_loss = 0.0
    tr_steps = 0
    model.train()

    for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
        train_text, train_price, train_label = batch
        outputs = model(train_text, train_price, )

        loss = outputs[0]

        loss.backward()
        tr_steps += 1

        #train_loss += loss.item()
        train_loss += loss

        optimizer.step()
        optimizer.zero_grad()

    return train_loss/tr_steps



def train(args, model, train_datatset, val_dataset, test_dataset, adj, optimizer, logger):
    model_save_path = utils.make_date_dir('./model_save')
    logger.info("Model save path: {}".format(model_save_path))

    best_loss = 0.0
    patience = 0

    for epoch in args.n_epoch:
        patience += 1

        logger.info("=======================Train=======================")
        train_loss = train_epoch(args, model, train_datatset, optimizer)
        logger.info("[Train Epoch {}] Train_loss: {}".format(epoch+1, train_loss,))