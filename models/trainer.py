import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import torch
from torch.utils.data import DataLoader, RandomSampler

import utils

def train_epoch(args, model, train_dataset, adj, optimizer):

    trainSampler = RandomSampler(train_dataset)
    trainDataLoader = DataLoader(
        train_dataset, sampler =trainSampler, batch_size=args.train_batch_size
    )
    train_loss = 0.0
    tr_steps = 0
    model.train()

    for _, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
        train_text, train_price, train_label = batch
        outputs = model(train_text, train_price, train_label, adj, train = True)

        loss = outputs[0]

        loss.backward()
        tr_steps += 1

        #train_loss += loss.item()
        train_loss += loss

        optimizer.step()
        optimizer.zero_grad()

    return train_loss/tr_steps

def val_epoch(args, model, train_dataset, adj):

    valSampler = RandomSampler(train_dataset)
    valDataLoader = DataLoader(
        train_dataset, sampler =valSampler, batch_size=args.val_batch_size
    )
    val_loss = 0.0
    tr_steps = 0
    model.eval()

    for _, batch in enumerate(tqdm(valDataLoader, desc="Iteration")):
        val_text, val_price, val_label = batch
        outputs = model(val_text, val_price, val_label, adj, train = False)

        loss = outputs[0]
        tr_steps += 1

        #train_loss += loss.item()
        val_loss += loss

    return val_loss/tr_steps

def test_epoch(args, model, train_dataset, adj):

    testSampler = RandomSampler(train_dataset)
    testDataLoader = DataLoader(
        train_dataset, sampler =testSampler, batch_size=args.test_batch_size
    )

    model.eval()

    outputs = []
    labels = []

    for _, batch in enumerate(tqdm(testDataLoader, desc="Iteration")):
        test_text, test_price, test_label = batch
        outputs = model(test_text, test_price, test_label, adj, train = False)

        _, output = outputs[0]

        outputs.append(output)
        labels.append(test_label)

    return [], outputs, labels

def test_score(pred, label):
    f1 = f1_score(np.array(pred).reshape((-1,)), np.array(label).reshape((-1,)), average='micro')
    mat = matthews_corrcoef(np.array(pred).reshape((-1,)), np.array(label).reshape((-1,)))
    acc = accuracy_score(pred, label)
    return acc, f1, mat 


def train(args, model, train_datatset, val_dataset, test_dataset, adj, optimizer, logger):
    model_save_path = utils.make_date_dir('./model_save')
    logger.info("Model save path: {}".format(model_save_path))

    best_loss = 0.0
    patience = 0

    for epoch in range(1, args.n_epochs):
        patience += 1

        logger.info("=======================Train=======================")
        train_loss = train_epoch(args, model, train_datatset, adj, optimizer)
        logger.info("[Train Epoch {}] Train_loss: {}".format(epoch+1, train_loss,))

        logger.info("=======================Validation=======================")
        val_loss = val_epoch(args, model, val_dataset, adj)
        logger.info("[Val Epoch {}] Val_loss: {}".format(epoch+1, val_loss,))

        logger.info("=======================Test=======================")
        _, preds, labels = test_epoch(args, model, test_dataset, adj)
        acc, f1, mat = test_score(preds, labels)
        logger.info("[Test Epoch {}] Accuracy: {} F1-score: {} MCC: {}"
                .format(epoch+1, acc, f1, mat))

        if val_loss > best_loss:
            torch.save(model.state_dict(),
                     os.path.join(model_save_path, 'model_' + str(epoch+1)+'.pt'))
            best_epoch = epoch
            best_loss = val_loss
            best_acc = acc
            best_f1 = f1
            best_mat = mat
            patience = 0
        
        if patience==5:
            break
        
    logger.info(f"[Best Epoch {best_epoch}] Best_Acc: {best_acc} Best_f1: {best_f1} Best_mat: {best_mat}")
            
