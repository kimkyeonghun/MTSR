import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

import utils
from MTSRDataset import collate

weight = 1

loss_fct = torch.nn.MSELoss()


def train_epoch(args, model, train_dataset, optimizer):

    trainSampler = RandomSampler(train_dataset)
    trainDataLoader = DataLoader(
        train_dataset, sampler=trainSampler, batch_size=args.train_batch_size, collate_fn=collate,
    )
    train_loss = 0.0
    tr_steps = 0
    model.train()
    preds = []
    labels = []
    for _, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
        _, train_text, train_time, train_label, _, _ = batch
        train_label = train_label.squeeze(0)
        train_text = train_text.squeeze(0)
        train_time = train_time.squeeze(0)
        loss = 0
        true_label = 0
        for s in range(0, train_text.size(0), 1):
            one_train_text = train_text[s:s +
                                        args.stock_batch_size, :].to("cuda:1")
            one_train_time = train_time[s:s +
                                        args.stock_batch_size, :].to("cuda:1")
            outputs = model(one_train_text, one_train_time, s)
            if all(torch.zeros(args.stock_batch_size) == train_label[s:s+args.stock_batch_size].view(-1)):
                del one_train_text
                del one_train_time
                del outputs
            else:
                true_label += args.stock_batch_size
                loss += 10 * \
                    loss_fct(
                        train_label[s:s+args.stock_batch_size].view(-1).to("cuda:1"), outputs.view(-1))
                preds.append(
                    np.ravel(3*torch.tanh(outputs).detach().cpu().numpy())[0])
                labels.append(
                    np.ravel(train_label[s:s+args.stock_batch_size].detach().cpu().numpy())[0])
                del one_train_text
                del one_train_time
                del outputs
        if true_label:
            loss /= true_label

            loss.backward()
            tr_steps += 1

            #train_loss += loss.item()
            train_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

    del trainDataLoader

    return train_loss/tr_steps, preds, labels


def val_epoch(args, model, val_dataset, logger):

    valSampler = RandomSampler(val_dataset)
    valDataLoader = DataLoader(
        val_dataset, sampler=valSampler, batch_size=args.val_batch_size, collate_fn=collate,
    )
    val_loss = 0.0
    tr_steps = 0
    model.eval()

    labels = []
    preds = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataLoader, desc="Iteration")):
            _, val_text, val_time, val_label, _, _ = batch
            val_label = val_label.squeeze(0)
            val_text = val_text.squeeze(0)
            val_time = val_time.squeeze(0)
            loss = 0
            for s in range(0, val_text.size(0), args.stock_batch_size):
                one_val_text = val_text[s:s +
                                        args.stock_batch_size, :].to("cuda:1")
                one_val_time = val_time[s:s +
                                        args.stock_batch_size, :].to("cuda:1")
                outputs = model(one_val_text, one_val_time, s)
                loss += 10 * \
                    loss_fct(
                        val_label[s:s+args.stock_batch_size].view(-1).to("cuda:1"), outputs.view(-1))
                preds.append(
                    np.ravel(3*torch.tanh(outputs).detach().cpu().numpy())[0])
                labels.append(
                    np.ravel(val_label[s:s+args.stock_batch_size].detach().cpu().numpy())[0])
                del one_val_text
                del one_val_time
                del outputs

            loss /= val_text.size(0)
            #train_loss += loss.item()
            val_loss += loss
            tr_steps += 1

    return val_loss/tr_steps, preds, labels


def test_epoch(args, model, test_dataset):

    testSampler = RandomSampler(test_dataset)
    testDataLoader = DataLoader(
        test_dataset, sampler=testSampler, batch_size=args.test_batch_size, collate_fn=collate,
    )

    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataLoader, desc="Iteration")):
            _, test_text, test_time, test_label, _, _ = batch
            test_label = test_label.squeeze(0)
            test_text = test_text.squeeze(0)
            test_time = test_time.squeeze(0)
            for s in range(0, test_text.size(0), args.stock_batch_size):
                one_test_text = test_text[s:s +
                                          args.stock_batch_size, :].to("cuda:1")
                one_test_time = test_time[s:s +
                                          args.stock_batch_size, :].to("cuda:1")
                outputs = model(one_test_text, one_test_time, s)
                # print(train_label[s:s+1])
                preds.append(np.ravel(outputs.detach().cpu().numpy())[0])
                labels.append(
                    np.ravel(test_label[s:s+args.stock_batch_size].detach().cpu().numpy())[0])
                del one_test_text
                del one_test_time
                del outputs

    return [], preds, labels


def ACC7(value, true):
    """
    for 7 label
    """
    # print("-----------------------------------------------------------------------------------------")
    # print(value)
    assert len(value) == len(true)
    for i, v in enumerate(value):
        if v < -2:
            value[i] = -3
        elif -2 <= v < -1:
            value[i] = -2
        elif -1 <= v < 0:
            value[i] = -1
        elif v == 0:
            value[i] = 0
        elif 0 < v <= 1:
            value[i] = 1
        elif 1 < v <= 2:
            value[i] = 2
        elif v > 2:
            value[i] = 3

    for i, v in enumerate(true):
        if v < -2:
            true[i] = -3
        elif -2 <= v < -1:
            true[i] = -2
        elif -1 <= v < 0:
            true[i] = -1
        elif v == 0:
            true[i] = 0
        elif 0 < v <= 1:
            true[i] = 1
        elif 1 < v <= 2:
            true[i] = 2
        elif v > 2:
            true[i] = 3
    return np.sum(np.array(value) == np.array(true))/float(len(true))


def test_score(pred, label):
    acc = ACC7(pred, label)
    return acc


def train(args, model, train_datatset, val_dataset, test_dataset, optimizer, logger):
    model_save_path = utils.make_date_dir('./model_save')
    logger.info("Model save path: {}".format(model_save_path))

    best_loss = float('inf')
    best_acc = 0
    patience = 0

    for epoch in range(args.n_epochs):
        patience += 1

        logger.info("=======================Train=======================")
        train_loss, preds, labels = train_epoch(
            args, model, train_datatset, optimizer)
        # for name, param in model.named_parameters():
        #     print(name, param)
        logger.info("[Train Epoch {}] Train_loss: {}".format(
            epoch+1, train_loss,))

        train_acc = test_score(preds, labels)

        logger.info("[Train Epoch {}] Accuracy: {}"
                    .format(epoch+1, train_acc))

        logger.info("=======================Validation=======================")
        val_loss, preds, labels = val_epoch(args, model, val_dataset, logger)
        logger.info("[Val Epoch {}] Val_loss: {}".format(epoch+1, val_loss,))
        logger.info(preds)
        val_acc = test_score(preds, labels)

        logger.info("[Val Epoch {}] Accuracy: {}"
                    .format(epoch+1, val_acc))

        logger.info("=======================Test=======================")
        _, preds, labels = test_epoch(args, model, test_dataset)

        acc = test_score(preds, labels)
        logger.info("[Test Epoch {}] Accuracy: {}"
                    .format(epoch+1, acc))

        # if val_loss < best_loss:
        if val_acc > best_acc:
            torch.save(model.state_dict(),
                       os.path.join(model_save_path, 'model_' + str(epoch+1)+'.pt'))
            best_epoch = epoch+1
            best_loss = val_loss
            best_acc = val_acc
            patience = 0

        if patience == 50:
            break

    logger.info(f"[Best Epoch {best_epoch}] Best_Acc: {best_acc}")
