import torch
import sys
import os
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_acc_pre_rec_f1, take_no_pad
from dataset_wikiann import idx2tag


def train(model, iterator, optimizer, criterion, EP):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0
    for i, batch in enumerate(iterator):
        words, x, tags, y, seqlens = batch

        # break if the last batch is not full
        if i+1 == len(iterator) and (not len(seqlens) == EP.BATCH_SIZE):
                break
                
        for i in range(len(x)):
            assert len(x[i]) == len(y[i]), f'false statement x == y == sl {len(x[i])} == {len(y[i])}'

        optimizer.zero_grad()
        x = torch.transpose(x, 0, 1) # (seq len, batch size)

        # forward pass
        logits, y, y_pred = model(x, y, words)  # logits: (N, T, TAGS), y: (N, T)

        # for measurment save only the non padded tags
        y_true_noPad, y_pred_noPad = take_no_pad(seqlens, y_pred, y)

        # reshape for further computation
        logits = logits.view(-1, logits.shape[-1])  # (N*T, TAGS)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        ########
        # evaluate train result
        ########
        
        # make 1D array
        temp_pred = []
        temp_true = []
        for arrT, arrP in zip(y_true_noPad, y_pred_noPad):
            temp_pred.extend(arrP)
            temp_true.extend(arrT)
        y_pred_noPad = temp_pred
        y_true_noPad = temp_true

        acc_epoch, pr_epoch, rec_epoch, f1_epoch = get_acc_pre_rec_f1(y_true_noPad, y_pred_noPad)
        epoch_loss += loss.item()
        epoch_acc += acc_epoch
        f1 += f1_epoch
        precision += pr_epoch
        recall += rec_epoch

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision / len(iterator), recall / len(
        iterator), f1 / len(iterator)


def eval(model, iterator, criterion, optimizer, EP):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, tags, y, seqlens = batch

            # break if the last batch is not full
            if i+1 == len(iterator) and (not len(seqlens) == EP.BATCH_SIZE):
                break

            for i in range(len(x)):
                assert len(x[i]) == len(y[i]), f'false statement x == y == sl {len(x[i])} == {len(y[i])}'

            optimizer.zero_grad()
            x = torch.transpose(x, 0, 1) # (seq len, batch size)

            # forward pass
            logits, y, y_pred = model(x, y, words)  # logits: (N, T, TAGS), y: (N, T)

            # for measurment save only the non padded tags
            y_true_noPad, y_pred_noPad = take_no_pad(seqlens, y_pred, y)

            # reshape for further computation
            logits = logits.view(-1, logits.shape[-1])  # (N*T, TAGS)
            y = y.view(-1)  # (N*T,)

            loss = criterion(logits, y)

            ########
            # evaluate train result
            ########
            
            # make 1D array
            temp_pred = []
            temp_true = []
            for arrT, arrP in zip(y_true_noPad, y_pred_noPad):
                temp_pred.extend(arrP)
                temp_true.extend(arrT)
            y_pred_noPad = temp_pred
            y_true_noPad = temp_true

            # get current loss etc.
            acc_epoch, pr_epoch, rec_epoch, f1_epoch = get_acc_pre_rec_f1(y_true_noPad, y_pred_noPad)
            epoch_loss += loss.item()
            epoch_acc += acc_epoch
            f1 += f1_epoch
            precision += pr_epoch
            recall += rec_epoch

        return epoch_loss / len(iterator), epoch_acc / len(iterator), precision / len(iterator), recall / len(
            iterator), f1 / len(iterator)
