import torch
import sys
import os
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_acc_pre_rec_f1, take_no_pad


def train(model, iterator, optimizer, criterion, EP, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, att_mask = batch
        x = x.to(device)
        y = y.to(device)
        att_mask = att_mask.to(device)

        optimizer.zero_grad()

        # forward pass
        if EP.use_rnn:
            logits, y, y_pred = model(x, y, att_mask)  # logits: (N, T, VOCAB), y: (N, T)
        else:
            logits, y, y_pred, loss = model(x, y, att_mask)  # logits: (N, T, VOCAB), y: (N, T)

        # for measurment save only the non padded tags
        y_true_noPad, y_pred_noPad = take_no_pad(seqlens, y_pred, y)
        
        # reshape for further computation
        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)
        #y_pred = y_pred.view(-1)  # (N*T,)

        # if through LSTM we can get the loss of the bert model and have to calculate it our own
        if EP.use_rnn:
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

        # get current loss etc.
        acc_epoch, pr_epoch, rec_epoch, f1_epoch = get_acc_pre_rec_f1(y_true_noPad, y_pred_noPad)
        epoch_loss += loss.item()
        epoch_acc += acc_epoch
        f1 += f1_epoch
        precision += pr_epoch
        recall += rec_epoch

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision / len(iterator), recall / len(
        iterator), f1 / len(iterator)


def eval(model, iterator, criterion, EP, device, test=False):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens, att_mask = batch
            x = x.to(device)
            y = y.to(device)
            att_mask = att_mask.to(device)
            # forward pass
            if EP.use_rnn:
                logits, y, y_pred = model(x, y, att_mask)  # logits: (N, T, VOCAB), y: (N, T)
            else:
                logits, y, y_pred, loss = model(x, y, att_mask)  # logits: (N, T, VOCAB), y: (N, T)


            # for measurment save only the non padded tags
            y_true_noPad, y_pred_noPad = take_no_pad(seqlens, y_pred, y)
            
            # reshape for further computation
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            #y_pred = y_pred.view(-1)  # (N*T,)

            # if through LSTM we can get the loss of the bert model and have to calculate it our own
            if EP.use_rnn:
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
