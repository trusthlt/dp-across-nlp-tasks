import time
import torch
from queue import Queue
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def get_acc_pre_rec_f1(y_true, y_pred):
    assert (len(y_true) == len(y_pred))

    # accuracy
    acc = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            acc += 1

    # precision, recall, f1
    pr_epoch, rec_epoch, f1_epoch, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return acc / len(y_true), pr_epoch, rec_epoch, f1_epoch

def take_no_pad(seqlens, y_pred, y):
    # for measurment save only the non padded tags
    y_true_noPad = []
    y_pred_noPad = []

    for i, seqlen in enumerate(seqlens):
        y_pred_noPad.append(y_pred[i][:seqlen].cpu().detach().numpy())
        y_true_noPad.append(y[i][:seqlen].cpu().detach().numpy())

        if not (len(y_true_noPad[i]) == seqlens[i] and len(y_pred_noPad[i]) == seqlens[i]):
            print(y_pred)
            print(len(y_pred))
            print(y)
            print(len(y))
            print(f'{len(y_true_noPad[i])} == {seqlens[i]} and {len(y_pred_noPad[i])} == {seqlens[i]}')
            print(f'{y_true_noPad[i]} with length: {seqlens[i]}')
            print(f'{y_pred_noPad[i]} with length: {seqlens[i]}')

        # sanity check if seq len is actual length of seqence
        assert(len(y_true_noPad[i]) == seqlens[i] and len(y_pred_noPad[i]) == seqlens[i])
    
    return y_true_noPad, y_pred_noPad


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.q = Queue(maxsize = self.patience)
        self.max_acc = -1
        self.counter = 0
    
    def should_stop(self, accuracy):

        # check if accuracy is greater than max than empy out queue and set new max
        if accuracy > self.max_acc:
            self.q.queue.clear()
            self.max_acc = accuracy
            self.counter = 0
        else:
            # else add element to queue and check if queue is full (if we should do early stopping)
            self.q.put(accuracy)
            self.counter += 1
            if self.q.full():
                # do early stopping
                return True