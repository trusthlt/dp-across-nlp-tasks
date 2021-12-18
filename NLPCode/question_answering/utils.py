import time
import torch
from queue import Queue
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1) # get the index of the max probability
    correct = max_preds.eq(y)
    acc = correct.sum().cpu() / torch.FloatTensor([y.shape[0]])
    pr, rec, f1, _ = precision_recall_fscore_support(y.cpu().detach().numpy(), max_preds.cpu().detach().numpy(), average='micro')
    return acc, pr, rec, f1

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
            print(f'early stopping counter is set to {self.counter}')
        else:
            # else add element to queue and check if queue is full (if we should do early stopping)
            self.q.put(accuracy)
            self.counter += 1
            print(f'early stopping counter is set to {self.counter}')
            if self.q.full():
                # do early stopping
                return True