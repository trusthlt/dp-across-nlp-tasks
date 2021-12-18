import time
import torch
from queue import Queue
import numpy as np

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc



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