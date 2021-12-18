import torchtext
from torch.utils import data
from torchtext.legacy import datasets
import spacy
import torch
import pandas as pd
from MyDataset import POSDataset, tag2idx
import numpy as np

def prep_data(EP, device):
    # load data from txt; save in pandas the text and tags and call the custom dataset class
    train_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_train.txt', EP)
    val_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_val.txt', EP, train_data.vocab)
    test_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_test.txt', EP, train_data.vocab)

    '''
    Number of training examples: 12543
    Number of validation examples: 2002
    Number of testing examples: 2077
    '''

    train_iter = data.DataLoader(dataset=train_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    
    val_iter = data.DataLoader(dataset=val_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    
    test_iter = data.DataLoader(dataset=test_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    return train_data.vocab, train_data.embedding_matrix, train_data.udTag, train_iter, val_iter, test_iter, EP.BATCH_SIZE/len(train_data)

# function called for the batch
def pad(batch):
    '''Pads to the longest sample'''
    """
    sentences, tags, index_tags, index_tokens, seqlen = batch 
    """
    f = lambda x: [sample[x] for sample in batch]

    words = f(0)
    tags = f(1)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    # create padding
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <PAD> 

    x = f(3, maxlen)
    y = f(2, maxlen)

    f = torch.LongTensor
    # sanity check
    # check if token and tags have the same length
    for arr1, arr2 in zip(words, tags):
        assert len(arr1) == len(arr2), f'words and tags have differend length {len(arr1)} == {len(arr2)}'

    # check if length of x y are == seqlens
    for arr1, arr2 in zip(x, y):
        assert len(arr1) == len(arr2) == np.max(seqlens), f'x and y and seqlens have differend length {len(arr1)} == {len(arr2)} == {np.max(seqlens)}'
    return words, f(x), tags, f(y), seqlens