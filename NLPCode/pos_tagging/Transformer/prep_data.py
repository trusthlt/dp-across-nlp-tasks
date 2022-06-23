import functools
from transformers import BertTokenizer
from torchtext.legacy import data
from torchtext.legacy import datasets
from MyDataset import POSDataset
import torch
import numpy as np

def prep_data(device, EP):
    if EP.use_BERT:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")
        
    train_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_train.txt', EP, tokenizer)
    val_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_val.txt', EP, tokenizer)
    test_data = POSDataset(f'{EP.dataset_dir}/{EP.dataset_choice}_test.txt', EP, tokenizer)


    train_iterator = torch.utils.data.DataLoader(dataset=train_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    valid_iterator = torch.utils.data.DataLoader(dataset=val_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    test_iterator = torch.utils.data.DataLoader(dataset=test_data,
                                 batch_size=EP.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)    

    return train_iterator, valid_iterator, test_iterator, EP.BATCH_SIZE/len(train_data)


# helper functions
def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens


def pad(batch):
    '''Pads to the longest sample'''
    """
    words, x, tags, y, seqlen
    ('[CLS] 6. Ryo 38:34.682 [SEP]', 
    [101, 127, 119, 155, 7490, 3383, 131, 3236, 119, 5599, 1477, 102], 
    '<PAD> O B-PER O <PAD>', 
    [0, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0], 
    12)
    """
    f = lambda x: [sample[x] for sample in batch]

    words = f(0)
    tags = f(2)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    # create padding
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad> 

    x = f(1, maxlen)
    y = f(-2, maxlen)

    # create attention_mask
    att_mask = torch.tensor([[1 if el > 0 else 0 for el in sample] for sample in y], dtype=torch.float32)

    f = torch.LongTensor

    return words, f(x), tags, f(y), seqlens, att_mask