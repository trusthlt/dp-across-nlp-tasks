'''
An entry or sent looks like ...
SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O
Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
# We will concentrate on four types of named entities: 
#persons (PER), locations (LOC), organizations (ORG) and names of miscellaneous (MISC) entities that do not belong to the previous three groups.

import numpy as np
import torch
from torch.utils import data
from transformers import BertTokenizer

TAGS_wikiann = ('<PAD>', "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
tag2idx = {tag: idx for idx, tag in enumerate(TAGS_wikiann)}
idx2tag = {idx: tag for idx, tag in enumerate(TAGS_wikiann)}

class NerDatasetWikiann(data.Dataset):
    def __init__(self, fpath, EP):
        self.tokenizer = BertTokenizer.from_pretrained(EP.bert_model_type, do_lower_case=False)
        """
        fpath: [train|valid|test].txt
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], [] # list of lists
        for entry in entries:            
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li
    
    

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            # get token and their ids accoding to bert tokanizer
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            
            assert len(tokens) == len(xx), f"tokens not equal ids:{len(tokens)} == {len(xx)}"

            is_head = [1] + [0]*(len(tokens) - 1)

            # get tags and their ids
            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision -> if a token consists of more than one tag, make only the first
                                                     # tag the one that counts and the rest a pad tag so it will be ignored

            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            
        # truncate since the max length of BERT is 512
        if len(x) >=512:
            x = x[:512]
            y = y[:512]
            is_heads = is_heads[:512]

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad_Wikiann(batch):
    '''Pads to the longest sample'''
    """
    ('[CLS] 6. Ryo 38:34.682 [SEP]', 
    [101, 127, 119, 155, 7490, 3383, 131, 3236, 119, 5599, 1477, 102], 
    [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 
    '<PAD> O B-PER O <PAD>', 
    [0, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0], 
    12)
    """
    f = lambda x: [sample[x] for sample in batch]

    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    # create padding
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad> 

    x = f(1, maxlen)
    y = f(-2, maxlen)

    # create attention_mask
    att_mask = torch.tensor([[1 if el > 0 else 0 for el in sample] for sample in x], dtype=torch.float32)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens, att_mask