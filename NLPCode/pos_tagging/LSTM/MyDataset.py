# source: https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from glove_embed import GloveEmbedding
PAD_TAG = '<PAD>'
UNKNOWN_TAG = '<UNK>'
tag2idx = {PAD_TAG:0, 'NOUN':1, 'PUNCT':2, 'VERB':3, 'PRON':4, 'ADP':5, 'DET':6, 'PROPN':7, 'ADJ':8, 'AUX':9, 'ADV':10, 'CCONJ':11, 'PART':12, 'NUM':13, 'SCONJ':14, 'X':15, 'INTJ':16, 'SYM':17}
class POSDataset(Dataset):
    def __init__(self, path, EP, word2idx=None):
        # iterate through the file. The file has the structure Text \t UD Tag \t ptb tag
        # if we encounter an empty line, start a new entry
        # the resulting array is of shape [[text1, UDtag1, ptbtag1], ..., [text n, UDtag n, ptbtag n]]

        text_data = []
        tagUD_data = []
        tagPTB_data = []
        temp_text = []
        temp_tagUD = []
        temp_tagPTB  = []
        file = open(path, "r")
        lines = file.readlines()
        for line in lines:
            if line == "\n": # start a new entry
                if temp_text == []: # check for last empty lines
                    continue
                text_data.append(temp_text)
                tagUD_data.append(temp_tagUD)
                tagPTB_data.append(temp_tagPTB)
                temp_text = []
                temp_tagUD = []
                temp_tagPTB = []
            else:
                l = line.split("\t")
                temp_text.append(l[0].lower())
                temp_tagUD.append(l[1])
                temp_tagPTB.append(l[2].replace("\n", ""))
        data = pd.DataFrame(list(zip(text_data, tagUD_data, tagPTB_data)),
               columns =["Text", "udtags", "ptbtags"])

        self.text = data["Text"]
        self.udTag = data["udtags"]
        self.ptbTag = data["ptbtags"]

        
        if word2idx == None:
            self.text = data["Text"]
            self.udTag = data["udtags"]
            self.ptbTag = data["ptbtags"]
            # get glove embedding and vocab
            ge = GloveEmbedding(EP)
            self.embedding_matrix, self.vocab = ge.create_embedding(self.text, self.udTag, UNKNOWN_TAG,  word2idx=word2idx)
        else:
            self.vocab = word2idx

    def __len__(self):
            return len(self.udTag)
            
    def __getitem__(self, idx):
        # index_tokens = [self.word2idx[w.lower()] if w.lower() in self.word2idx else self.word2idx[UNKNOWN_TAG] for w in sentences]
        token_ids = [self.vocab[t] if t in self.vocab else self.vocab[UNKNOWN_TAG] for t in self.text[idx]]
        tag_ids = [tag2idx[t] for t in self.udTag[idx]]
        try:
            assert len(self.text[idx]) == len(token_ids) and len(tag_ids) == len(self.udTag[idx]) and len(tag_ids) == len(token_ids)
        except AssertionError:
            print("self.text[idx]", self.text[idx], "token_ids", token_ids, "tag_ids", tag_ids, "self.udTag[idx]", self.udTag[idx])
            assert len(self.text[idx]) == len(token_ids) and len(tag_ids) == len(self.udTag[idx]) and len(tag_ids) == len(token_ids)
        
        # sentences, tags, index_tags, index_tokens, seqlen = sample
        return self.text[idx], self.udTag[idx], tag_ids, token_ids, len(token_ids)