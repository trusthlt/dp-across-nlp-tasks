# source: https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

PAD_TAG = '<PAD>'
CLS_TAG = '[CLS]'
UNKNOWN_TAG = '<UNK>'
tag2idx = {PAD_TAG:0, 'NOUN':1, 'PUNCT':2, 'VERB':3, 'PRON':4, 'ADP':5, 'DET':6, 'PROPN':7, 'ADJ':8, 'AUX':9, 'ADV':10, 'CCONJ':11, 'PART':12, 'NUM':13, 'SCONJ':14, 'X':15, 'INTJ':16, 'SYM':17}

class POSDataset(Dataset):
    def __init__(self, path, EP, tokenizer, word2idx=None):
        # iterate through the file. The file has the structure Text \t UD Tag \t ptb tag
        # if we encounter an empty line, start a new entry
        # the resulting array is of shape [[text1, UDtag1, ptbtag1], ..., [text n, UDtag n, ptbtag n]]
        self.tokenizer = tokenizer
        text_data = []
        tagUD_data = []
        tagPTB_data = []
        temp_text = [CLS_TAG]
        temp_tagUD = [PAD_TAG]
        temp_tagPTB  = [PAD_TAG]
        file = open(path, "r")
        lines = file.readlines()
        for line in lines:
            if line == "\n": # start a new entry
                if temp_text == [CLS_TAG]: # check for last empty lines
                    continue
                text_data.append(temp_text)
                tagUD_data.append(temp_tagUD)
                tagPTB_data.append(temp_tagPTB)
                temp_text = [CLS_TAG]
                temp_tagUD = [PAD_TAG]
                temp_tagPTB = [PAD_TAG]
            else:
                l = line.split("\t")
                # add the CSL token to the beginning of each sentence and a pad token for the tags
                temp_text.append(l[0].lower())
                temp_tagUD.append(l[1])
                temp_tagPTB.append(l[2].replace("\n", ""))
        data = pd.DataFrame(list(zip(text_data, tagUD_data, tagPTB_data)),
               columns =["Text", "udtags", "ptbtags"])

        self.text = data["Text"]
        self.udTag = data["udtags"]
        self.ptbTag = data["ptbtags"]


    def __len__(self):
            return len(self.udTag)
            
    def __getitem__(self, idx):

        words, tags = self.text[idx], self.udTag[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        for w, t in zip(words, tags):
            # get token and their ids accoding to bert tokanizer
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            
            assert len(tokens) == len(xx), f"tokens not equal ids:{len(tokens)} == {len(xx)}"


            # get tags and their ids
            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision -> if a token consists of more than one tag, make only the first
                                                     # tag the one that counts and the rest a pad tag so it will be ignored

            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            y.extend(yy)

        assert len(x)==len(y), f"len(x)={len(x)}, len(y)={len(y)}"

        # seqlen
        seqlen = len(y)

        # to string
        #words = " ".join(words)
        #stags = " ".join(tags)

        return words, x, tags, y, seqlen