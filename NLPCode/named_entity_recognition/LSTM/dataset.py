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
import spacy

TAGS = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}
idx2tag = {idx: tag for idx, tag in enumerate(TAGS)}
nlp = spacy.load("en_core_web_lg")
UNKNOWN_TAG = '<UNK>'

class NerDataset(data.Dataset):
    def __init__(self, fpath, EP, word2idx=None):
        """
        fpath: [train|valid|test].txt
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents_token, sents_from_glove, tags_li =[], [], [] # list of lists
        for entry in entries:           
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents_from_glove.append(words)
            tags_li.append(tags)
        # since a word can contain multiple spacy token one tag might be dublicated
        tokened_tag_list = []
        token_list = []


        # tokanize the sentences
        print("start tokenizing")
        for i, (sent, tags) in enumerate(zip(sents_from_glove, tags_li)):
            # tokanize words with spacy
            tokens = nlp(" ".join(sent))
            ###############
            # if all tokens of spacy are 1 to 1 mapping to words in list just keep them
            ###############
            if len(tokens) == len(tags):
                token_list.append([t.text for t in tokens])
                tokened_tag_list.append(tags)
            else:
                # go through words and find composition of tokens
                cur_tok_idx = 0 # the current position on the spacy tokenized array
                temp_token = [] # save the tokens of that sentence
                temp_tag = [] # save the tags of that sentence
                # iterate through all words and find their
                for i, w in enumerate(sent):
                    word = "" # the composition of tokens
                    counter_comp = 0 # the number of tokens that make up the word
                    start_indx = cur_tok_idx # start index of the word
                    found = False
                    for cur_tok in range(cur_tok_idx, len(tokens)):
                        word += tokens[cur_tok].text
                        counter_comp += 1
                        if word == w:
                            found = True
                            # found composition
                            temp_token.extend([t.text for t in tokens[start_indx : start_indx + counter_comp]])
                            temp_tag.extend(tags[i] for _ in range(counter_comp))
                            assert len(temp_token) == len(temp_tag), f"False statement len(temp_token) == len(temp_tag): {len(temp_token)} == {len(temp_tag)}"
                            cur_tok_idx = cur_tok + 1
                            break
                    assert found, f"couldn't find a composition for word: {w}, token: {tokens}"
                        
                tokened_tag_list.append(temp_tag)
                token_list.append(temp_token)
        print("end tokanizing")

        # in tags you can find all the gold labeld tags in sents you can find an array of tokens of type string
        self.sents = token_list        
        self.tags_li = tokened_tag_list

        # if train set init vocab for embedding layer
        word2idx_g = {}
        self.glove = {}
        if not word2idx == None:
            # if it is not the train set
            self.word2idx = word2idx
            # if it is the train set
        else:
            # source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
            # create a word to index using glove
            words = []
            idx = 0
            vectors = []

            # create a dic that given an word outputs the index of the glove
            with open(f'{EP.glove_dir}/glove.6B.100d.txt', 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx_g[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)

            # dict that given a word outputs a 100 dim vecor according to glove
            glove = {w: vectors[word2idx_g[w]] for w in words}
            
            # get the vocab size
            # save each token in a word set to get the length
            word_set = set() 
            for sent in self.sents:
                for tok in sent:
                    word_set.add(tok)

            # now create the pretrained embedding weights from glove and a word to index vector
            matrix_len = len(word_set)
            weights_matrix = np.zeros((matrix_len + 1, 100))
            words_found = 0
            self.word2idx = {}
            for i, word in enumerate(word_set):
                try: 
                    weights_matrix[i] = glove[word.lower()]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
                self.word2idx[word.lower()] = i
            # add an unknown token for future training sets
            self.word2idx[UNKNOWN_TAG] = len(self.word2idx)
            # init the matrix random for this unknown tag
            weights_matrix[-1] = np.random.normal(scale=0.6, size=(100, ))
            self.weights_matrix = weights_matrix
            print(f'found {words_found} out of {len(word_set)} tokens.')        

        
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sentences, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # take words and convert them to their index         
        index_tokens = [self.word2idx[w.lower()] if w.lower() in self.word2idx else self.word2idx[UNKNOWN_TAG] for w in sentences]

        # convert the tags to index
        index_tags = [tag2idx[t] for t in tags]

        seqlen = len(tags)
        if not len(sentences) == len(tags) == len(index_tags) == len(index_tokens) == seqlen:
            print(sentences)
            print(tags)
            print(index_tags)
            print(index_tokens)
        assert len(sentences) == len(tags) == len(index_tags) == len(index_tokens) == np.max(seqlen), \
            f"false statement: {len(sentences)} == {len(tags)} == {len(index_tags)} == {len(index_tokens)} == {np.max(seqlen)}"

        return sentences, tags, index_tags, index_tokens, seqlen


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
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad> 

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