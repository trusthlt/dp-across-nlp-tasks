import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification
from opacus.layers import DPLSTM
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tuning_structs import Tuning, TrainingBERT, Privacy
from dataset import tag2idx


class Net(nn.Module):
    def __init__(self, tag_size, weights_matrix, device, EP):
        super().__init__()
        self.EP = EP
        self.device = device

        self.word_embeds, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix)

        if EP.privacy:
            self.rnn = DPLSTM(bidirectional=EP.BIDIRECTIONAL, num_layers=EP.N_LAYERS, 
                                    input_size=embedding_dim, hidden_size=EP.HIDDEN_DIM , batch_first=EP.BATCH_FIRST)
        else:
            self.rnn = nn.LSTM(bidirectional=EP.BIDIRECTIONAL, num_layers=EP.N_LAYERS, 
                                input_size=embedding_dim, hidden_size=EP.HIDDEN_DIM , batch_first=EP.BATCH_FIRST)
        self.linear = nn.Linear(EP.INPUT_SIZE, tag_size)
    
    # import pretrained embedding layer from train set
    def create_emb_layer(self, weights_matrix, not_trainable=False):
        num_embeddings, embedding_dim = len(weights_matrix), len(weights_matrix[0])
        emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=tag2idx['<PAD>'])
        emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
        if not_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
        

    def forward(self, x, y, words):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, TAGS)
        '''
        
        x = x.to(self.device)
        y = y.to(self.device)

        embedded = self.word_embeds(x) # (SEQ len, batch size, embedd dim)

        hidden, _ = self.rnn(embedded) # (seq len, batch size, input size)

        logits = self.linear(hidden) # (seq len, batch size, tag size)

        seq = torch.transpose(logits.argmax(-1), 0, 1)  # (batch size, seq length)

        return logits, y, seq
