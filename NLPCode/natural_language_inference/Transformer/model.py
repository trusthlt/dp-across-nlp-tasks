from transformers import BertModel
import torch.nn as nn
from opacus.layers import DPLSTM
from tuning_structs import TrainingBERT
import torch

class BERTNLIModel(nn.Module):
    def __init__(self, EP, output_dim):
        super().__init__()
        self.trainable_layers = nn.ModuleList([])
        self.EP = EP
        self.use_rnn = EP.use_rnn

        self.bert = BertModel.from_pretrained(EP.bert_model_type)
        
        # finetune bert
        if EP.tuning.training_bert == TrainingBERT.Freeze_Total:
            print("no additional bert training")
            for p in self.bert.parameters():
                p.requires_grad = False

        elif EP.tuning.training_bert < 0 or EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
            
            for name, param in self.bert.named_parameters(recurse=True):
                    if name in EP.tuning.freeze_array:
                        param.requires_grad = False

            if EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
                print(f"only freez embeds")
                # append trainable layers for DP (Everything but embeds)           
                self.trainable_layers.append(self.bert.pooler)
                self.trainable_layers.append(self.bert.encoder)
            else:
                # append trainable layers for DP        
                for layer in self.bert.encoder.layer[EP.tuning.training_bert:]:
                    self.trainable_layers.append(layer)
        else:
            print("train whole bert")
            # append trainable layers for DP   
            self.trainable_layers.append(self.bert)
        
        # additional LSTM?
        if self.use_rnn:
            # do privacy
            if EP.privacy:
                self.rnn = DPLSTM(bidirectional=EP.BIDIRECTIONAL, num_layers=EP.N_LAYERS, 
                                     input_size=EP.INPUT_SIZE, hidden_size=EP.HIDDEN_DIM , batch_first=EP.BATCH_FIRST)
                self.trainable_layers.append(self.rnn)
            else:
                self.rnn = nn.LSTM(bidirectional=EP.BIDIRECTIONAL, num_layers=EP.N_LAYERS, 
                                   input_size=EP.INPUT_SIZE, hidden_size=EP.HIDDEN_DIM , batch_first=EP.BATCH_FIRST)
                self.trainable_layers.append(self.rnn)
            self.out_lstm = nn.Linear(EP.HIDDEN_DIM * 2 if EP.BIDIRECTIONAL else EP.HIDDEN_DIM, output_dim)
            self.dropout = nn.Dropout(EP.DROPOUT)
        else:
            embedding_dim = self.bert.config.to_dict()['hidden_size']
            self.out = nn.Linear(embedding_dim, output_dim)
            self.trainable_layers.append(self.out)
            print(self.trainable_layers)

    def forward(self, sequence, attn_mask, token_type, device):
        if self.use_rnn:
            embedded = self.bert(input_ids = sequence, attention_mask = attn_mask, token_type_ids = token_type)[0]
            # embedded = [batch size, sent len, emb dim]
            _, (hidden, cell) = self.rnn(embedded) # hidden = [n layers * n directions, batch size, emb dim]
            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).to(device))
            else:
                hidden = self.dropout(hidden[-1, :, :])
            # hidden = [batch size, hid dim]
            output = self.out_lstm(hidden) # output = [batch size, out dim (3)]
            return output
        else:
            embedded = self.bert(input_ids = sequence, attention_mask = attn_mask, token_type_ids= token_type)[1] # (batch_size, hidden_size)
            output = self.out(embedded) # (batch_size, out dim (3))
            return output