import torch
import torch.nn as nn
from opacus.layers import DPLSTM
from tuning_structs import TrainingBERT, Tuning
import pdb

print_layers = False
class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, tuning_dict: Tuning, use_rnn):
        super().__init__()

        self.bert = bert
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.use_rnn = use_rnn
        self.trainable_layers = nn.ModuleList([]) # for the privacy engine

        ######################################
        # freeze bert according to dictionary
        ######################################
        
        if tuning_dict.training_bert == TrainingBERT.Freeze_Total:
            print("no additional bert training")
            for p in self.bert.parameters():
                p.requires_grad = False

        elif tuning_dict.training_bert < 0 or tuning_dict.training_bert == TrainingBERT.OnlyEmbeds:
            
            for name, param in self.bert.named_parameters(recurse=True):
                    if name in tuning_dict.freeze_array:
                        param.requires_grad = False

            if tuning_dict.training_bert == TrainingBERT.OnlyEmbeds:
                print(f"only freez embeds")
                # append trainable layers for DP (Everything but embeds)           
                self.trainable_layers.append(self.bert.pooler)
                self.trainable_layers.append(self.bert.encoder)
            else:
                print(f"only train last {abs(tuning_dict.training_bert)} layers of bert")
                # append trainable layers for DP        
                for layer in self.bert.encoder.layer[tuning_dict.training_bert:]:        
                    self.trainable_layers.append(layer)
        else:
            # (if tuning_dict.training_bert == TrainingBERT.Train_All) freeze nothing
            print("train whole bert")
            # append trainable layers for DP   
            self.trainable_layers.append(self.bert)
        # count trainable params
        print(f'The number of trainable params is: {sum([param.requires_grad for param in self.bert.parameters()])}')

        
        if print_layers:
            fi = open("<text file where to print the layer>", "w")
            for name, param in self.bert.named_parameters(recurse=True):
                fi.write(f'{name}\t|\t{param.requires_grad}\n')
            fi.close()



        if self.use_rnn:
            ######################################
            # handle privacy preserving
            ######################################
            if tuning_dict.privacy:
                self.rnn = DPLSTM(self.embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                batch_first=True,
                                dropout=0 if n_layers < 2 else dropout)
            else:
                self.rnn = nn.LSTM(self.embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                batch_first=True,
                                dropout=0 if n_layers < 2 else dropout)
            
            
            self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            # append trainable layers for DP   
            self.trainable_layers.append(self.rnn)
            self.trainable_layers.append(self.out)
            self.trainable_layers.append(self.dropout)
        else:
            # if no use of rnn
            self.linear = nn.Linear(self.embedding_dim, output_dim)
            # append trainable layers for DP
            self.trainable_layers.append(self.linear)

            for n, p in self.linear.named_parameters():
                print(f'{n} : {p.requires_grad} and {hasattr(p, "grad_sample")}')

    def forward(self, text, device):

        # text = [batch size, sent len]

        if self.use_rnn:
            embedded = self.bert(text)[0]
            # embedded = [batch size, sent len, emb dim]
            _, (hidden, cell) = self.rnn(embedded)
            # hidden = [n layers * n directions, batch size, emb dim]
            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).to(device))
            else:
                hidden = self.dropout(hidden[-1, :, :])
            # hidden = [batch size, hid dim]

            output = self.out(hidden)
            # output = [batch size, out dim]

            return output
        
        else:
            embedded = self.bert(text)[1]
            # embedding dim -> (batchsize, embedding dim)
            out = self.linear(embedded) #-> (embed dim, output dim)
            return out
            # pooler output -> (index one instead index 0) bert model     