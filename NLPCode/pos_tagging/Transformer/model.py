import torch
import torch.nn as nn
from transformers import BertModel
from tuning_structs import TrainingBERT
from opacus.layers import DPLSTM

print_layers = False
class BERTPoSTagger(nn.Module):
    def __init__(self, output_dim, EP):
        
        super().__init__()
        self.trainable_layers = nn.ModuleList([]) # for the privacy engine
        self.use_rnn = EP.use_rnn
        if EP.use_BERT:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")
        embedding_dim = bert.config.to_dict()['hidden_size']


        # finetune bert
        if EP.tuning.training_bert == TrainingBERT.Freeze_Total:
            for p in self.bert.parameters():
                p.requires_grad = False
        elif EP.tuning.training_bert < 0 or EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
            for name, param in self.bert.named_parameters(recurse=True):
                if name in EP.tuning.freeze_array:
                    param.requires_grad = False
            if EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
                self.trainable_layers.append(self.bert.encoder)
            else:      
                for layer in self.bert.encoder.layer[EP.tuning.training_bert:]:
                    self.trainable_layers.append(layer)
                
        else:
            for name, param in self.bert.named_parameters(recurse=True):
                if name in EP.tuning.freeze_array:
                    param.requires_grad = False
            self.trainable_layers.append(self.bert.embeddings)
            self.trainable_layers.append(self.bert.encoder)
            # pooler is incompatible

        # print bert layers
        if print_layers:
            f = open(f'{EP.output_dir}/layers.txt', "w")
            for name, param in self.bert.named_parameters(recurse=True):
                f.write(f'{name} | {param.requires_grad}\n')
            f.write("\n\n")
            f.write(str(self.trainable_layers))
            f.close
            
        # add LSTM
        if self.use_rnn:
            if EP.tuning.privacy:
                self.rnn = DPLSTM(embedding_dim,
                                EP.HIDDEN_DIM,
                                num_layers=EP.N_LAYERS,
                                bidirectional=EP.BIDIRECTIONAL,
                                batch_first=True,
                                dropout=0 if EP.N_LAYERS < 2 else EP.dropout)
            else:
                self.rnn = nn.LSTM(embedding_dim,
                                EP.HIDDEN_DIM,
                                num_layers=EP.N_LAYERS,
                                bidirectional=EP.BIDIRECTIONAL,
                                batch_first=True,
                                dropout=0 if EP.N_LAYERS < 2 else EP.dropout)
            
            
            self.out = nn.Linear(EP.HIDDEN_DIM * 2 if EP.BIDIRECTIONAL else EP.HIDDEN_DIM, output_dim)
            self.dropout = nn.Dropout(EP.dropout)
            # append trainable layers for DP   
            self.trainable_layers.append(self.rnn)
            self.trainable_layers.append(self.out)
            self.trainable_layers.append(self.dropout)
        else:
            self.fc = nn.Linear(embedding_dim, output_dim)
            self.dropout = nn.Dropout(EP.dropout)
            self.trainable_layers.append(self.fc)
            self.trainable_layers.append(self.dropout)
        
    def forward(self, text, att_mask):

        #print("text", text.size())
        #text = [batch size, sent len]
        if self.use_rnn:

            embedded = self.bert(text, att_mask)[0]
            # embedded = [batch size, sent len, emb dim]

            outputs, (hidden, cell) = self.rnn(embedded)
            #output = [batch size, sent len, hid dim * n directions]
            #hidden/cell = [n layers * n directions, batch size, hid dim]
            
            #we use our outputs to make a prediction of what the tag should be
            predictions = self.out(self.dropout(outputs))
            #predictions = [batch size, sent len, output dim]
            
            return predictions
        else:
            embedded = self.dropout(self.bert(text)[0])
            #embedded = [batch size, seq len, emb dim]  
            predictions = self.fc(self.dropout(embedded))
            #predictions = [batch size, sent len, output dim]
            
        return predictions