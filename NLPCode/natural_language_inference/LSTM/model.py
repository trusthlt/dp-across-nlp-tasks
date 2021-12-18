import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPLSTM

class NLIBiLSTM(nn.Module):
    def __init__(self, input_dim, EP, output_dim, pad_idx):
        
        super().__init__()
        self.trainable_layers = nn.ModuleList([])
                                
        self.embedding = nn.Embedding(input_dim, EP.EMBEDDING_DIM, padding_idx = pad_idx)
        
        #self.translation = nn.Linear(EP.EMBEDDING_DIM, EP.HIDDEN_DIM)
        if EP.privacy:
            self.lstm = DPLSTM(EP.EMBEDDING_DIM, # EP.EMBEDDING_DIM davor hidden dim
                              EP.HIDDEN_DIM,
                              num_layers=EP.N_LSTM_LAYERS,
                              bidirectional=True,
                              dropout=EP.DROPOUT if EP.N_LSTM_LAYERS > 1 else 0)
        else:
            self.lstm = nn.LSTM(EP.EMBEDDING_DIM, 
                                EP.HIDDEN_DIM, 
                                num_layers = EP.N_LSTM_LAYERS, 
                                bidirectional = True, 
                                dropout=EP.DROPOUT if EP.N_LSTM_LAYERS > 1 else 0)
        
        fc_dim = EP.HIDDEN_DIM * 2
        
        fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(EP.N_FC_LAYERS)]
        
        self.fcs = nn.ModuleList(fcs)
        
        self.fc_out = nn.Linear(fc_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(EP.DROPOUT)

        # add layers for privacy so no embedding is included
        #self.trainable_layers.append(self.translation)
        self.trainable_layers.append(self.lstm)
        self.trainable_layers.append(self.fcs)
        self.trainable_layers.append(self.fc_out)
        self.trainable_layers.append(self.dropout)
        
    def forward(self, prem, hypo):

        prem_seq_len, batch_size = prem.shape
        hypo_seq_len, _ = hypo.shape
        
        #prem = [prem sent len, batch size]
        #hypo = [hypo sent len, batch size]
        
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)
        
        #embedded_prem = [prem sent len, batch size, embedding dim]
        #embedded_hypo = [hypo sent len, batch size, embedding dim]
        
        '''translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))'''
        
        #translated_prem = [prem sent len, batch size, hidden dim]
        #translated_hypo = [hypo sent len, batch size, hidden dim]
        
        outputs_prem, (hidden_prem, cell_prem) = self.lstm(embedded_prem)#self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(embedded_hypo)#self.lstm(translated_hypo)

        #outputs_x = [sent len, batch size, n directions * hid dim]
        #hidden_x = [n layers * n directions, batch size, hid dim]
        #cell_x = [n layers * n directions, batch size, hid dim]
        
        hidden_prem = torch.cat((hidden_prem[-1], hidden_prem[-2]), dim=-1)
        hidden_hypo = torch.cat((hidden_hypo[-1], hidden_hypo[-2]), dim=-1)
        
        #hidden_x = [batch size, fc dim]

        hidden = torch.cat((hidden_prem, hidden_hypo), dim=1)

        #hidden = [batch size, fc dim * 2]
            
        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden)
        
        #prediction = [batch size, output dim]
        
        return prediction