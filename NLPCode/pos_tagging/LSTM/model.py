import torch
import torch.nn as nn
from opacus.layers import DPLSTM
from MyDataset import tag2idx

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, input_dim, output_dim, pad_idx, weights_matrix, EP):
        
        super().__init__()
        self.trainable_layers = nn.ModuleList([]) # for the privacy engine
        self.embedding, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix)

        if EP.privacy:
            self.lstm = DPLSTM(EP.EMBEDDING_DIM, 
                            EP.HIDDEN_DIM, 
                            num_layers = EP.N_LAYERS, 
                            bidirectional = EP.BIDIRECTIONAL,
                            dropout = EP.dropout if EP.N_LAYERS > 1 else 0)
        else:
            self.lstm = nn.LSTM(EP.EMBEDDING_DIM, 
                            EP.HIDDEN_DIM, 
                            num_layers = EP.N_LAYERS, 
                            bidirectional = EP.BIDIRECTIONAL,
                            dropout = EP.dropout if EP.N_LAYERS > 1 else 0)
        
        self.fc = nn.Linear(EP.HIDDEN_DIM * 2 if EP.BIDIRECTIONAL else EP.HIDDEN_DIM, output_dim)
        
        self.dropout = nn.Dropout(EP.dropout)

        self.trainable_layers.append(self.lstm)
        self.trainable_layers.append(self.fc)
        self.trainable_layers.append(self.dropout)
    
    # import pretrained embedding layer from train set
    def create_emb_layer(self, weights_matrix):
        num_embeddings, embedding_dim = len(weights_matrix), len(weights_matrix[0])
        emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=tag2idx['<PAD>'])
        emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
        emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
        
    def forward(self, text):

        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim] 

        #pass text through embedding layer
        embedded = self.dropout(embedded)
        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(1,0,2)
        #embedded = [sent len, batch size, emb dim]

        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        #output = [sent len,  batch size, hid dim * n directions] 

        outputs = outputs.permute(1,0,2)
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [ batch size, sent len, hid dim * n directions] 
        #hidden/cell = [n layers * n directions, batch size, hid dim]

        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        #predictions = [batch size, sent len, output dim]

        return predictions