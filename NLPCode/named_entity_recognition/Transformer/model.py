import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification, BertConfig
from opacus.layers import DPLSTM
from dataset import TAGS
import sys
import os
from transformers import AutoModel
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tuning_structs import Tuning, TrainingBERT, Privacy
print_param = False

class Net(nn.Module):
    def __init__(self, tag_size, EP):
        super().__init__()
        self.EP = EP
        self.use_rnn = EP.use_rnn
        # the default output of logis is 2 set it to the tag_size length
        # depending on if we use the rnn we need to activate the hidden states
        if not self.use_rnn:
            if EP.use_BERT:
                self.bert = BertForTokenClassification.from_pretrained( EP.bert_model_type,
                                                                        num_labels=tag_size,
                                                                        output_attentions = False,
                                                                        output_hidden_states = False
                                                                        )
            else:
                self.bert = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")                

            # when using the token classification the bert layers have a bert. in front of their names
            use_BFTC = "bert."
        else:
            if EP.use_BERT:
                self.bert = BertModel.from_pretrained(EP.bert_model_type)
            else:
                self.bert = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")
            use_BFTC = ""
        
        if not EP.use_BERT:
            self.linear = nn.Linear(EP.INPUT_SIZE, tag_size)
        
        self.trainable_layers = nn.ModuleList([])
        self.EP = EP

        # finetune bert:
        if EP.tuning.training_bert == TrainingBERT.Freeze_Total:
            print("no additional bert training")
            for name, param in self.bert.named_parameters(recurse=True):
                if name.replace(use_BFTC, "") in EP.tuning.freeze_array:
                    param.requires_grad = False
            if not self.use_rnn:
                if EP.use_BERT:
                    self.trainable_layers.append(self.bert.classifier)
                    #self.trainable_layers.append(self.bert.dropout)
                else:
                    self.trainable_layers.append(self.linear)
                    #self.trainable_layers.append(self.dropout)

        elif EP.tuning.training_bert < 0:
            print(f"only train last {abs(EP.tuning.training_bert)} layers of bert")
            for name, param in self.bert.named_parameters(recurse=True):
                if name.replace(use_BFTC, "") in EP.tuning.freeze_array:
                    param.requires_grad = False

            # append trainable layers for DP 
            if not self.use_rnn and EP.use_BERT:
                # if bert for token classification is used     
                for layer in self.bert.bert.encoder.layer[EP.tuning.training_bert:]:        
                    self.trainable_layers.append(layer)
                # add the linear layer to the trainable layers
                self.trainable_layers.append(self.bert.classifier)
                self.trainable_layers.append(self.bert.dropout)
            else:
                # if normal bert model is used
                for layer in self.bert.encoder.layer[EP.tuning.training_bert:]:        
                    self.trainable_layers.append(layer)
                if not EP.use_BERT:
                    self.trainable_layers.append(self.linear)
            
        elif EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
            if not self.use_rnn and EP.use_BERT:
                # freeze embedding
                self.bert.bert.embeddings.word_embeddings.weight.requires_grad = False
                self.bert.bert.embeddings.position_embeddings.weight.requires_grad = False
                self.bert.bert.embeddings.token_type_embeddings.weight.requires_grad = False
                self.bert.bert.embeddings.LayerNorm.weight.requires_grad = False
                self.bert.bert.embeddings.LayerNorm.bias.requires_grad = False
                self.bert.bert.embeddings.dropout.requires_grad = False
                # add rest for DP
                self.trainable_layers.append(self.bert.bert.encoder)
                self.trainable_layers.append(self.bert.bert.pooler)
                self.trainable_layers.append(self.bert.classifier)
                self.trainable_layers.append(self.bert.dropout)
            else:
                # freeze embedding
                self.bert.embeddings.word_embeddings.weight.requires_grad = False
                self.bert.embeddings.position_embeddings.weight.requires_grad = False
                self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
                self.bert.embeddings.LayerNorm.weight.requires_grad = False
                self.bert.embeddings.LayerNorm.bias.requires_grad = False
                self.bert.embeddings.dropout.requires_grad = False
                self.bert.pooler.dense.requires_grad = False
                self.bert.pooler.activation.requires_grad = False
                # add rest for DP
                self.trainable_layers.append(self.bert.encoder)
                #self.trainable_layers.append(self.bert.pooler)

                if not EP.use_BERT:
                    self.trainable_layers.append(self.linear)

        else:
            # freeze nothing
            print("train whole bert")
            # append trainable layers for DP   
            self.trainable_layers.append(self.bert)
            if not EP.use_BERT:
                self.trainable_layers.append(self.linear)

        # use LSTM?
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
            if EP.use_BERT:
                self.linear = nn.Linear(EP.INPUT_SIZE, tag_size)
                self.trainable_layers.append(self.linear)
        

    def forward(self, x, y, attention_mask):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, tag_size)
        '''
        
        # check what x, y etc. look like, does it make sence (semantically)
        if self.use_rnn and not self.EP.use_BERT:
            bert_out = self.bert(x, attention_mask=attention_mask).last_hidden_state # (batch_size, sequence_length, hidden_size)
            print('bert_out', bert_out.size())
            hidden, _ = self.rnn(bert_out) # hidden (BATCH_SIZE, SEQ lens, INPUT_SIZE)
            print('hidden', hidden.size())
            logits = self.linear(hidden) # (BATCH_SIZE, SEQ LENGHT, tag_size)
            print('logits', logits.size())
            seq = logits.argmax(-1)
            print('seq', seq.size())
            return logits, y, seq
            
        else:
            if self.EP.use_BERT:
                out = self.bert(x, attention_mask=attention_mask, labels=y)
                logits = out.logits # hidden (BATCH_SIZE, SEQ lens, INPUT_SIZE)
                loss = out.loss
                seq = logits.argmax(-1)
                return logits, y, seq, loss
            else:
                bert_out = self.bert(x, attention_mask=attention_mask).last_hidden_state # (batch_size, sequence_length, hidden_size)
                #dropout_out = self.dropout(bert_out) # (batch_size, sequence_length, hidden_size)
                logits = self.linear(bert_out) # (batch_size, sequence_length, tag size)
                seq = logits.argmax(-1) # (batch_size, sequence_length)
                return logits, y, seq

