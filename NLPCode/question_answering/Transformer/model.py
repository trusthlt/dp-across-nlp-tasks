import torch
import torch.nn as nn
from pytorch_transformers import BertForQuestionAnswering
from tuning_structs import TrainingBERT
from opacus.layers import DPLSTM
from transformers import AutoModelForQuestionAnswering
print_layers = False

class BERTQAModel(nn.Module):
    def __init__(self, EP):
        super().__init__()
                if EP.use_BERT:
            self.bertQA = BertForQuestionAnswering.from_pretrained(EP.bert_model_type)
        else:
            self.bertQA = AutoModelForQuestionAnswering.from_pretrained(EP.bert_model_type)
            
        self.bert = self.bertQA.bert # for finetuning only freez/unfreez bert of bert QA
        self.trainable_layers = nn.ModuleList([])
        self.use_rnn = EP.use_rnn
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        if EP.tuning.training_bert == TrainingBERT.Freeze_Total:
            for p in self.bert.parameters():
                p.requires_grad = False
                if not self.use_rnn: # if we use rnn than 
                    self.trainable_layers.append(self.bertQA.qa_outputs)
        elif EP.tuning.training_bert < 0 or EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
            for name, param in self.bert.named_parameters(recurse=True):
                if name in EP.tuning.freeze_array:
                    param.requires_grad = False
            if EP.tuning.training_bert == TrainingBERT.OnlyEmbeds:
                #self.trainable_layers.append(self.bert.pooler)
                self.trainable_layers.append(self.bert.encoder)
                self.trainable_layers.append(self.bertQA.qa_outputs)
            else:      
                for layer in self.bert.encoder.layer[EP.tuning.training_bert:]:
                    self.trainable_layers.append(layer)
                self.trainable_layers.append(self.bertQA.qa_outputs)
        else:
            self.trainable_layers.append(self.bertQA)

        # print bert layers
        if print_layers:
            f = open('<text file where to print the layers>', 'w')
            for name, param in self.bertQA.named_parameters(recurse=True):
                    f.write(f'{name}\t|\t{param.requires_grad}\n')
            f.write(f'\n\n{self.trainable_layers}')
            f.close()

        # add LSTM
        if self.use_rnn:
            if EP.tuning.privacy:
                self.rnn = DPLSTM(embedding_dim,
                                EP.HIDDEN_DIM,
                                num_layers=EP.N_LAYERS,
                                bidirectional=EP.BIDIRECTIONAL,
                                batch_first=True,
                                dropout=0 if EP.N_LAYERS < 2 else EP.DROPOUT)
            else:
                self.rnn = nn.LSTM(embedding_dim,
                                EP.HIDDEN_DIM,
                                num_layers=EP.N_LAYERS,
                                bidirectional=EP.BIDIRECTIONAL,
                                batch_first=True,
                                dropout=0 if EP.N_LAYERS < 2 else EP.DROPOUT)
            
            
            self.out = nn.Linear(EP.HIDDEN_DIM * 2 if EP.BIDIRECTIONAL else EP.HIDDEN_DIM, EP.OUTPUT_DIM)
            self.dropout = nn.Dropout(EP.DROPOUT)
            # append trainable layers for DP   
            self.trainable_layers.append(self.rnn)
            self.trainable_layers.append(self.out)
            self.trainable_layers.append(self.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        if self.use_rnn:
            # use bert model of bertQA to LSTM
            # input_ids = [Batch Size, Seq Length]
            embedded = self.bertQA.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            # last_hidden_state = embedded = [Batch Size, Seq Length, Hidden Dim]
            
            outputs, (hidden, cell) = self.rnn(embedded)
            #outputs holds the backward and forward hidden states in the final layer

            logits = self.out(self.dropout(outputs))
            # logits = [Batch Size, Seq Length, Output Dim]

            # this code is from huggingface
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            return logits, start_logits, end_logits
        else:
            return self.bertQA(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                            start_positions=start_positions, end_positions=end_positions)