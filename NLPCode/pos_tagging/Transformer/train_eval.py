import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import categorical_accuracy
import torch

def train(model, iterator, optimizer, criterion, tag_pad_idx, device):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_F1 = 0
    
    model.train()
    
    for batch in iterator:
        text, x, tags, y, seqlens, att_mask = batch

        x = x.to(device)
        y = y.to(device)
        att_mask = att_mask.to(device)
                
        optimizer.zero_grad()
        
        #x = [sent len, batch size]
        
        predictions = model(x, att_mask)
        
        #predictions = [sent len, batch size, output dim]
        #y = [sent len, batch size]
        
        #predictions = predictions.view(-1, predictions.shape[-1])
        sent_batch = predictions.size()[0] * predictions.size()[1] # sent len * batch size
        out_dim = predictions.size()[2] # output dim
        predictions = torch.reshape(predictions, (sent_batch, out_dim))

        y = y.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #y = [sent len * batch size]
        
        loss = criterion(predictions, y)
                
        acc, pr, rec, f1  = categorical_accuracy(predictions, y, tag_pad_idx, device)
        print(acc)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_precision += pr
        epoch_recall += rec
        epoch_F1 += f1
        
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_F1 / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx, device):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_F1 = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, x, tags, y, seqlens, att_mask = batch

            x = x.to(device)
            y = y.to(device)
            att_mask = att_mask.to(device)
            
            predictions = model(x, att_mask)
            
            #predictions = predictions.view(-1, predictions.shape[-1])
            sent_batch = predictions.size()[0] * predictions.size()[1] # sent len * batch size
            out_dim = predictions.size()[2] # output dim
            predictions = torch.reshape(predictions, (sent_batch, out_dim))
            y = y.view(-1)
            
            loss = criterion(predictions, y)
            
            acc, pr, rec, f1  = categorical_accuracy(predictions, y, tag_pad_idx, device)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_precision += pr
            epoch_recall += rec
            epoch_F1 += f1
            
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_F1 / len(iterator)