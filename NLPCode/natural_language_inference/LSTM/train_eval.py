import torch
from utils import categorical_accuracy

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0
    
    model.train()
    
    for batch in iterator:
        
        prem = batch.premise
        hypo = batch.hypothesis
        labels = batch.label
        
        optimizer.zero_grad()
        
        #prem = [prem sent len, batch size]
        #hypo = [hypo sent len, batch size]
        
        predictions = model(prem, hypo)
        
        #predictions = [batch size, output dim]
        #labels = [batch size]
        
        loss = criterion(predictions, labels)
                
        acc, pr_epoch, rec_epoch, f1_epoch = categorical_accuracy(predictions, labels)

        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        precision += pr_epoch
        recall += rec_epoch
        f1 += f1_epoch
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision / len(iterator), recall / len(iterator), f1 / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
    precision = 0
    recall = 0
    f1 = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            prem = batch.premise
            hypo = batch.hypothesis
            labels = batch.label
                        
            predictions = model(prem, hypo)
            
            loss = criterion(predictions, labels)
                
            acc, pr_epoch, rec_epoch, f1_epoch = categorical_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            precision += pr_epoch
            recall += rec_epoch
            f1 += f1_epoch
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision / len(iterator), recall / len(iterator), f1 / len(iterator)