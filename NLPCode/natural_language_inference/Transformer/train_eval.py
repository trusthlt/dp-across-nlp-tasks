import torch
from utils import categorical_accuracy


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_f1 = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory

        sequence = batch.sequence
        attn_mask = batch.attention_mask
        token_type = batch.token_type
        label = batch.label

        predictions = model(sequence, attn_mask, token_type, device)
        loss = criterion(predictions, label)
        acc, pr, rec, f1 = categorical_accuracy(predictions, label)
    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += pr
        epoch_rec += rec
        epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_rec / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_f1 = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            sequence = batch.sequence
            attn_mask = batch.attention_mask
            token_type = batch.token_type
            labels = batch.label

            predictions = model(sequence, attn_mask, token_type, device)
            loss = criterion(predictions, labels)

            acc, pr, rec, f1  = categorical_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += pr
            epoch_rec += rec
            epoch_f1 += f1
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_rec / len(iterator), epoch_f1 / len(iterator)